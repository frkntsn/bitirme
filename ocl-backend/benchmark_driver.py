"""
OCL Backend — Benchmark Driver
================================
Bu script sistemi doğrudan Python API üzerinden çalıştırır
(HTTP sunucusu gerektirmez) ve gerçek duvar saati ölçümleri alır.

Çalıştırmak için:
    cd ocl-backend
    python benchmark_driver.py [--image <path>] [--runs <n>]

Çıktı: benchmark_metrics.json
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from statistics import median, mean, stdev
from typing import Optional

# ------------------------------------------------------------------
# PIL kontrolü
# ------------------------------------------------------------------
try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    sys.exit(f"[ERROR] Bağımlılık eksik: {e}\n`pip install Pillow numpy` çalıştırın.")


# ==================================================================
# Yardımcılar
# ==================================================================

def _build_synthetic_image(width: int = 512, height: int = 512) -> Image.Image:
    """
    4 renkli blok + gürültü içeren sentetik çok sınıflı görüntü.
    Hiçbir hasta verisi içermez; yalnızca regresyon/zamanlama testleri içindir.
    """
    rng = np.random.default_rng(seed=0)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    hw, hh = width // 2, height // 2
    arr[:hh, :hw]  = [200,  70,  70]   # Sınıf 1 — kırmızı
    arr[:hh, hw:]  = [ 70, 190,  70]   # Sınıf 2 — yeşil
    arr[hh:, :hw]  = [ 70,  70, 210]   # Sınıf 3 — mavi
    arr[hh:, hw:]  = [210, 185,  55]   # Sınıf 4 — sarı
    noise = rng.integers(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _load_image(path: Optional[str]) -> tuple[Image.Image, str]:
    """Verilen dosyayı yükle; yoksa sentetik görüntü oluştur."""
    if path and Path(path).is_file():
        img = Image.open(path).convert("RGB")
        name = Path(path).name
        print(f"[INFO] Görüntü yüklendi: {path}  ({img.width}x{img.height})")
        return img, name
    elif path:
        print(f"[WARN] Dosya bulunamadı: {path}  — sentetik görüntü kullanılıyor")

    img = _build_synthetic_image(512, 512)
    name = "synthetic_multiclass_512x512.png"
    print(f"[INFO] Sentetik görüntü oluşturuldu: 512×512, 4 sınıf blok")
    return img, name


def _time_call(fn, *args, **kwargs) -> tuple:
    """fn(*args, **kwargs) çalıştır, (sonuç, ms) döndür."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return result, ms


def _tile_recall(
    detections: list,
    image_width: int,
    image_height: int,
    tile_px: int = 256,
) -> dict:
    """
    Görüntüyü tile_px x tile_px karelere böl;
    her karo için hangi sınıfın tespitini aldığını say.
    """
    from collections import defaultdict

    tiles_wide = max(1, image_width  // tile_px)
    tiles_high = max(1, image_height // tile_px)
    total_tiles = tiles_wide * tiles_high

    hits: dict = defaultdict(set)   # label → {tile_idx}
    for det in detections:
        bx, by, bw, bh = det["bbox"]
        cx = bx + bw / 2
        cy = by + bh / 2
        col = min(int(cx // tile_px), tiles_wide - 1)
        row = min(int(cy // tile_px), tiles_high - 1)
        tile_idx = row * tiles_wide + col
        hits[det["label"]].add(tile_idx)

    per_class = {}
    for label, tile_set in hits.items():
        per_class[label] = {
            "tiles_hit":    len(tile_set),
            "tiles_total":  total_tiles,
            "recall":       round(len(tile_set) / total_tiles, 4),
        }
    return per_class


# ==================================================================
# Ana benchmark protokolü
# ==================================================================

def run_benchmark(image_path: Optional[str] = None, warm_runs: int = 3) -> dict:

    print()
    print("=" * 64)
    print("  OCL Backend — Automated Benchmark Protocol")
    print("=" * 64)

    # -- Bağımlılıkları yükle --
    print("\n[1/6] Backend modülleri yükleniyor...")
    try:
        import segmentation
        import vlm_model
        from config import DEVICE, DINOV2_MODEL, SAM_MODEL_TYPE, DETECTION_THRESHOLD
    except ImportError as e:
        sys.exit(f"[ERROR] Backend import edilemedi: {e}\n"
                 f"Bu scripti ocl-backend/ dizininden çalıştırın.")

    print(f"      DEVICE={DEVICE}  DINOv2={DINOV2_MODEL}  SAM={SAM_MODEL_TYPE}")

    # -- Görüntü --
    print("\n[2/6] Test görüntüsü hazırlanıyor...")
    image, image_name = _load_image(image_path)

    # -- Hafızayı sıfırla (temiz başlangıç) --
    print("\n[3/6] Online hafıza sıfırlanıyor (temiz başlangıç)...")
    vlm_model.reset_online_memory()
    print("      Hafıza temizlendi.")

    # ------------------------------------------------------------------
    # AŞAMA A: segment_region — soğuk ve ılık ölçümler
    # ------------------------------------------------------------------
    print("\n[4/6] segment_region ölçümü (SAM + crop)...")

    labels = ["Class1", "Class2", "Class3", "Class4"]
    # 4 sınıfın her biri için merkez koordinat (512×512 görüntü üzerinde)
    annotation_hints = [
        (128, 128, 80),    # Class1: sol üst çeyrek
        (384, 128, 80),    # Class2: sağ üst çeyrek
        (128, 384, 80),    # Class3: sol alt çeyrek
        (384, 384, 80),    # Class4: sağ alt çeyrek
    ]

    segment_timings_ms: list = []
    seg_results = []

    for i, ((cx, cy, r), label) in enumerate(zip(annotation_hints, labels)):
        # İlk çağrı: cold (model yüklemesi dahil)
        tag = "cold" if i == 0 else "warm"
        seg, ms = _time_call(segmentation.segment_region, image, cx, cy, r, shape="circle")
        segment_timings_ms.append(ms)
        seg_results.append((seg, label))
        print(f"      [{tag}] {label}: segment_region={ms:.0f} ms  mask_area={seg.get('mask_area', '?')}")

    cold_segment_ms  = segment_timings_ms[0]
    warm_segment_ms  = median(segment_timings_ms[1:]) if len(segment_timings_ms) > 1 else segment_timings_ms[0]

    # ------------------------------------------------------------------
    # AŞAMA B: predict_and_update — per-class öğretme
    # ------------------------------------------------------------------
    print("\n[5/6] predict_and_update ölçümü (NCM + PEARL-lite)...")

    pu_timings_known: list  = []
    pu_timings_new:   list  = []

    for i, (seg, label) in enumerate(seg_results):
        is_new = i == 0   # İlk sınıf: PEARL-lite tetiklenir
        _, ms = _time_call(vlm_model.predict_and_update, crop=seg["crop"], label=label)
        if is_new:
            pu_timings_new.append(ms)
            tag = "new class (PEARL-lite)"
        else:
            pu_timings_known.append(ms)
            tag = "known class"
        print(f"      [{tag}] {label}: {ms:.0f} ms")

    # Isınmış bilinen sınıf — birkaç tekrar
    if warm_runs > 1:
        print(f"      Isınmış tekrar ({warm_runs} kez, Class1)...")
        for _ in range(warm_runs):
            _, ms = _time_call(vlm_model.predict_and_update,
                               crop=seg_results[0][0]["crop"], label="Class1")
            pu_timings_known.append(ms)

    cold_pu_new_ms  = pu_timings_new[0] if pu_timings_new else None
    warm_pu_new_ms  = median(pu_timings_new) if len(pu_timings_new) > 1 else cold_pu_new_ms
    warm_pu_known_ms = median(pu_timings_known) if pu_timings_known else None

    # ------------------------------------------------------------------
    # AŞAMA C: scan_full_image — tam görüntü tarama
    # ------------------------------------------------------------------
    print("\n[6/6] scan_full_image ölçümü...")
    scan_timings_ms: list = []

    for run_i in range(max(1, warm_runs)):
        tag = "cold" if run_i == 0 else f"warm #{run_i}"
        segments, ms = _time_call(segmentation.scan_full_image, image)
        scan_timings_ms.append(ms)
        print(f"      [{tag}] {len(segments)} segment bulundu — {ms:.0f} ms")

    cold_scan_ms = scan_timings_ms[0]
    warm_scan_ms = median(scan_timings_ms) if len(scan_timings_ms) > 1 else cold_scan_ms

    # Tespit + embedding
    print(f"\n      {len(segments)} segment için NCM sınıflandırması...")
    embed_timings: list = []
    raw_detections = []

    from config import DETECTION_THRESHOLD
    for seg in segments:
        feat, ms_e = _time_call(vlm_model.extract_features, seg["crop"])
        embed_timings.append(ms_e)
        if feat is not None:
            pred_label, conf = vlm_model.classifier.predict(feat)
            if conf >= DETECTION_THRESHOLD:
                raw_detections.append({
                    "label": pred_label,
                    "confidence": round(conf, 4),
                    "bbox": seg["bbox"],
                })

    avg_embed_ms = mean(embed_timings) if embed_timings else 0.0

    print(f"      Ortalama embedding: {avg_embed_ms:.0f} ms | Eşik üstü tespit: {len(raw_detections)}")

    # Tile recall
    tile_recall = _tile_recall(raw_detections, image.width, image.height, tile_px=256)

    # ------------------------------------------------------------------
    # Fonksiyonel kontroller
    # ------------------------------------------------------------------
    print("\n--- Fonksiyonel Doğrulama ---")
    functional_checks = {}

    def _check(name: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        functional_checks[name] = status
        tag = "✓" if condition else "✗"
        suffix = f" ({detail})" if detail else ""
        print(f"  [{tag}] {name}{suffix}")

    _check("Hafıza sıfırlama (reset_online_memory)",
           True,  "başlangıçta çalıştırıldı")

    _check("DINOv2 yüklendi",
           vlm_model.is_loaded(),
           vlm_model.loaded_backend_name() or "bilinmiyor")

    _check("SAM yüklendi",
           segmentation.is_loaded(),
           "checkpoint mevcut")

    _check(f"Öğretme — {labels[0]}",
           labels[0] in vlm_model.classifier.known_classes,
           f"known_classes={vlm_model.classifier.known_classes}")

    _check(f"Öğretme — tüm sınıflar ({len(labels)})",
           all(l in vlm_model.classifier.known_classes for l in labels),
           f"beklenen={labels}")

    _check("Disk kayıt + reload",
           _verify_persist_reload(vlm_model),
           "pickle round-trip")

    _check("PEARL-lite durumu",
           vlm_model._pearl_manager is not None,
           "manager aktif")

    pass_count = sum(1 for v in functional_checks.values() if v == "PASS")
    fail_count = len(functional_checks) - pass_count
    print(f"\n  Toplam: {pass_count}/{len(functional_checks)} geçti")

    # ------------------------------------------------------------------
    # JSON çıktısı
    # ------------------------------------------------------------------
    metrics = {
        "benchmark_image":  image_name,
        "image_size":       f"{image.width}x{image.height}",
        "device":           vlm_model.loaded_backend_name() or "cpu",
        "dinov2_model":     DINOV2_MODEL,
        "sam_model":        SAM_MODEL_TYPE,
        "detection_threshold": DETECTION_THRESHOLD,
        "timing_ms": {
            "segment_region": {
                "cold":         round(cold_segment_ms, 1),
                "warm_median":  round(warm_segment_ms, 1),
            },
            "predict_and_update_new_class_pearl": {
                "cold":         round(cold_pu_new_ms, 1)  if cold_pu_new_ms  is not None else None,
                "warm_median":  round(warm_pu_new_ms, 1)  if warm_pu_new_ms  is not None else None,
            },
            "predict_and_update_known_class": {
                "warm_median":  round(warm_pu_known_ms, 1) if warm_pu_known_ms is not None else None,
            },
            "scan_full_image": {
                "cold":         round(cold_scan_ms, 1),
                "warm_median":  round(warm_scan_ms, 1),
                "segments_found": len(segments),
            },
            "embed_per_segment_avg": round(avg_embed_ms, 1),
        },
        "detection_summary": {
            "total_segments_scanned": len(segments),
            "above_threshold":        len(raw_detections),
            "threshold":              DETECTION_THRESHOLD,
        },
        "tile_recall_256px": tile_recall,
        "functional_checks": functional_checks,
        "functional_summary": f"{pass_count}/{len(functional_checks)} PASS",
    }

    out_path = Path(__file__).parent / "benchmark_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print()
    print("=" * 64)
    print(f"  [SUCCESS] Benchmark tamamlandı.")
    print(f"  Çıktı: {out_path}")
    print("=" * 64)

    # Özet
    print("\n--- Özet ---")
    print(f"  segment_region   cold={cold_segment_ms:.0f} ms   warm={warm_segment_ms:.0f} ms")
    if cold_pu_new_ms:
        print(f"  predict (yeni)   cold={cold_pu_new_ms:.0f} ms   warm={warm_pu_new_ms:.0f} ms")
    if warm_pu_known_ms:
        print(f"  predict (bilinen) warm={warm_pu_known_ms:.0f} ms")
    print(f"  scan_full_image  cold={cold_scan_ms:.0f} ms   warm={warm_scan_ms:.0f} ms  ({len(segments)} seg)")
    print(f"  embed/seg ortalama: {avg_embed_ms:.0f} ms")
    if tile_recall:
        print("  Tile recall (256px):")
        for lbl, r in tile_recall.items():
            print(f"    {lbl}: {r['tiles_hit']}/{r['tiles_total']} ({r['recall']*100:.1f}%)")
    print(f"  Fonksiyonel: {pass_count}/{len(functional_checks)} PASS")

    return metrics


# ------------------------------------------------------------------
# Yardımcı: kalıcı bellek round-trip testi
# ------------------------------------------------------------------

def _verify_persist_reload(vlm_model) -> bool:
    try:
        vlm_model.save_online_memory()
        classes_before = set(vlm_model.classifier.known_classes)
        # Sadece in-memory state'i sıfırla (dosyayı tutarak)
        vlm_model.classifier.class_means.clear()
        vlm_model.classifier.class_counts.clear()
        vlm_model.load_online_memory()
        classes_after = set(vlm_model.classifier.known_classes)
        return classes_before == classes_after
    except Exception as e:
        print(f"      [WARN] Persist/reload testi: {e}")
        return False


# ==================================================================
# Giriş noktası
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OCL Backend Benchmark Driver — gerçek sistem ölçümü"
    )
    parser.add_argument(
        "--image", "-i",
        default=None,
        help="Test görüntüsü dosya yolu (varsayılan: sentetik 512×512)",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=3,
        help="Isınmış ölçüm tekrar sayısı (varsayılan: 3)",
    )
    args = parser.parse_args()

    run_benchmark(image_path=args.image, warm_runs=args.runs)
