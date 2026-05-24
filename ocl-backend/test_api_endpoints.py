"""
OCL Backend — API Smoke Tests
==============================
FastAPI TestClient kullanarak gerçek endpoint doğrulaması.
Çalıştırmak için:
    cd ocl-backend
    python test_api_endpoints.py
    # veya pytest ile:
    pytest test_api_endpoints.py -v
"""

import base64
import io
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------
# PIL varlık kontrolü
# ------------------------------------------------------------------
try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Pillow kurulu değil. `pip install Pillow` çalıştırın.")

# ------------------------------------------------------------------
# TestClient import
# ------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
except ImportError:
    sys.exit("[ERROR] FastAPI kurulu değil. `pip install fastapi[all]` çalıştırın.")


# ------------------------------------------------------------------
# Uygulama import — bu dosya ocl-backend/ içinden çalıştırılmalı
# ------------------------------------------------------------------
try:
    from main import app
except ImportError as exc:
    sys.exit(f"[ERROR] main.py import edilemedi: {exc}\n"
             f"Lütfen bu scripti ocl-backend/ dizininden çalıştırın.")


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _make_test_image_b64(width: int = 256, height: int = 256) -> str:
    """Gerçek bir PNG görüntü oluştur ve base64'e çevir."""
    import numpy as np

    rng = np.random.default_rng(seed=42)
    # 4 farklı renk bloğu içeren sentetik görüntü
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    hw, hh = width // 2, height // 2
    arr[:hh, :hw] = [220, 80, 80]    # Kırmızı bölge (Sınıf A)
    arr[:hh, hw:] = [80, 180, 80]    # Yeşil bölge (Sınıf B)
    arr[hh:, :hw] = [80, 80, 220]    # Mavi bölge (Sınıf C)
    arr[hh:, hw:] = [200, 180, 60]   # Sarı bölge (Sınıf D)
    # Hafif gürültü ekle (daha gerçekçi)
    noise = rng.integers(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


_TEST_IMAGE_B64 = _make_test_image_b64()


def _pass(test_name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  [PASS] {test_name}{suffix}")


def _fail(test_name: str, reason: str) -> None:
    print(f"  [FAIL] {test_name} — {reason}", file=sys.stderr)


# ------------------------------------------------------------------
# Test fonksiyonları (pytest uyumlu + doğrudan çalıştırılabilir)
# ------------------------------------------------------------------

client = TestClient(app, raise_server_exceptions=False)


def test_health_endpoint():
    """/health endpoint'i 200 döndürmeli ve beklenen alanları içermeli."""
    response = client.get("/health")
    assert response.status_code == 200, \
        f"Beklenen 200, alınan {response.status_code}"
    body = response.json()
    assert "status" in body, "'status' alanı eksik"
    assert body["status"] == "ok", f"status={body['status']}"
    assert "vlm_loaded" in body, "'vlm_loaded' alanı eksik"
    assert "sam_loaded" in body, "'sam_loaded' alanı eksik"
    assert "known_classes" in body, "'known_classes' alanı eksik"
    assert "device" in body, "'device' alanı eksik"
    _pass("GET /health", f"device={body['device']} vlm={body['vlm_loaded']} sam={body['sam_loaded']}")
    return body


def test_reset_memory_endpoint():
    """/reset-memory endpoint'i 200 döndürmeli ve hafızayı temizlemeli."""
    response = client.post("/reset-memory")
    assert response.status_code == 200, \
        f"Beklenen 200, alınan {response.status_code}"
    body = response.json()
    assert body.get("ok") is True, "ok != True"
    assert body.get("known_classes") == [], \
        f"Hafıza sıfırlandıktan sonra known_classes boş olmalı, alınan: {body.get('known_classes')}"
    assert body.get("buffer_size") == 0, \
        f"buffer_size 0 olmalı, alınan: {body.get('buffer_size')}"
    _pass("POST /reset-memory", "NCM + buffer temizlendi")


def test_segment_endpoint():
    """/segment endpoint'i geçerli bir istek için 200 ve sonuç döndürmeli."""
    payload = {
        "image": _TEST_IMAGE_B64,
        "annotations": [
            {"x": 64, "y": 64, "radius": 40, "label": "ClassA", "shape": "circle"},
        ],
    }
    response = client.post("/segment", json=payload)
    assert response.status_code == 200, \
        f"Beklenen 200, alınan {response.status_code}: {response.text[:300]}"
    body = response.json()
    assert "results" in body, "'results' alanı eksik"
    assert len(body["results"]) == 1, \
        f"1 annotation için 1 sonuç bekleniyor, alınan: {len(body['results'])}"
    result = body["results"][0]
    assert result["label"] == "ClassA", \
        f"label='ClassA' bekleniyor, alınan: {result['label']}"
    assert "confidence" in result, "'confidence' alanı eksik"
    assert "model_updated" in body, "'model_updated' alanı eksik"
    _pass("POST /segment", f"label={result['label']} conf={result['confidence']:.4f} "
                           f"detections={len(body.get('detections', []))}")
    return body


def test_class_persistence_in_health():
    """
    /segment ile öğretilen sınıf /health'te görünmeli.
    Bu test test_segment_endpoint'in ardından çalışmalıdır.
    """
    # Önce bir sınıf öğret
    payload = {
        "image": _TEST_IMAGE_B64,
        "annotations": [
            {"x": 64, "y": 64, "radius": 40, "label": "PersistClass", "shape": "circle"},
        ],
    }
    seg_resp = client.post("/segment", json=payload)
    assert seg_resp.status_code == 200, f"/segment başarısız: {seg_resp.status_code}"

    # Health'te görünmeli
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    body = health_resp.json()
    known = body.get("known_classes", [])
    assert "PersistClass" in known, \
        f"'PersistClass' known_classes içinde yok: {known}"
    _pass("Class persistence via /health", f"known_classes={known}")


# ------------------------------------------------------------------
# Doğrudan çalıştırma (python test_api_endpoints.py)
# ------------------------------------------------------------------

def _run_all_tests() -> None:
    print("=" * 60)
    print("  OCL Backend — API Smoke Test Suite")
    print("=" * 60)

    tests = [
        ("GET /health",                  test_health_endpoint),
        ("POST /reset-memory",           test_reset_memory_endpoint),
        ("POST /segment",                test_segment_endpoint),
        ("Class persistence (/health)",  test_class_persistence_in_health),
    ]

    passed = 0
    failed = 0
    start_total = time.perf_counter()

    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            passed += 1
            print(f"         ({elapsed*1000:.0f} ms)")
        except AssertionError as err:
            elapsed = time.perf_counter() - t0
            _fail(name, str(err))
            failed += 1
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            _fail(name, f"Beklenmedik hata: {exc}")
            failed += 1

    total_ms = (time.perf_counter() - start_total) * 1000
    print("=" * 60)
    print(f"  Sonuç: {passed} geçti / {failed} başarısız  ({total_ms:.0f} ms toplam)")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all_tests()
