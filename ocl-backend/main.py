"""
OCL Backend — FastAPI

POST /segment:
  1. Annotasyon bölgesini SAM ile kes → DINOv2 gömü → NCM güncelle
  2. Tüm görseli SAM ile tara → her segment için DINOv2 gömü → NCM tahmin
  3. Threshold üstündeki eşleşmeleri döndür
"""

import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import (
    CORS_ORIGINS,
    DETECTION_THRESHOLD,
    DETECTION_TOP_K,
    DETECTION_NMS_IOU,
)
from schemas import (
    SegmentRequest, SegmentResponse, SegmentResult, DetectedRegion, HealthResponse,
    InferRequest, FeedbackRequest, FeedbackResponse, ResetMemoryResponse,
)
import segmentation
import vlm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    vlm_model.load_online_memory()
    yield
    vlm_model.save_online_memory()


app = FastAPI(title="OCL Annotation Backend", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _iou_xywh(a: List[float], b: List[float]) -> float:
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(a_x2, b_x2), min(a_y2, b_y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_detections(dets: List[DetectedRegion], iou_thresh: float, top_k: int) -> List[DetectedRegion]:
    """Sınıf başına IoU tabanlı bastırma; sonra güvene göre global top_k."""
    by_label: dict = defaultdict(list)
    for d in dets:
        by_label[d.label].append(d)
    kept: List[DetectedRegion] = []
    for lbl in sorted(by_label.keys()):
        group = sorted(by_label[lbl], key=lambda x: x.confidence, reverse=True)
        while group:
            cur = group.pop(0)
            kept.append(cur)
            group = [g for g in group if _iou_xywh(cur.bbox, g.bbox) < iou_thresh]
    kept.sort(key=lambda x: x.confidence, reverse=True)
    return kept[:top_k]


@app.get("/health", response_model=HealthResponse)
def health():
    from config import DEVICE
    return HealthResponse(
        status="ok",
        sam_loaded=segmentation.is_loaded(),
        vlm_loaded=vlm_model.is_loaded(),
        vlm_backend=vlm_model.loaded_backend_name(),
        device=DEVICE,
        buffer_size=len(vlm_model.buffer),
        known_classes=vlm_model.classifier.known_classes,
        pearl_lite=vlm_model.pearl_status(),
    )


@app.post("/reset-memory", response_model=ResetMemoryResponse)
def reset_memory():
    """
    Online öğrenme hafızasını temizler (NCM + negatif hafıza + replay buffer + disk dosyası).
    DINOv2/CLIP gömü modeli aynı kalır. Bunu siz çağırana kadar öğrenilen sınıflar kalıcı dosyada saklanır.
    """
    vlm_model.reset_online_memory()
    return ResetMemoryResponse(
        ok=True,
        buffer_size=len(vlm_model.buffer),
        known_classes=vlm_model.classifier.known_classes,
    )


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):

    # 1. Görseli çöz
    try:
        image = segmentation.decode_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel çözümlenemedi: {e}")

    annotation_results = []
    any_updated = False

    # 2. Her annotation için: SAM → feature → NCM öğren
    for ann in req.annotations:
        try:
            seg = segmentation.segment_region(image, ann.x, ann.y, ann.radius)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SAM hatası: {e}")

        predicted_class, confidence, updated = vlm_model.predict_and_update(
            crop=seg["crop"],
            label=ann.label,
        )
        if updated:
            any_updated = True

        annotation_results.append(SegmentResult(
            label=ann.label,
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            crop_b64=seg.get("crop_b64"),
            mask_area=seg.get("mask_area"),
        ))

    # 3. Tüm görseli tara — bilinen sınıflarla eşleştir
    detections = []

    if vlm_model.classifier.known_classes:
        try:
            all_segments = segmentation.scan_full_image(image)
        except Exception as e:
            logger.error(f"Tam tarama hatası: {e}")
            all_segments = []

        for seg in all_segments:
            feature = vlm_model.extract_features(seg["crop"])
            if feature is None:
                continue

            pred_label, conf = vlm_model.classifier.predict(feature)

            if conf >= DETECTION_THRESHOLD:
                detections.append(DetectedRegion(
                    bbox=seg["bbox"],
                    label=pred_label,
                    confidence=round(conf, 4),
                    crop_b64=seg.get("crop_b64"),
                    mask_area=seg.get("mask_area"),
                ))

        raw_n = len(detections)
        detections = _nms_detections(detections, DETECTION_NMS_IOU, DETECTION_TOP_K)
        logger.info(
            f"Tarama tamamlandı: {len(all_segments)} aday, {raw_n} eşik üstü, "
            f"{len(detections)} son (threshold={DETECTION_THRESHOLD}, "
            f"nms_iou={DETECTION_NMS_IOU}, top_k={DETECTION_TOP_K})"
        )

    if any_updated:
        vlm_model.save_online_memory()

    return SegmentResponse(
        results=annotation_results,
        detections=detections,
        model_updated=any_updated,
    )


@app.post("/infer", response_model=SegmentResponse)
def infer(req: InferRequest):
    """Annotation olmadan sadece tarama/inference."""
    try:
        image = segmentation.decode_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Görsel çözümlenemedi: {e}")

    detections = []
    if vlm_model.classifier.known_classes:
        try:
            all_segments = segmentation.scan_full_image(image)
        except Exception as e:
            logger.error(f"Tam tarama hatası: {e}")
            all_segments = []

        for seg in all_segments:
            feature = vlm_model.extract_features(seg["crop"])
            if feature is None:
                continue
            pred_label, conf = vlm_model.classifier.predict(feature)
            if conf >= DETECTION_THRESHOLD:
                detections.append(DetectedRegion(
                    bbox=seg["bbox"],
                    label=pred_label,
                    confidence=round(conf, 4),
                    crop_b64=seg.get("crop_b64"),
                    mask_area=seg.get("mask_area"),
                ))

        detections = _nms_detections(detections, DETECTION_NMS_IOU, DETECTION_TOP_K)

    return SegmentResponse(results=[], detections=detections, model_updated=False)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """
    Kullanıcının doğru/yanlış geri bildirimi.
    - accepted=True  => pozitif güncelleme
    - accepted=False => yanlış sınıfa negatif, corrected_label varsa doğru sınıfa pozitif güncelleme
    """
    updated = 0
    skipped = 0

    for item in req.items:
        if not item.crop_b64:
            skipped += 1
            continue
        try:
            crop = segmentation.decode_image(item.crop_b64)
            feature = vlm_model.extract_features(crop)
            if feature is None:
                skipped += 1
                continue

            predicted_label = (item.predicted_label or item.label).strip()
            corrected_label = (item.corrected_label or "").strip() or None
            did_update = vlm_model.classifier.apply_feedback(
                feature=feature,
                predicted_label=predicted_label,
                accepted=item.accepted,
                corrected_label=corrected_label,
            )
            if did_update and item.accepted:
                vlm_model.buffer.add(feature, corrected_label or predicted_label)

            if did_update:
                updated += 1
            else:
                skipped += 1
        except Exception as e:
            logger.error(f"Feedback işlenemedi: {e}")
            skipped += 1

    if updated > 0:
        vlm_model.save_online_memory()

    return FeedbackResponse(updated_count=updated, skipped_count=skipped)
