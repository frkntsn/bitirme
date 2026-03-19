"""
OCL Backend — FastAPI

POST /segment:
  1. Annotasyon bölgesini SAM ile kes → UNI2 feature → NCM güncelle
  2. Tüm görseli SAM ile tara → her segment için UNI2 feature → NCM tahmin
  3. Threshold üstündeki eşleşmeleri döndür
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import CORS_ORIGINS
from schemas import SegmentRequest, SegmentResponse, SegmentResult, DetectedRegion, HealthResponse
import segmentation
import vlm_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="OCL Annotation Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eşleşme için minimum confidence eşiği
DETECTION_THRESHOLD = 0.60


@app.get("/health", response_model=HealthResponse)
def health():
    from config import DEVICE
    return HealthResponse(
        status="ok",
        sam_loaded=segmentation.is_loaded(),
        vlm_loaded=vlm_model.is_loaded(),
        device=DEVICE,
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

        logger.info(
            f"Tarama tamamlandı: {len(all_segments)} segment, "
            f"{len(detections)} eşleşme (threshold={DETECTION_THRESHOLD})"
        )

    return SegmentResponse(
        results=annotation_results,
        detections=detections,
        model_updated=any_updated,
    )
