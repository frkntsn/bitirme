from pydantic import BaseModel, Field
from typing import List, Optional


class AnnotationIn(BaseModel):
    x: float
    y: float
    radius: float = Field(..., gt=0)
    label: str = Field(..., min_length=1)


class SegmentRequest(BaseModel):
    image: str
    annotations: List[AnnotationIn] = Field(..., min_length=1)


class SegmentResult(BaseModel):
    label: str
    predicted_class: str
    confidence: float
    crop_b64: Optional[str] = None
    mask_area: Optional[int] = None


class DetectedRegion(BaseModel):
    bbox: List[float]          # [x, y, w, h] — orijinal görsel koordinatları
    label: str                 # eşleşen sınıf
    confidence: float
    crop_b64: Optional[str] = None
    mask_area: Optional[int] = None


class SegmentResponse(BaseModel):
    results: List[SegmentResult]        # annotation başına sonuç
    detections: List[DetectedRegion]    # tüm görseldeki eşleşmeler
    model_updated: bool = False


class HealthResponse(BaseModel):
    status: str
    sam_loaded: bool
    vlm_loaded: bool
    device: str
    buffer_size: int
    known_classes: List[str]
