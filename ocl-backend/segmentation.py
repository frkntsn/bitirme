"""
SAM wrapper - iki mod:
  1. segment_region()   : tek daire → tek maske (annotation için)
  2. scan_full_image()  : tüm görsel → tüm maskeler → VLM eşleştirme
"""

import logging
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    SAM_CHECKPOINT,
    SAM_MODEL_TYPE,
    CROP_SIZE,
    DEVICE,
    SAM_DEVICE,
    SAM_AUTO_POINTS_PER_SIDE,
    SAM_AUTO_PRED_IOU_THRESH,
    SAM_AUTO_STABILITY_THRESH,
    SAM_AUTO_MIN_MASK_AREA,
    SCAN_FALLBACK,
    SLIDING_WINDOW,
    SLIDING_STRIDE_RATIO,
    SCAN_MAX_WINDOWS,
    GRID_FALLBACK_N,
)

logger = logging.getLogger(__name__)

_predictor = None
_auto_generator = None


def _load_sam():
    global _predictor
    if _predictor is not None:
        return _predictor
    checkpoint = Path(SAM_CHECKPOINT)
    if not checkpoint.exists():
        logger.warning(f"SAM ağırlığı bulunamadı: {checkpoint}")
        return None
    try:
        from segment_anything import SamPredictor, sam_model_registry
        logger.info(f"SAM yükleniyor: {SAM_MODEL_TYPE} @ {SAM_DEVICE} (projede DEVICE={DEVICE})")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint))
        sam.to(SAM_DEVICE)
        _predictor = SamPredictor(sam)
        logger.info("SAM hazır.")
        return _predictor
    except Exception as e:
        logger.error(f"SAM yüklenemedi: {e}")
        return None


def _load_auto_generator():
    global _auto_generator
    if _auto_generator is not None:
        return _auto_generator
    checkpoint = Path(SAM_CHECKPOINT)
    if not checkpoint.exists():
        return None
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint))
        sam.to(SAM_DEVICE)
        # points_per_side arttırılırsa daha hassas ama yavaş
        _auto_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=SAM_AUTO_POINTS_PER_SIDE,
            pred_iou_thresh=SAM_AUTO_PRED_IOU_THRESH,
            stability_score_thresh=SAM_AUTO_STABILITY_THRESH,
            min_mask_region_area=SAM_AUTO_MIN_MASK_AREA,
        )
        logger.info("SAM AutoGenerator hazır.")
        return _auto_generator
    except Exception as e:
        logger.error(f"AutoGenerator yüklenemedi: {e}")
        return None


def is_loaded() -> bool:
    return _load_sam() is not None


def decode_image(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(BytesIO(data)).convert("RGB")


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def segment_region(image: Image.Image, cx: float, cy: float, radius: float) -> dict:
    """Tek nokta → SAM maske → normalize crop"""
    predictor = _load_sam()
    img_np = np.array(image)

    if predictor is None:
        return _fallback_crop(image, cx, cy, radius)

    predictor.set_image(img_np)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]], dtype=np.float32),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]
    crop = _crop_from_mask(image, mask, cx, cy, radius)
    return {
        "mask": mask,
        "crop": crop,
        "crop_b64": image_to_b64(crop),
        "mask_area": int(mask.sum()),
    }


def scan_full_image(image: Image.Image) -> list:
    """
    Tüm görseli SAM ile tara.
    Her segment için bbox + crop döndür.
    VLM eşleştirmesi main.py'de yapılır.

    Returns: [{ bbox, crop, crop_b64, mask_area }, ...]
    """
    generator = _load_auto_generator()
    img_np = np.array(image)

    if generator is None:
        logger.warning(
            f"SAM AutoGenerator yok — yedek tarama: {SCAN_FALLBACK} "
            f"(opencv-python-headless kurulu mu kontrol edin)"
        )
        if SCAN_FALLBACK == "grid":
            return _grid_scan_fallback(image, grid=GRID_FALLBACK_N)
        return _sliding_window_scan(image)

    logger.info("Tam görsel taranıyor...")
    masks = generator.generate(img_np)
    logger.info(f"{len(masks)} segment bulundu.")

    results = []
    for m in masks:
        bbox = m["bbox"]  # [x, y, w, h]
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        crop = _crop_from_mask(image, m["segmentation"], cx, cy, max(bbox[2], bbox[3]) / 2)
        results.append({
            "bbox": bbox,
            "crop": crop,
            "crop_b64": image_to_b64(crop),
            "mask_area": int(m["segmentation"].sum()),
            "stability_score": float(m.get("stability_score", 0)),
        })

    return results


def _crop_from_mask(image, mask, cx, cy, radius):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = max(5, int(radius * 0.1))
        rmin = max(0, rmin - pad)
        rmax = min(image.height - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(image.width - 1, cmax + pad)
    else:
        r = int(radius)
        cmin = max(0, int(cx) - r)
        cmax = min(image.width - 1, int(cx) + r)
        rmin = max(0, int(cy) - r)
        rmax = min(image.height - 1, int(cy) + r)
    cropped = image.crop((cmin, rmin, cmax + 1, rmax + 1))
    return cropped.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)


def _fallback_crop(image, cx, cy, radius):
    r = int(radius)
    cmin = max(0, int(cx) - r)
    cmax = min(image.width - 1, int(cx) + r)
    rmin = max(0, int(cy) - r)
    rmax = min(image.height - 1, int(cy) + r)
    cropped = image.crop((cmin, rmin, cmax + 1, rmax + 1))
    resized = cropped.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    return {"mask": None, "crop": resized, "crop_b64": image_to_b64(resized), "mask_area": 0}


def _grid_scan_fallback(image: Image.Image, grid: int) -> list:
    """Kaba grid yedeği (SCAN_FALLBACK=grid)."""
    w, h = image.size
    grid = max(2, min(grid, 24))
    cw, ch = max(1, w // grid), max(1, h // grid)
    results = []
    for row in range(grid):
        for col in range(grid):
            x0, y0 = col * cw, row * ch
            crop = image.crop((x0, y0, x0 + cw, y0 + ch))
            crop_resized = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
            results.append({
                "bbox": [x0, y0, cw, ch],
                "crop": crop_resized,
                "crop_b64": image_to_b64(crop_resized),
                "mask_area": cw * ch,
                "stability_score": 1.0,
            })
    return results


def _tile_starts(dim: int, window: int, stride: int) -> list:
    """[0, dim-window] aralığını stride ile tara; sağ alt köşeyi kaçırmamak için son tile eklenir."""
    if dim <= window:
        return [0]
    starts = list(range(0, dim - window + 1, stride))
    last = dim - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sliding_window_scan(image: Image.Image) -> list:
    """
    SAM auto yokken: örtüşen pencerelerle aday bölgeler (4x4 grid'den daha kontrollü).
    Pencere sayısı SCAN_MAX_WINDOWS'u aşarsa stride büyütülür.
    """
    w, h = image.size
    win = SLIDING_WINDOW if SLIDING_WINDOW > 0 else max(96, min(288, min(w, h) // 3))
    win = int(max(64, min(win, w, h)))

    stride = max(24, int(win * SLIDING_STRIDE_RATIO))

    for _ in range(48):
        xs = _tile_starts(w, win, stride)
        ys = _tile_starts(h, win, stride)
        if len(xs) * len(ys) <= SCAN_MAX_WINDOWS:
            break
        if stride >= win:
            break
        stride = min(win - 1, int(stride * 1.35) + 8)

    xs = _tile_starts(w, win, stride)
    ys = _tile_starts(h, win, stride)
    if len(xs) * len(ys) > SCAN_MAX_WINDOWS:
        stride = win
        xs = _tile_starts(w, win, stride)
        ys = _tile_starts(h, win, stride)

    results = []
    for y in ys:
        for x in xs:
            x2, y2 = min(x + win, w), min(y + win, h)
            bw, bh = x2 - x, y2 - y
            if bw < 32 or bh < 32:
                continue
            crop = image.crop((x, y, x2, y2))
            crop_resized = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
            results.append({
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "crop": crop_resized,
                "crop_b64": image_to_b64(crop_resized),
                "mask_area": int(bw * bh),
                "stability_score": 1.0,
            })

    logger.info(f"Sliding fallback: win={win} stride={stride} aday={len(results)}")
    return results
