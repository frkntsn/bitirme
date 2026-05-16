from pathlib import Path
from dotenv import load_dotenv
import os

# Proje kök dizini
BASE_DIR = Path(__file__).parent

# Bu backend için doğru `.env` dosyasını özellikle yükle.
# Böylece yukarı dizinlerdeki başka `.env` değerleri yanlışlıkla override etmez.
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Ana PyTorch device (DINOv2, CLIP, …)
DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")

# SAM
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", str(BASE_DIR / "weights" / "sam_vit_b_01ec64.pth"))
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")   # vit_b | vit_l | vit_h
# segment_anything (özellikle SamAutomaticMaskGenerator) MPS'te float64 hatası veriyor.
# Varsayılan: DEVICE=mps iken SAM CPU'da; DINOv2 vb. yine DEVICE'da kalır.
_sam_dev = os.getenv("SAM_DEVICE", "").strip()
SAM_DEVICE = _sam_dev if _sam_dev else ("cpu" if DEVICE == "mps" else DEVICE)

# CLIP yedek gömü (DINOv2 yüklenemezse)
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "openai/clip-vit-base-patch32")

# DINOv2 — torch.hub: facebookresearch/dinov2
# Örnekler: dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14
DINOV2_MODEL = os.getenv("DINOV2_MODEL", "dinov2_vitb14")
# ViT patch 14 → giriş kenarı 14'e bölünebilir olmalı (224, 518, …)
DINOV2_IMG_SIZE = int(os.getenv("DINOV2_IMG_SIZE", "224"))

# Görsel işleme
CROP_SIZE = int(os.getenv("CROP_SIZE", "224"))

# Tespit kalitesi (main.py NCM sonrası)
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.65"))
DETECTION_TOP_K = int(os.getenv("DETECTION_TOP_K", "30"))
DETECTION_NMS_IOU = float(os.getenv("DETECTION_NMS_IOU", "0.40"))

# SAM otomatik maske (tam görsel tarama)
SAM_AUTO_POINTS_PER_SIDE = int(os.getenv("SAM_AUTO_POINTS_PER_SIDE", "24"))
SAM_AUTO_PRED_IOU_THRESH = float(os.getenv("SAM_AUTO_PRED_IOU_THRESH", "0.86"))
SAM_AUTO_STABILITY_THRESH = float(os.getenv("SAM_AUTO_STABILITY_THRESH", "0.88"))
SAM_AUTO_MIN_MASK_AREA = int(os.getenv("SAM_AUTO_MIN_MASK_AREA", "120"))

# SAM auto yokken: sliding (önerilir) | grid
SCAN_FALLBACK = os.getenv("SCAN_FALLBACK", "sliding").strip().lower()
SLIDING_WINDOW = int(os.getenv("SLIDING_WINDOW", "0"))  # 0 = görüye göre otomatik
SLIDING_STRIDE_RATIO = float(os.getenv("SLIDING_STRIDE_RATIO", "0.45"))
SCAN_MAX_WINDOWS = int(os.getenv("SCAN_MAX_WINDOWS", "320"))
GRID_FALLBACK_N = int(os.getenv("GRID_FALLBACK_N", "6"))

# Replay buffer (Faz 3)
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "5000"))

# Öğrenilen sınıflar (NCM + buffer) — sunucu restart sonrası korunur; sadece /reset-memory ile silinir
_online_flag = os.getenv("ONLINE_MEMORY_ENABLED", "1").strip().lower()
ONLINE_MEMORY_ENABLED = _online_flag not in ("0", "false", "no", "off")
ONLINE_MEMORY_PATH = Path(
    os.getenv("ONLINE_MEMORY_PATH", str(BASE_DIR / "data" / "online_memory.pkl"))
)

# CORS — frontend adresi
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# HuggingFace (CLIP vb. Hugging Face ağırlıkları için isteğe bağlı)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# PEARL-lite — DINOv2 son bloklarında LoRA + SVD dinamik rank
_pearl_flag = os.getenv("PEARL_LITE_ENABLED", "1").strip().lower()
PEARL_LITE_ENABLED = _pearl_flag not in ("0", "false", "no", "off")
PEARL_NUM_BLOCKS = int(os.getenv("PEARL_NUM_BLOCKS", "2"))
PEARL_LORA_R_MAX = int(os.getenv("PEARL_LORA_R_MAX", "16"))
PEARL_LORA_ALPHA = float(os.getenv("PEARL_LORA_ALPHA", "16"))
PEARL_TRAIN_STEPS = int(os.getenv("PEARL_TRAIN_STEPS", "12"))
PEARL_LR = float(os.getenv("PEARL_LR", "1e-4"))
PEARL_MIN_CROPS = int(os.getenv("PEARL_MIN_CROPS", "1"))
PEARL_MAX_CROPS_PER_CLASS = int(os.getenv("PEARL_MAX_CROPS_PER_CLASS", "16"))
