from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Proje kök dizini
BASE_DIR = Path(__file__).parent

# SAM
SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", str(BASE_DIR / "weights" / "sam_vit_b_01ec64.pth"))
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")   # vit_b | vit_l | vit_h

# VLM (Faz 3'te kullanılacak)
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "openai/clip-vit-base-patch32")

# Görsel işleme
CROP_SIZE = int(os.getenv("CROP_SIZE", "224"))

# Replay buffer (Faz 3)
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "5000"))

# CORS — frontend adresi
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# HuggingFace (UNI2-h gated model için)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# Device
DEVICE = os.getenv("DEVICE", "cuda" if os.path.exists("/dev/nvidia0") else "cpu")
