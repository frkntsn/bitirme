#!/usr/bin/env python3
"""DINOv2'yi proje önbelleğine indirip doğrular."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os

import config  # noqa: F401 — TORCH_HOME / HF_HOME ayarları
import vlm_model

if __name__ == "__main__":
    print(f"TORCH_HOME={os.environ.get('TORCH_HOME')}")
    print(f"DEVICE={config.DEVICE}")
    print(f"DINOV2_MODEL={config.DINOV2_MODEL}")
    ok = vlm_model.is_loaded()
    name = vlm_model.loaded_backend_name()
    print(f"loaded={ok} backend={name}")
    if not ok or "dinov2" not in (name or "").lower():
        sys.exit(1)
    print("DINOv2 hazır.")
