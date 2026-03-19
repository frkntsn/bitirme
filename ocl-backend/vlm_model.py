"""
VLM-CL Modeli — UNI2-h Backbone
"""

import logging
from collections import defaultdict
from typing import Optional

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import BUFFER_SIZE, DEVICE, HF_TOKEN

logger = logging.getLogger(__name__)

UNI2_MODEL_NAME = "MahmoodLab/uni2-h"
_model = None
_transform = None
_clip_model = None
_clip_processor = None

# Loglarda görünen hata shape'i genelde beklenen patch grid ile uyuşmazlıktan çıkar.
# Bu model checkpoint'i çoğu ortamda 16x16 grid'e (img_size=256, patch=16 varsayımı) karşılık gelen pozisyonel
# embedding değerleriyle geliyor; bu yüzden default'u 256 yapıyoruz.
UNI2_IMG_SIZE = int(os.getenv("UNI2_IMG_SIZE", "256"))
UNI2_IMG_SIZE_CANDIDATES = os.getenv(
    "UNI2_IMG_SIZE_CANDIDATES",
    "256,240,224,192,288,320,384"
)
try:
    _UNI2_CANDIDATES = [int(x.strip()) for x in UNI2_IMG_SIZE_CANDIDATES.split(",") if x.strip()]
except Exception:
    _UNI2_CANDIDATES = [UNI2_IMG_SIZE]
if UNI2_IMG_SIZE not in _UNI2_CANDIDATES:
    _UNI2_CANDIDATES = [UNI2_IMG_SIZE] + _UNI2_CANDIDATES


def _load_uni2():
    global _model, _transform
    if _model is not None:
        return _model, _transform
    try:
        import timm
        from torchvision import transforms
        from huggingface_hub import login
        if HF_TOKEN:
            login(token=HF_TOKEN)
        logger.info(f"UNI2-h yükleniyor...")
        # Bazı ortam/versiyon kombinasyonlarında dynamic_img_size True iken
        # modelin pozisyonel embedding yeniden boyutlandırması shape error ile
        # patlayabiliyor. Bu yüzden:
        # 1) dynamic_img_size=True deniyoruz
        # 2) başarısız olursa dynamic_img_size=False ile farklı img_size adaylarını deniyoruz
        try:
            _model = timm.create_model(
                "hf-hub:MahmoodLab/uni2-h",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        except Exception as e:
            logger.warning(
                f"UNI2-h dynamic yükleme başarısız: {e}. Sabit img_size adayları deneniyor: {_UNI2_CANDIDATES}"
            )
            last_err = e
            _model = None
            for cand in _UNI2_CANDIDATES:
                try:
                    _model = timm.create_model(
                        "hf-hub:MahmoodLab/uni2-h",
                        pretrained=True,
                        init_values=1e-5,
                        dynamic_img_size=False,
                        img_size=cand,
                    )
                    logger.info(f"UNI2-h sabit img_size ile hazır: {cand}")
                    break
                except Exception as e2:
                    last_err = e2
                    logger.warning(f"UNI2-h img_size={cand} başarısız: {e2}")
            if _model is None:
                raise last_err
        _model.to(DEVICE)
        _model.eval()
        _transform = transforms.Compose([
            transforms.Resize(UNI2_IMG_SIZE),
            transforms.CenterCrop(UNI2_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        logger.info("UNI2-h hazır.")
        return _model, _transform
    except Exception as e:
        logger.error(f"UNI2-h yüklenemedi: {e}")
        return None, None


def _load_clip():
    """
    UNI2-h bu ortam/versiyon uyumsuzluğu yüzünden yüklenemiyorsa
    alternatif olarak CLIP kullan.
    """
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor

    # config.py zaten VLM_MODEL_NAME'i okuyor; biz burada env'yi direkt alıyoruz.
    # Örn: openai/clip-vit-base-patch32
    clip_name = os.getenv("VLM_MODEL_NAME", "openai/clip-vit-base-patch32")
    try:
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"CLIP yükleniyor: {clip_name} @ {DEVICE}")
        _clip_processor = CLIPProcessor.from_pretrained(clip_name)
        _clip_model = CLIPModel.from_pretrained(clip_name)
        _clip_model.to(DEVICE)
        _clip_model.eval()
        logger.info("CLIP hazır.")
        return _clip_model, _clip_processor
    except Exception as e:
        logger.error(f"CLIP yüklenemedi: {e}")
        return None, None


def is_loaded() -> bool:
    m, _ = _load_uni2()
    if m is not None:
        return True
    cm, _ = _load_clip()
    return cm is not None


def loaded_backend_name() -> Optional[str]:
    """
    Yüklenmiş halde hangisi aktifse onu döner.
    Not: Bu fonksiyon yeni model indirmeye/yeniden yüklemeye çalışmaz;
    sadece cache değişkenlerine bakar.
    """
    if _model is not None and _transform is not None:
        return "MahmoodLab/uni2-h"
    if _clip_model is not None and _clip_processor is not None:
        return "openai/clip-vit-base-patch32"
    return None


def _to_feature_tensor(x) -> Optional[torch.Tensor]:
    """
    Bazı modeller tensor yerine output dataclass döndürebilir
    (örn. BaseModelOutputWithPooling). normalize edebilmek için
    (B, D) şekilli tensor'a çeviriyoruz.
    """
    if isinstance(x, torch.Tensor):
        return x

    # Transformers/timm output dataclass'ları
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, torch.Tensor):
                return v

    # last_hidden_state => (B, T, D) -> mean pool
    if hasattr(x, "last_hidden_state"):
        v = getattr(x, "last_hidden_state")
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            return v.mean(dim=1)

    return None


def extract_features(crop: Image.Image) -> Optional[np.ndarray]:
    model, transform = _load_uni2()
    if model is not None and transform is not None:
        with torch.no_grad():
            tensor = transform(crop).unsqueeze(0).to(DEVICE)
            feats_out = model(tensor)
            feats = _to_feature_tensor(feats_out)
            if feats is None:
                return None
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()[0]

    # Fallback: CLIP
    clip_model, clip_processor = _load_clip()
    if clip_model is None or clip_processor is None:
        return None

    with torch.no_grad():
        inputs = clip_processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats_out = clip_model.get_image_features(**inputs)
        feats = _to_feature_tensor(feats_out)
        if feats is None:
            return None
        feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy()[0]


class MIRBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.features: list = []
        self.labels: list = []
        self._n_seen = 0

    def add(self, feature: np.ndarray, label: str):
        self._n_seen += 1
        if len(self.features) < self.max_size:
            self.features.append(feature)
            self.labels.append(label)
        else:
            idx = np.random.randint(0, self._n_seen)
            if idx < self.max_size:
                self.features[idx] = feature
                self.labels[idx] = label

    def sample(self, n=10):
        if not self.features:
            return [], []
        n = min(n, len(self.features))
        indices = np.random.choice(len(self.features), n, replace=False)
        return [self.features[i] for i in indices], [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.features)


class NCMClassifier:
    def __init__(self):
        self.class_means: dict = {}
        self.class_counts: dict = defaultdict(int)

    def update(self, label: str, feature: np.ndarray):
        n = self.class_counts[label]
        if label not in self.class_means:
            self.class_means[label] = feature.copy()
        else:
            self.class_means[label] = (self.class_means[label] * n + feature) / (n + 1)
            norm = np.linalg.norm(self.class_means[label])
            if norm > 0:
                self.class_means[label] /= norm
        self.class_counts[label] += 1

    def predict(self, feature: np.ndarray) -> tuple:
        if not self.class_means:
            return "unknown", 0.0
        best_label, best_sim = "unknown", -1.0
        feat_norm = feature / (np.linalg.norm(feature) + 1e-9)
        for label, mean in self.class_means.items():
            sim = float(np.dot(feat_norm, mean))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label, (best_sim + 1.0) / 2.0

    @property
    def known_classes(self) -> list:
        return list(self.class_means.keys())


buffer = MIRBuffer(max_size=BUFFER_SIZE)
classifier = NCMClassifier()


def predict_and_update(crop: Image.Image, label: str) -> tuple:
    feature = extract_features(crop)
    if feature is None:
        return label, 1.0, False
    predicted_class, confidence = classifier.predict(feature)
    replay_feats, replay_labels = buffer.sample(n=10)
    classifier.update(label, feature)
    for rf, rl in zip(replay_feats, replay_labels):
        classifier.update(rl, rf)
    buffer.add(feature, label)
    logger.info(f"Update: '{label}' → pred='{predicted_class}' conf={confidence:.2f} | buffer={len(buffer)}")
    return predicted_class, confidence, True
