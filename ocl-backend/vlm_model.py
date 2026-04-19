"""
Gömü motoru — DINOv2 (torch.hub) + isteğe bağlı CLIP yedeği.
NCM + MIR buffer ile online güncelleme aynı kalır.
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import (
    BUFFER_SIZE,
    DEVICE,
    DINOV2_IMG_SIZE,
    DINOV2_MODEL,
    VLM_MODEL_NAME,
    ONLINE_MEMORY_ENABLED,
    ONLINE_MEMORY_PATH,
)

logger = logging.getLogger(__name__)

_dinov2_model = None
_dinov2_transform = None
_dinov2_backend_label: Optional[str] = None
_DINOV2_LOAD_FAILED = False

_clip_model = None
_clip_processor = None


def _dinov2_cls_embedding(model: torch.nn.Module, tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """forward_features çıktısından (B, D) CLS vektörü."""
    out = model.forward_features(tensor)
    if isinstance(out, dict):
        t = out.get("x_norm_clstoken")
        if isinstance(t, torch.Tensor):
            if t.dim() == 3:
                return t[:, 0]
            return t
        t = out.get("x_norm_patchtokens")
        if isinstance(t, torch.Tensor) and t.dim() == 3:
            return t.mean(dim=1)
        return None
    if isinstance(out, torch.Tensor):
        if out.dim() == 3:
            return out[:, 0]
        if out.dim() == 2:
            return out
    return None


def _load_dinov2():
    global _dinov2_model, _dinov2_transform, _dinov2_backend_label
    global _DINOV2_LOAD_FAILED

    if _dinov2_model is not None:
        return _dinov2_model, _dinov2_transform
    if _DINOV2_LOAD_FAILED:
        return None, None

    if DINOV2_IMG_SIZE % 14 != 0:
        logger.warning(
            f"DINOV2_IMG_SIZE={DINOV2_IMG_SIZE} patch 14 için uygun değil; 224 kullanılıyor."
        )
        img_size = 224
    else:
        img_size = DINOV2_IMG_SIZE

    try:
        logger.info(f"DINOv2 yükleniyor: {DINOV2_MODEL} @ {DEVICE} (giriş {img_size})")
        model = torch.hub.load(
            "facebookresearch/dinov2",
            DINOV2_MODEL,
            pretrained=True,
            trust_repo=True,
        )
        model.to(DEVICE)
        model.eval()

        _dinov2_transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        _dinov2_model = model
        _dinov2_backend_label = f"facebookresearch/dinov2:{DINOV2_MODEL}"
        logger.info("DINOv2 hazır.")
        return _dinov2_model, _dinov2_transform
    except Exception as e:
        logger.error(f"DINOv2 yüklenemedi: {e}")
        _DINOV2_LOAD_FAILED = True
        _dinov2_model = None
        _dinov2_transform = None
        _dinov2_backend_label = None
        return None, None


def _load_clip():
    """DINOv2 yüklenemezse alternatif görü gömüsü."""
    global _clip_model, _clip_processor
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor

    clip_name = os.getenv("VLM_MODEL_NAME", VLM_MODEL_NAME)
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
    m, _ = _load_dinov2()
    if m is not None:
        return True
    cm, _ = _load_clip()
    return cm is not None


def loaded_backend_name() -> Optional[str]:
    """
    Aktif gömü backend'i (yeni yükleme denemez; sadece önbelleğe bakar).
    """
    if _dinov2_model is not None and _dinov2_transform is not None:
        return _dinov2_backend_label
    if _clip_model is not None and _clip_processor is not None:
        return os.getenv("VLM_MODEL_NAME", VLM_MODEL_NAME)
    return None


def _to_feature_tensor(x) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, torch.Tensor):
                return v
    if hasattr(x, "last_hidden_state"):
        v = getattr(x, "last_hidden_state")
        if isinstance(v, torch.Tensor) and v.dim() >= 3:
            return v.mean(dim=1)
    return None


def extract_features(crop: Image.Image) -> Optional[np.ndarray]:
    model, transform = _load_dinov2()
    if model is not None and transform is not None:
        with torch.no_grad():
            tensor = transform(crop).unsqueeze(0).to(DEVICE)
            feats = _dinov2_cls_embedding(model, tensor)
            if feats is None:
                return None
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()[0]

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
        self.negative_means: dict = {}
        self.negative_counts: dict = defaultdict(int)

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

        neg_penalty = 0.0
        neg_mean = self.negative_means.get(best_label)
        if neg_mean is not None:
            neg_penalty = max(0.0, float(np.dot(feat_norm, neg_mean)))

        conf = (best_sim + 1.0) / 2.0
        conf = max(0.0, min(1.0, conf - 0.25 * neg_penalty))
        return best_label, conf

    def update_negative(self, label: str, feature: np.ndarray):
        n = self.negative_counts[label]
        if label not in self.negative_means:
            self.negative_means[label] = feature.copy()
        else:
            self.negative_means[label] = (self.negative_means[label] * n + feature) / (n + 1)
            norm = np.linalg.norm(self.negative_means[label])
            if norm > 0:
                self.negative_means[label] /= norm
        self.negative_counts[label] += 1

    def apply_feedback(
        self,
        feature: np.ndarray,
        predicted_label: str,
        accepted: bool,
        corrected_label: Optional[str] = None,
    ) -> bool:
        if accepted:
            target = (corrected_label or predicted_label or "").strip()
            if not target:
                return False
            self.update(target, feature)
            return True

        wrong = (predicted_label or "").strip()
        if wrong:
            self.update_negative(wrong, feature)

        corrected = (corrected_label or "").strip()
        if corrected:
            self.update(corrected, feature)

        return bool(wrong or corrected)

    @property
    def known_classes(self) -> list:
        return list(self.class_means.keys())


buffer = MIRBuffer(max_size=BUFFER_SIZE)
classifier = NCMClassifier()


def save_online_memory() -> None:
    """NCM + buffer'ı diske yazar (atomik)."""
    if not ONLINE_MEMORY_ENABLED:
        return
    path = Path(ONLINE_MEMORY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "class_means": {k: np.asarray(v, dtype=np.float32) for k, v in classifier.class_means.items()},
        "class_counts": dict(classifier.class_counts),
        "negative_means": {k: np.asarray(v, dtype=np.float32) for k, v in classifier.negative_means.items()},
        "negative_counts": dict(classifier.negative_counts),
        "buffer_features": [np.asarray(f, dtype=np.float32) for f in buffer.features],
        "buffer_labels": list(buffer.labels),
        "buffer_n_seen": buffer._n_seen,
    }
    tmp = path.with_suffix(".pkl.tmp")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        logger.debug(f"Online hafıza kaydedildi: {path}")
    except Exception as e:
        logger.warning(f"Online hafıza kaydedilemedi: {e}")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_online_memory() -> None:
    """Diskteki NCM + buffer'ı yükler; dosya yoksa mevcut boş durum kalır."""
    global buffer, classifier
    if not ONLINE_MEMORY_ENABLED:
        return
    path = Path(ONLINE_MEMORY_PATH)
    if not path.is_file():
        logger.info("Kalıcı online hafıza dosyası yok, boş başlanıyor.")
        return
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
    except Exception as e:
        logger.warning(f"Kalıcı hafıza okunamadı: {e}")
        return

    clf = NCMClassifier()
    means = state.get("class_means") or {}
    clf.class_means = {str(k): np.asarray(v, dtype=np.float32) for k, v in means.items()}
    clf.class_counts = defaultdict(int, state.get("class_counts") or {})
    neg_m = state.get("negative_means") or {}
    clf.negative_means = {str(k): np.asarray(v, dtype=np.float32) for k, v in neg_m.items()}
    clf.negative_counts = defaultdict(int, state.get("negative_counts") or {})

    buf = MIRBuffer(max_size=BUFFER_SIZE)
    feats = state.get("buffer_features") or []
    buf.features = [np.asarray(x, dtype=np.float32) for x in feats]
    buf.labels = list(state.get("buffer_labels") or [])
    buf._n_seen = int(state.get("buffer_n_seen", len(buf.features)))

    classifier = clf
    buffer = buf
    logger.info(f"Kalıcı online hafıza yüklendi: {path} sınıflar={classifier.known_classes}")


def reset_online_memory() -> None:
    """
    NCM + replay buffer sıfırlanır; kalıcı dosya silinir.
    DINOv2 ağırlıkları değişmez.
    """
    global buffer, classifier
    buffer = MIRBuffer(max_size=BUFFER_SIZE)
    classifier = NCMClassifier()
    if ONLINE_MEMORY_ENABLED:
        p = Path(ONLINE_MEMORY_PATH)
        if p.is_file():
            try:
                p.unlink()
            except OSError as e:
                logger.warning(f"Hafıza dosyası silinemedi: {e}")
    logger.info("Online hafıza sıfırlandı (NCM + buffer + disk).")


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
