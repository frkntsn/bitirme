"""
VLM-CL Modeli — UNI2-h Backbone
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import BUFFER_SIZE, DEVICE, HF_TOKEN

logger = logging.getLogger(__name__)

UNI2_MODEL_NAME = "MahmoodLab/uni2-h"
_model = None
_transform = None


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
        _model = timm.create_model(
            "hf-hub:MahmoodLab/uni2-h",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        _model.to(DEVICE)
        _model.eval()
        _transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        logger.info("UNI2-h hazır.")
        return _model, _transform
    except Exception as e:
        logger.error(f"UNI2-h yüklenemedi: {e}")
        return None, None


def is_loaded() -> bool:
    m, _ = _load_uni2()
    return m is not None


def extract_features(crop: Image.Image) -> Optional[np.ndarray]:
    model, transform = _load_uni2()
    if model is None:
        return None
    with torch.no_grad():
        tensor = transform(crop).unsqueeze(0).to(DEVICE)
        feats = model(tensor)
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
