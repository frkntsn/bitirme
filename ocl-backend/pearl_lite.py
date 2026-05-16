"""
PEARL-lite: DINOv2 son bloklarında LoRA + SVD ile dinamik rank.

Yeni bir sınıf (known_class) ilk kez öğrenildiğinde:
  1) Kısa LoRA fine-tune (görev vektörü ≈ LoRA delta)
  2) Katman bazında SVD + PEARL Eq.3–4 basitleştirilmiş rank seçimi
  3) LoRA'yı seçilen rank'e indirgeme
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """nn.Linear üzerine düşük-rank adaptasyon (taban ağırlıklar donuk)."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 16.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank >= 1 gerekli")
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        dev = linear.weight.device

        self.lora_A = nn.Parameter(torch.zeros(rank, in_f, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)
        base = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base + delta * self.scaling

    def delta_matrix(self) -> torch.Tensor:
        """Tam delta W ≈ B @ A  (out × in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def set_delta_from_svd(self, k: int, delta: np.ndarray) -> None:
        """delta üzerinde SVD; LoRA'yı rank k ile yeniden başlat."""
        k = max(1, min(k, self.rank, min(delta.shape) - 1 if min(delta.shape) > 1 else 1))
        try:
            U, s, Vt = np.linalg.svd(delta, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        k = min(k, len(s))
        if k < 1:
            return
        Uk = torch.from_numpy(U[:, :k].astype(np.float32)).to(self.lora_B.device)
        sk = torch.from_numpy(s[:k].astype(np.float32)).to(self.lora_B.device)
        Vtk = torch.from_numpy(Vt[:k, :].astype(np.float32)).to(self.lora_A.device)
        with torch.no_grad():
            self.lora_B.zero_()
            self.lora_A.zero_()
            r = min(self.rank, k)
            self.lora_B[:, :r] = Uk[:, :r] * sk[:r].unsqueeze(0)
            self.lora_A[:r, :] = Vtk[:r, :]
            # Fazla sütunları sıfırla (rank < r_max)
            if r < self.rank:
                self.lora_B[:, r:].zero_()
                self.lora_A[r:, :].zero_()

    def effective_rank(self) -> int:
        with torch.no_grad():
            s = torch.linalg.svdvals(self.delta_matrix())
            total = (s ** 2).sum().item() + 1e-9
            cum = torch.cumsum(s ** 2, dim=0) / total
            idx = (cum >= 0.9).nonzero()
            if len(idx) == 0:
                return 1
            return int(idx[0].item()) + 1


def attach_lora_to_dinov2(
    model: nn.Module,
    num_last_blocks: int = 2,
    rank_max: int = 16,
    alpha: float = 16.0,
) -> Dict[str, LoRALinear]:
    """Son `num_last_blocks` bloktaki attn.qkv katmanlarına LoRA tak."""
    if not hasattr(model, "blocks"):
        logger.warning("DINOv2 blocks bulunamadı; PEARL-lite atlanıyor.")
        return {}
    blocks = model.blocks
    n = len(blocks)
    start = max(0, n - num_last_blocks)
    modules: Dict[str, LoRALinear] = {}
    for i in range(start, n):
        attn = blocks[i].attn
        if not hasattr(attn, "qkv") or isinstance(attn.qkv, LoRALinear):
            continue
        key = f"blocks.{i}.attn.qkv"
        lora = LoRALinear(attn.qkv, rank=rank_max, alpha=alpha)
        attn.qkv = lora
        modules[key] = lora
    logger.info(f"PEARL-lite LoRA: {len(modules)} katman (son {num_last_blocks} blok).")
    return modules


def _proximity_threshold(Wc: np.ndarray, Wr: np.ndarray, Wt: np.ndarray) -> float:
    """PEARL Eq.3 basitleştirilmiş: T ∈ [0, 1]."""
    num = float(np.sum(Wc ** 2))
    den = float(np.sum(Wt ** 2) + np.sum(Wr ** 2)) + 1e-9
    return min(1.0, num / den)


def dynamic_rank_from_svd(
    Wc: np.ndarray,
    Wr: np.ndarray,
    Wt: np.ndarray,
    rank_max: int,
) -> int:
    """PEARL Eq.4: kümülatif singular-value varyansı ≥ T."""
    try:
        _, s, _ = np.linalg.svd(Wc, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1
    if len(s) == 0:
        return 1
    T = _proximity_threshold(Wc, Wr, Wt)
    total = float(np.sum(s ** 2)) + 1e-9
    cum = np.cumsum(s ** 2) / total
    k = int(np.searchsorted(cum, T, side="left")) + 1
    return max(1, min(k, rank_max, len(s)))


def _contrastive_loss(
    feats: torch.Tensor,
    pos_proto: torch.Tensor,
    neg_protos: List[torch.Tensor],
    margin: float = 0.2,
) -> torch.Tensor:
    feats = F.normalize(feats, dim=-1)
    pos = F.normalize(pos_proto.unsqueeze(0), dim=-1)
    pos_sim = (feats * pos).sum(dim=-1)
    loss = -pos_sim.mean()
    if neg_protos:
        neg = torch.stack([F.normalize(p, dim=-1) for p in neg_protos], dim=0)
        neg_sim = feats @ neg.T
        loss = loss + F.relu(neg_sim - pos_sim.unsqueeze(1) + margin).mean()
    return loss


class PearlLiteManager:
    def __init__(
        self,
        model: nn.Module,
        lora_modules: Dict[str, LoRALinear],
        embed_fn,
        device: str,
        rank_max: int = 16,
        train_steps: int = 12,
        lr: float = 1e-4,
        min_crops: int = 1,
    ):
        self.model = model
        self.lora_modules = lora_modules
        self.embed_fn = embed_fn
        self.device = device
        self.rank_max = rank_max
        self.train_steps = train_steps
        self.lr = lr
        self.min_crops = min_crops
        self.adapted_classes: set = set()

    def state_dict(self) -> dict:
        return {
            "adapted_classes": list(self.adapted_classes),
            "lora": {k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()} for k, m in self.lora_modules.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self.adapted_classes = set(state.get("adapted_classes") or [])
        lora_state = state.get("lora") or {}
        for key, m in self.lora_modules.items():
            if key not in lora_state:
                continue
            entry = lora_state[key]
            A, B = entry.get("A"), entry.get("B")
            if A is None or B is None:
                continue
            with torch.no_grad():
                r = min(m.rank, A.shape[0], B.shape[1])
                m.lora_A[:r].copy_(A[:r].to(m.lora_A.device))
                m.lora_B[:, :r].copy_(B[:, :r].to(m.lora_B.device))

    def _lora_snapshot(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        snap = {}
        for key, m in self.lora_modules.items():
            snap[key] = (m.lora_A.detach().clone(), m.lora_B.detach().clone())
        return snap

    def _apply_svd_rank_shrink(self, ref_snap: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for key, m in self.lora_modules.items():
            A_ref, B_ref = ref_snap.get(key, (None, None))
            with torch.no_grad():
                delta_now = m.delta_matrix().cpu().numpy()
                if A_ref is not None and B_ref is not None:
                    delta_ref = (B_ref @ A_ref).cpu().numpy() * m.scaling
                    Wc = delta_now - delta_ref
                    Wr = delta_ref
                    Wt = delta_now
                else:
                    Wc = delta_now
                    Wr = np.zeros_like(Wc)
                    Wt = delta_now
            k = dynamic_rank_from_svd(Wc, Wr, Wt, m.rank)
            m.set_delta_from_svd(k, Wc)
            logger.info(f"PEARL-lite SVD {key}: rank→{k} (max={m.rank})")

    def adapt_new_class(
        self,
        label: str,
        crops: List[Image.Image],
        class_means: dict,
        exclude_label: Optional[str] = None,
    ) -> bool:
        """
        Yeni sınıf için kısa LoRA eğitimi + SVD rank seçimi.
        `exclude_label`: henüz güncellenmemiş prototip (genelde yeni sınıf).
        """
        if label in self.adapted_classes:
            return False
        if len(crops) < self.min_crops:
            logger.info(f"PEARL-lite '{label}': yeterli crop yok ({len(crops)}<{self.min_crops})")
            return False
        if not self.lora_modules:
            return False

        transform = getattr(self, "_transform", None)
        if transform is None:
            logger.warning("PEARL-lite: transform yok, adaptasyon atlandı.")
            return False

        tensors = []
        for c in crops[:32]:
            try:
                tensors.append(transform(c))
            except Exception:
                continue
        if not tensors:
            return False

        batch_all = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            feats_init = self.embed_fn(self.model, batch_all)
            if feats_init is None:
                return False
            pos_proto = feats_init.mean(dim=0)

        neg_protos: List[torch.Tensor] = []
        for lbl, mean in (class_means or {}).items():
            if lbl in (exclude_label, label):
                continue
            neg_protos.append(
                torch.from_numpy(np.asarray(mean, dtype=np.float32)).to(self.device)
            )

        ref_snap = self._lora_snapshot()
        trainable = [p for m in self.lora_modules.values() for p in (m.lora_A, m.lora_B) if p.requires_grad]
        if not trainable:
            return False

        self.model.train()
        opt = torch.optim.Adam(trainable, lr=self.lr)

        for step in range(self.train_steps):
            idx = torch.randint(0, batch_all.size(0), (min(8, batch_all.size(0)),))
            batch = batch_all[idx]
            opt.zero_grad(set_to_none=True)
            feats = self.embed_fn(self.model, batch)
            if feats is None:
                continue
            loss = _contrastive_loss(feats, pos_proto, neg_protos)
            loss.backward()
            opt.step()

        self.model.eval()
        self._apply_svd_rank_shrink(ref_snap)
        self.adapted_classes.add(label)
        logger.info(f"PEARL-lite adapt tamam: '{label}' ({self.train_steps} adım)")
        return True
