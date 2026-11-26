"""F-YOLO-style fuzzy detector used in the RATFA experiments.

The official F-YOLO code and weights are not public, so this module re-implements
the three F-YOLO components described in the project report on top of a
COCO-pretrained one-stage YOLO detector (Ultralytics YOLOv8n):

  * GPBlock  - global pooling context block on the deepest backbone feature.
  * FFM      - Fuzzy Fusion Module: weights every input of a neck concatenation
               by Gaussian fuzzy memberships of its global feature energy.
  * FA       - Fuzzy Attention: channel attention driven by Gaussian fuzzy
               memberships of the channel descriptors (P3/P4/P5 outputs).

Every fuzzy unit owns K Gaussian membership functions
    mu_k(z) = exp(-(z - c_k)^2 / (2 sigma_k^2))
whose centres ``c`` and widths ``sigma`` are the *adaptable fuzzy parameters*
that the RATFA controller is allowed to move at run time.

All modules start close to the identity (rule weights ~ 0, so the fuzzy gates
output ~0.5), so the pretrained detector is preserved at the start of fine-tuning,
while FA / FFM have an active gain (scale > 0) so the memberships are trained.
"""
from __future__ import annotations

import glob
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn

K_DEFAULT = 3


class _FuzzySet(nn.Module):
    """K Gaussian membership functions over a standardised scalar descriptor."""

    def __init__(self, n_features: int, k: int = K_DEFAULT):
        super().__init__()
        self.centres = nn.Parameter(torch.linspace(-1.0, 1.0, k))
        self.sigmas = nn.Parameter(torch.full((k,), 0.6))
        # reference statistics of the descriptor on clear training data
        self.register_buffer("ref_mean", torch.zeros(n_features))
        self.register_buffer("ref_std", torch.ones(n_features))
        self.calibrating = False
        self._calib: List[torch.Tensor] = []

    def standardise(self, d: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            self._calib.append(d.detach().reshape(-1, d.shape[-1]).cpu())
        return (d - self.ref_mean) / self.ref_std

    def finish_calibration(self) -> None:
        data = torch.cat(self._calib, 0)
        self.ref_mean.copy_(data.mean(0))
        self.ref_std.copy_(data.std(0).clamp_min(1e-4))
        self._calib, self.calibrating = [], False

    def membership(self, z: torch.Tensor) -> torch.Tensor:
        sig = self.sigmas.abs().clamp_min(1e-3)
        return torch.exp(-((z.unsqueeze(-1) - self.centres) ** 2) / (2 * sig ** 2))


class FuzzyAttention(nn.Module):
    """FA: fuzzy channel attention  y = x * (1 + s * (2a - 1))."""

    def __init__(self, channels: int, k: int = K_DEFAULT):
        super().__init__()
        self.fuzzy = _FuzzySet(channels, k)
        self.rule_w = nn.Parameter(torch.randn(channels, k) * 0.5)
        self.rule_b = nn.Parameter(torch.zeros(channels))
        # s = 1: y = 2a * x, an active fuzzy attention that is the identity when a = 0.5
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        z = self.fuzzy.standardise(x.mean((2, 3)))             # B,C
        mu = self.fuzzy.membership(z)                           # B,C,K
        a = torch.sigmoid((mu * self.rule_w).sum(-1) + self.rule_b)
        return x * (1 + self.scale * (2 * a - 1))[..., None, None]


class FuzzyFusion(nn.Module):
    """FFM: fuzzy weighting of each concatenation input before fusion."""

    def __init__(self, n_inputs: int = 2, k: int = K_DEFAULT):
        super().__init__()
        self.fuzzy = _FuzzySet(n_inputs, k)
        self.rule_w = nn.Parameter(torch.randn(n_inputs, k) * 0.5)
        self.rule_b = nn.Parameter(torch.zeros(n_inputs))
        self.scale = nn.Parameter(torch.full((n_inputs,), 0.5))

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        energy = torch.stack([x.abs().mean((1, 2, 3)) for x in xs], -1)   # B,N
        mu = self.fuzzy.membership(self.fuzzy.standardise(energy))         # B,N,K
        g = torch.sigmoid((mu * self.rule_w).sum(-1) + self.rule_b)        # B,N
        w = 1 + self.scale * (2 * g - 1)
        return [x * w[:, i, None, None, None] for i, x in enumerate(xs)]


class GPBlock(nn.Module):
    """Global pooling context block: y = x + s * MLP(GAP(x))."""

    def __init__(self, channels: int, r: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, channels // r, 1), nn.SiLU(),
                                 nn.Conv2d(channels // r, channels, 1))
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.scale * self.mlp(x.mean((2, 3), keepdim=True))


class _Wrapped(nn.Module):
    """Wraps an Ultralytics layer and keeps the routing attributes (f, i, type, np)."""

    def __init__(self, layer: nn.Module, pre: nn.Module = None, post: nn.Module = None):
        super().__init__()
        self.layer, self.pre, self.post = layer, pre, post
        for a in ("f", "i", "type", "np"):
            if hasattr(layer, a):
                setattr(self, a, getattr(layer, a))

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        x = self.layer(x)
        if self.post is not None:
            x = self.post(x)
        return x


def _out_channels(layer: nn.Module) -> int:
    for name in ("cv2", "cv1", "conv"):
        if hasattr(layer, name):
            sub = getattr(layer, name)
            conv = getattr(sub, "conv", sub)
            if isinstance(conv, nn.Conv2d):
                return conv.out_channels
    convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1].out_channels


def insert_fuzzy_modules(det_model: nn.Module) -> nn.Module:
    """Insert GPBlock / FFM / FA into an Ultralytics DetectionModel (in place)."""
    seq = det_model.model
    if getattr(det_model, "fuzzy_inserted", False):
        return det_model
    detect = seq[-1]
    head_inputs = list(detect.f)                       # P3, P4, P5 outputs
    for idx, layer in enumerate(seq):
        name = layer.type.split(".")[-1]
        if name == "SPPF":
            seq[idx] = _Wrapped(layer, post=GPBlock(_out_channels(layer)))
        elif name == "Concat":
            seq[idx] = _Wrapped(layer, pre=FuzzyFusion(len(layer.f)))
        elif idx in head_inputs:
            seq[idx] = _Wrapped(layer, post=FuzzyAttention(_out_channels(layer)))
    det_model.fuzzy_inserted = True
    return det_model


def fuzzy_sets(model: nn.Module) -> Dict[str, _FuzzySet]:
    """All adaptable fuzzy membership sets, keyed by a stable module name."""
    return {n: m for n, m in model.named_modules() if isinstance(m, _FuzzySet)}


def _letterbox(img: np.ndarray, size: int) -> torch.Tensor:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    t = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    return t


@torch.no_grad()
def calibrate_reference(det_model: nn.Module, image_paths: List[str], imgsz: int = 512) -> None:
    """Estimate reference descriptor statistics on clear training images."""
    sets = fuzzy_sets(det_model)
    for s in sets.values():
        s.calibrating = True
    was_training = det_model.training
    det_model.eval()
    p = next(det_model.parameters())
    for path in image_paths:
        img = cv2.imread(path)
        det_model(_letterbox(img, imgsz)[None].to(p.device, p.dtype))
    for s in sets.values():
        s.finish_calibration()
    det_model.train(was_training)


# ---------------------------------------------------------------------------
# Ultralytics trainer that builds the fuzzy model
# ---------------------------------------------------------------------------
def make_fuzzy_trainer(calib_glob: str, n_calib: int = 64, imgsz: int = 512):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class FuzzyDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            insert_fuzzy_modules(model)
            paths = sorted(glob.glob(calib_glob))[:: max(1, len(glob.glob(calib_glob)) // n_calib)]
            calibrate_reference(model, paths[:n_calib], imgsz)
            return model

    return FuzzyDetectionTrainer
