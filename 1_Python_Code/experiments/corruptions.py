"""Deterministic synthetic adverse-condition corruptions (fog, rain, low light).

Used because the UA-DETRAC subset has no per-sequence weather labels and BDD100K
val contains only 13 foggy images. Each corruption is seeded so results are
reproducible. ``severity`` lets the calibration split use different settings
from the test split.
"""
import cv2
import numpy as np

CONDITIONS = ["clear", "rain", "fog", "dark"]


def fog(img, seed=0, severity=1.0):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    beta = 1.0 * severity + rng.uniform(-0.1, 0.1)
    depth = np.linspace(1.0, 0.25, h)[:, None]           # far (top) -> near (bottom)
    t = np.exp(-beta * depth)[..., None]
    A = 220.0 + rng.uniform(-10, 10)
    return np.clip(img.astype(np.float32) * t + A * (1 - t), 0, 255).astype(np.uint8)


def dark(img, seed=0, severity=1.0):
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0
    x = np.power(x, 1.0 + 1.0 * severity) * 0.6
    x = x * 255.0 + rng.normal(0, 5 * severity, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def rain(img, seed=0, severity=1.0):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    layer = np.zeros((h, w), np.float32)
    n = int(1500 * severity * (h * w) / (960 * 540))
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    length = rng.integers(10, 25, n)
    for x, y, l in zip(xs, ys, length):
        cv2.line(layer, (int(x), int(y)), (int(x + l * 0.25), int(y + l)), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    base = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32) * 0.8
    out = base + layer[..., None] * 150.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply(img, condition, seed=0, severity=1.0):
    if condition == "clear":
        return img
    return {"fog": fog, "dark": dark, "rain": rain}[condition](img, seed, severity)
