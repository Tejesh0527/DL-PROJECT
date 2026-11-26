"""Trained scene-condition classifier (clear / rain / fog / dark).

Hand-crafted image statistics (brightness, contrast, sharpness, edge energy,
dark-channel haze cue, HSV histograms) feed a Random-Forest classifier.
The scene confidence q_t is the top class probability; the full probability
vector is the condition vector e_t used by the causal smoother (Eq. 2).
"""
import pickle
from typing import Tuple

import cv2
import numpy as np

from scene_analysis.feature_extractor import extract_scene_features

CONDITIONS = ["clear", "rain", "fog", "dark"]


def scene_feature_vector(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (320, 180))
    base = extract_scene_features(small)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark_channel = cv2.erode(small.min(axis=2), np.ones((7, 7), np.uint8))
    h = gray.shape[0]
    feats = [base["mean_brightness"], base["contrast"], np.log1p(base["laplacian_var"]),
             base["dark_pixel_frac"], base["vertical_edge_energy"],
             dark_channel.mean(), np.percentile(dark_channel, 90),
             gray[: h // 3].mean() - gray[2 * h // 3:].mean(),
             hsv[..., 1].mean(), hsv[..., 1].std(), np.percentile(gray, 5), np.percentile(gray, 95)]
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    feats += [np.abs(sx).mean(), np.abs(sy).mean(), (np.abs(sy) / (np.abs(sx) + 1)).mean()]
    for ch, rng in ((hsv[..., 2], 256), (hsv[..., 1], 256)):
        hist = np.histogram(ch, bins=8, range=(0, rng))[0].astype(np.float32)
        feats += list(hist / hist.sum())
    return np.asarray(feats, np.float32)


class SceneClassifier:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=300, random_state=42,
                                            class_weight="balanced", n_jobs=-1)
        self.model.fit(X, y)
        return self

    def predict_proba_img(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        p = self.model.predict_proba(scene_feature_vector(img)[None])[0]
        e = np.zeros(len(CONDITIONS))
        for cls, prob in zip(self.model.classes_, p):
            e[CONDITIONS.index(cls)] = prob
        return e, float(e.max())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))
