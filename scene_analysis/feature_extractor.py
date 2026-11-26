"""
scene_analysis/feature_extractor.py
=====================================
Low-level image statistics used by the heuristic scene-condition estimator.

These are image-processing heuristics documented as a BASELINE method.
They do NOT claim high accuracy; actual accuracy requires validation against
condition-labelled data.

Extracted features:
  • Mean brightness (per channel and luminance)
  • Global contrast (standard deviation of luminance)
  • Haze index (Laplacian variance – low ⟹ blurred / hazy)
  • Vertical edge energy (high-frequency vertical gradients ⟹ rain-like)
  • Dark-pixel fraction
"""

import logging
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_scene_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract low-level scene statistics from a BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 image (H × W × 3).

    Returns
    -------
    dict
        {
          "mean_brightness":   float,   # mean luminance 0–255
          "contrast":          float,   # std of luminance 0–127
          "laplacian_var":     float,   # Laplacian variance (sharpness)
          "dark_pixel_frac":   float,   # fraction of pixels below 50 lum
          "vertical_edge_energy": float # normalised vertical gradient energy
        }
    """
    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be a 3-D NumPy array (H, W, C).")

    # Convert to luminance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Laplacian variance: high ⟹ sharp, low ⟹ blurred / foggy
    lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    laplacian_var = float(np.var(lap))

    # Dark-pixel fraction
    dark_pixel_frac = float(np.mean(gray < 50))

    # Vertical edge energy via Sobel (rain often creates vertical streaks)
    sobel_v = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
    h_pixels = gray.shape[0] * gray.shape[1]
    # Normalise by image pixel count so the value is scale-invariant
    vertical_edge_energy = float(np.sum(np.abs(sobel_v)) / (h_pixels * 255.0))

    return {
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "laplacian_var": laplacian_var,
        "dark_pixel_frac": dark_pixel_frac,
        "vertical_edge_energy": vertical_edge_energy,
    }

