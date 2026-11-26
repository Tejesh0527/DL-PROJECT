"""
scene_analysis/condition_estimator.py
=======================================
Scene-condition estimator for RATFA.

Implements Equation (1) from the paper:
  e_t = [e_t^clear, e_t^rain, e_t^fog, e_t^dark]

The estimator returns:
  • A scene-condition vector e_t (4 non-negative values summing to 1)
  • A scene-condition confidence q_t ∈ [0, 1]

IMPORTANT: q_t is confidence in the SCENE ESTIMATE, NOT vehicle-detection
confidence.  A low q_t tells the RATFA controller to suppress large fuzzy-
parameter updates (confidence gate, Eq. 10).

Method
------
Two backends are supported:

  "heuristic" (DEFAULT – documented as a BASELINE):
      Uses image-processing statistics (brightness, contrast, Laplacian
      variance, vertical edge energy) to score each condition.  This is an
      engineering approximation; it has NOT been validated with ground-truth
      condition labels and should NOT be used as a production estimator.

  "classifier" (FUTURE):
      A trained CNN classifier.  Placeholder interface only; implementation
      requires a condition-labelled dataset and trained weights.

No accuracy claims are made for the heuristic method without proper validation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from scene_analysis.feature_extractor import extract_scene_features

logger = logging.getLogger(__name__)

CONDITIONS = ["clear", "rain", "fog", "dark"]


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class SceneConditionResult:
    """Container for a single scene-condition estimate."""

    def __init__(
        self,
        clear: float,
        rain: float,
        fog: float,
        dark: float,
        scene_confidence: float,
        raw_features: Optional[Dict] = None,
    ):
        self._vec = np.array(
            [clear, rain, fog, dark], dtype=np.float64
        )
        # Validate non-negative
        if np.any(self._vec < 0):
            raise ValueError(
                f"All condition scores must be non-negative; got {self._vec}"
            )
        self.scene_confidence = float(np.clip(scene_confidence, 0.0, 1.0))
        self.raw_features = raw_features or {}

        # Normalise to sum to 1 when total > 0
        total = self._vec.sum()
        if total > 1e-9:
            self._vec = self._vec / total
        else:
            # Uniform fallback; confidence is 0 in this case
            self._vec = np.ones(4, dtype=np.float64) / 4.0
            self.scene_confidence = 0.0

    @property
    def clear(self) -> float:
        return float(self._vec[0])

    @property
    def rain(self) -> float:
        return float(self._vec[1])

    @property
    def fog(self) -> float:
        return float(self._vec[2])

    @property
    def dark(self) -> float:
        return float(self._vec[3])

    @property
    def vector(self) -> np.ndarray:
        """Normalised 4-element condition vector [clear, rain, fog, dark]."""
        return self._vec.copy()

    def to_dict(self) -> Dict:
        return {
            "clear": self.clear,
            "rain": self.rain,
            "fog": self.fog,
            "dark": self.dark,
            "scene_confidence": self.scene_confidence,
        }

    def __repr__(self) -> str:
        return (
            f"SceneConditionResult(clear={self.clear:.3f}, rain={self.rain:.3f}, "
            f"fog={self.fog:.3f}, dark={self.dark:.3f}, "
            f"q={self.scene_confidence:.3f})"
        )


# ---------------------------------------------------------------------------
# Heuristic estimator (documented as baseline)
# ---------------------------------------------------------------------------


def _heuristic_estimate(
    frame: np.ndarray,
    dark_brightness_threshold: float = 60.0,
    fog_laplacian_threshold: float = 50.0,
    fog_contrast_threshold: float = 30.0,
    rain_edge_threshold: float = 0.15,
) -> SceneConditionResult:
    """Image-processing baseline for scene-condition estimation.

    BASELINE METHOD: heuristic rules; NOT validated with ground-truth labels.

    Heuristic logic (documented explicitly):
      dark  ← mean luminance below dark_brightness_threshold
      fog   ← Laplacian variance below fog_laplacian_threshold  AND
               contrast below fog_contrast_threshold  AND
               mean luminance is not very low (≥ dark threshold)
      rain  ← vertical edge energy above rain_edge_threshold AND
               image is not dark
      clear ← residual evidence after dark/fog/rain evidence is accounted for

    Confidence q_t:
      Derived from the peak score after normalisation; a uniform distribution
      (all conditions equally likely) gives q_t ≈ 0.25.  A dominant single
      condition gives q_t closer to 1.  This is a crude proxy for certainty.
    """
    feats = extract_scene_features(frame)
    b = feats["mean_brightness"]
    contrast = feats["contrast"]
    lap = feats["laplacian_var"]
    edge_v = feats["vertical_edge_energy"]

    # Raw evidence scores (non-negative)
    dark_score = max(0.0, (dark_brightness_threshold - b) / dark_brightness_threshold)

    # Fog: low sharpness + low contrast + medium brightness
    fog_lap_evidence = max(0.0, (fog_laplacian_threshold - lap) / fog_laplacian_threshold)
    fog_contrast_evidence = max(0.0, (fog_contrast_threshold - contrast) / fog_contrast_threshold)
    fog_brightness_evidence = 1.0 - dark_score  # less foggy evidence when it is dark
    fog_score = fog_lap_evidence * fog_contrast_evidence * fog_brightness_evidence

    # Rain: high vertical edge energy when not dark
    rain_score = max(0.0, edge_v - rain_edge_threshold) * (1.0 - dark_score)

    # Clear: residual – some clear evidence is always present
    clear_score = max(0.1, 1.0 - dark_score - fog_score - rain_score)

    # Confidence: measure of how "peaked" the distribution is.
    # We compute it before normalisation as the fraction captured by the top score.
    total = clear_score + rain_score + fog_score + dark_score
    top_score = max(clear_score, rain_score, fog_score, dark_score)
    if total > 1e-9:
        q_raw = top_score / total  # range 0.25 (uniform) to 1.0 (certain)
        # Map [0.25, 1.0] → [0.0, 1.0] for a normalised confidence
        scene_confidence = float(np.clip((q_raw - 0.25) / 0.75, 0.0, 1.0))
    else:
        scene_confidence = 0.0

    result = SceneConditionResult(
        clear=clear_score,
        rain=rain_score,
        fog=fog_score,
        dark=dark_score,
        scene_confidence=scene_confidence,
        raw_features=feats,
    )
    logger.debug("Heuristic scene estimate: %s", result)
    return result


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------


class SceneConditionEstimator:
    """Reusable scene-condition estimator (supports image and video frames).

    Parameters
    ----------
    method : str
        "heuristic" or "classifier".
    heuristic_params : dict, optional
        Override thresholds for the heuristic method.
    """

    SUPPORTED_METHODS = ("heuristic", "classifier")

    def __init__(
        self,
        method: str = "heuristic",
        heuristic_params: Optional[Dict] = None,
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {self.SUPPORTED_METHODS}; got '{method}'"
            )
        self.method = method
        self._heuristic_params = heuristic_params or {}

        if method == "classifier":
            logger.warning(
                "Classifier-based scene estimator is not yet implemented.  "
                "Falling back to heuristic method."
            )
            self.method = "heuristic"

    def estimate(self, frame: np.ndarray) -> SceneConditionResult:
        """Estimate scene condition for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 image (H × W × 3).

        Returns
        -------
        SceneConditionResult
        """
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a 3-D NumPy array (H, W, C).")

        if self.method == "heuristic":
            return _heuristic_estimate(frame, **self._heuristic_params)

        # Placeholder for future classifier
        raise NotImplementedError(
            "Classifier-based estimator is not implemented yet."
        )

    def estimate_batch(
        self, frames: List[np.ndarray]
    ) -> List[SceneConditionResult]:
        """Estimate scene conditions for a list of frames."""
        return [self.estimate(f) for f in frames]

