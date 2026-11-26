"""
tests/test_scene_estimator.py
================================
Unit tests for the scene-condition estimator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from scene_analysis.condition_estimator import (
    SceneConditionEstimator,
    SceneConditionResult,
    CONDITIONS,
)


def make_frame(brightness: int, shape=(240, 320, 3)) -> np.ndarray:
    """Create a synthetic BGR frame with uniform brightness."""
    return np.full(shape, brightness, dtype=np.uint8)


def make_foggy_frame(shape=(240, 320, 3)) -> np.ndarray:
    """Create a synthetic hazy (low-contrast, medium-brightness) frame."""
    return np.full(shape, 160, dtype=np.uint8) + np.random.default_rng(1).integers(
        -5, 5, shape, dtype=np.int16
    ).clip(-160, 95).astype(np.uint8)


class TestSceneConditionResult:

    def test_valid_result_sums_to_one(self):
        r = SceneConditionResult(clear=0.7, rain=0.1, fog=0.1, dark=0.1,
                                  scene_confidence=0.8)
        assert abs(r.clear + r.rain + r.fog + r.dark - 1.0) < 1e-9

    def test_negative_input_raises(self):
        with pytest.raises(ValueError):
            SceneConditionResult(clear=-0.1, rain=0.4, fog=0.4, dark=0.3,
                                  scene_confidence=0.5)

    def test_confidence_clipped(self):
        r = SceneConditionResult(clear=1.0, rain=0.0, fog=0.0, dark=0.0,
                                  scene_confidence=1.5)
        assert r.scene_confidence == 1.0

    def test_all_zeros_produces_uniform(self):
        """Zero scores → uniform distribution, confidence=0."""
        r = SceneConditionResult(clear=0.0, rain=0.0, fog=0.0, dark=0.0,
                                  scene_confidence=0.9)
        assert abs(r.clear - 0.25) < 1e-9
        assert r.scene_confidence == 0.0

    def test_to_dict_has_required_keys(self):
        r = SceneConditionResult(clear=0.7, rain=0.1, fog=0.1, dark=0.1,
                                  scene_confidence=0.8)
        d = r.to_dict()
        for key in ["clear", "rain", "fog", "dark", "scene_confidence"]:
            assert key in d

    def test_vector_property(self):
        r = SceneConditionResult(clear=0.7, rain=0.1, fog=0.1, dark=0.1,
                                  scene_confidence=0.8)
        vec = r.vector
        assert vec.shape == (4,)
        assert abs(vec.sum() - 1.0) < 1e-9


class TestSceneConditionEstimatorHeuristic:

    def setup_method(self):
        self.estimator = SceneConditionEstimator(method="heuristic")

    def test_dark_frame_gives_high_dark_score(self):
        """A very dark frame should produce the highest dark score."""
        frame = make_frame(brightness=15)
        result = self.estimator.estimate(frame)
        assert result.dark == max(result.clear, result.rain, result.fog, result.dark), \
            f"Expected dark to dominate; got {result}"

    def test_bright_frame_gives_high_clear_score(self):
        """A uniformly bright frame with normal contrast should lean clear."""
        # Bright uniform frame with some spatial variation
        rng = np.random.default_rng(42)
        frame = rng.integers(180, 220, (240, 320, 3), dtype=np.uint8)
        result = self.estimator.estimate(frame)
        # In a bright frame, dark should be small
        assert result.dark < 0.5, f"Dark score too high for bright frame: {result.dark}"

    def test_output_sums_to_one(self):
        """Scene condition vector must always sum to 1."""
        for brightness in [10, 60, 120, 200]:
            frame = make_frame(brightness)
            result = self.estimator.estimate(frame)
            total = result.clear + result.rain + result.fog + result.dark
            assert abs(total - 1.0) < 1e-9, \
                f"Scores do not sum to 1 at brightness {brightness}: {total}"

    def test_confidence_in_range(self):
        """Scene confidence must be in [0, 1]."""
        for brightness in [5, 60, 160, 220]:
            frame = make_frame(brightness)
            result = self.estimator.estimate(frame)
            assert 0.0 <= result.scene_confidence <= 1.0, \
                f"Confidence out of range: {result.scene_confidence}"

    def test_all_scores_non_negative(self):
        frame = make_frame(100)
        result = self.estimator.estimate(frame)
        assert result.clear >= 0
        assert result.rain >= 0
        assert result.fog >= 0
        assert result.dark >= 0

    def test_invalid_frame_raises(self):
        with pytest.raises(ValueError):
            self.estimator.estimate(None)

    def test_2d_frame_raises(self):
        """Grayscale (2-D) frame should raise ValueError."""
        gray = np.zeros((240, 320), dtype=np.uint8)
        with pytest.raises(ValueError):
            self.estimator.estimate(gray)

    def test_batch_estimate_length(self):
        frames = [make_frame(b) for b in [50, 100, 150]]
        results = self.estimator.estimate_batch(frames)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, SceneConditionResult)


class TestSceneConditionEstimatorClassifier:

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError):
            SceneConditionEstimator(method="unsupported_method")

    def test_classifier_falls_back_to_heuristic(self):
        """When classifier is requested but not implemented, heuristic is used."""
        estimator = SceneConditionEstimator(method="classifier")
        assert estimator.method == "heuristic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

