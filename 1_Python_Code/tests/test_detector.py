"""
tests/test_detector.py
========================
Tests for detector loading, inference, and error handling.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from detector.fyolo_detector import (
    StubDetector,
    FYOLODetector,
    create_detector,
    Detection,
)
from detector.vehicle_detection import (
    detect_vehicles,
    draw_detections,
    compute_reliability,
    compute_iou,
    apply_nms,
)


def make_frame(h=480, w=640, brightness=120):
    rng = np.random.default_rng(42)
    return rng.integers(brightness - 20, brightness + 20, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# StubDetector tests
# ---------------------------------------------------------------------------

class TestStubDetector:

    def setup_method(self):
        self.detector = StubDetector()
        self.detector.load_model()

    def test_is_loaded_after_load_model(self):
        assert self.detector.is_loaded is True

    def test_backend_name(self):
        assert "StubDetector" in self.detector.backend_name
        assert "NOT F-YOLO" in self.detector.backend_name

    def test_detect_returns_list(self):
        frame = make_frame()
        result = self.detector.detect(frame, frame_number=0)
        assert isinstance(result, list)

    def test_detections_have_correct_fields(self):
        frame = make_frame()
        detections = self.detector.detect(frame, frame_number=1)
        for det in detections:
            assert isinstance(det, Detection)
            assert 0 <= det.confidence <= 1.0
            assert det.class_id in [2, 3, 5, 7]
            assert det.class_name in ["car", "motorcycle", "bus", "truck"]
            x1, y1, x2, y2 = det.bbox
            assert x1 < x2 and y1 < y2

    def test_detect_before_load_raises(self):
        det = StubDetector()
        with pytest.raises(RuntimeError, match="load_model"):
            det.detect(make_frame())

    def test_detect_invalid_frame_raises(self):
        with pytest.raises(ValueError):
            self.detector.detect(None)

    def test_detect_2d_frame_raises(self):
        gray = np.zeros((240, 320), dtype=np.uint8)
        with pytest.raises(ValueError):
            self.detector.detect(gray)

    def test_bounding_boxes_within_frame(self):
        frame = make_frame(h=480, w=640)
        detections = self.detector.detect(frame, frame_number=5)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            assert 0 <= x1 < frame.shape[1]
            assert 0 <= y1 < frame.shape[0]
            assert x2 <= frame.shape[1]
            assert y2 <= frame.shape[0]

    def test_deterministic_output(self):
        """Same frame_number → same detections."""
        frame = make_frame()
        d1 = self.detector.detect(frame, frame_number=7)
        d2 = self.detector.detect(frame, frame_number=7)
        assert len(d1) == len(d2)
        for a, b in zip(d1, d2):
            assert a.bbox == pytest.approx(b.bbox, abs=1e-9)

    def test_confidence_above_threshold(self):
        """All returned confidences ≥ conf_threshold."""
        frame = make_frame()
        threshold = 0.25
        det = StubDetector(conf_threshold=threshold)
        det.load_model()
        detections = det.detect(frame)
        for d in detections:
            assert d.confidence >= threshold


# ---------------------------------------------------------------------------
# FYOLODetector tests (BLOCKED)
# ---------------------------------------------------------------------------

class TestFYOLODetectorBlocked:

    def test_missing_checkpoint_raises(self):
        """FYOLODetector should raise FileNotFoundError for missing checkpoint."""
        detector = FYOLODetector(
            checkpoint_path="nonexistent/checkpoint.pt",
            config_path="nonexistent/config.yaml",
        )
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            detector.load_model()

    def test_is_not_loaded_before_load(self):
        detector = FYOLODetector(
            checkpoint_path="nonexistent.pt",
            config_path="nonexistent.yaml",
        )
        assert detector.is_loaded is False

    def test_backend_name_contains_blocked(self):
        detector = FYOLODetector(
            checkpoint_path="nonexistent.pt",
            config_path="",
        )
        assert "BLOCKED" in detector.backend_name


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCreateDetector:

    def test_stub_backend_created(self):
        cfg = {"backend": "stub", "conf_threshold": 0.25, "iou_threshold": 0.45}
        det = create_detector(cfg)
        assert det.is_loaded
        assert "StubDetector" in det.backend_name

    def test_unknown_backend_raises(self):
        cfg = {"backend": "unknown_backend"}
        with pytest.raises(ValueError, match="Unknown detector backend"):
            create_detector(cfg)


# ---------------------------------------------------------------------------
# Vehicle detection utilities
# ---------------------------------------------------------------------------

class TestVehicleDetectionUtils:

    def setup_method(self):
        self.detector = StubDetector()
        self.detector.load_model()
        self.frame = make_frame()

    def test_detect_vehicles_wrapper(self):
        detections = detect_vehicles(self.detector, self.frame)
        assert isinstance(detections, list)

    def test_draw_detections_shape_unchanged(self):
        detections = detect_vehicles(self.detector, self.frame)
        annotated = draw_detections(self.frame, detections)
        assert annotated.shape == self.frame.shape

    def test_draw_detections_different_from_original(self):
        """Annotated frame should differ from original when detections exist."""
        # Use a frame that will produce detections
        frame = make_frame()
        det = StubDetector(seed=42)
        det.load_model()
        detections = det.detect(frame, frame_number=1)
        if detections:
            annotated = draw_detections(frame, detections)
            assert not np.array_equal(annotated, frame)

    def test_detection_to_dict(self):
        det = Detection(
            frame_number=0,
            bbox=(10.0, 20.0, 100.0, 200.0),
            class_id=2,
            class_name="car",
            confidence=0.85,
            timestamp_ms=0.0,
        )
        d = det.to_dict()
        assert d["class_id"] == 2
        assert d["class_name"] == "car"
        assert d["confidence"] == pytest.approx(0.85)
        assert "bbox" in d


# ---------------------------------------------------------------------------
# Reliability computation
# ---------------------------------------------------------------------------

class TestReliabilityComputation:

    def test_no_detections_returns_neutral(self):
        result = compute_reliability([])
        assert result["reliability"] == 0.5
        assert result["valid"] is False

    def test_high_confidence_detections(self):
        detections = [
            Detection(0, (10, 10, 100, 100), 2, "car", 0.95),
            Detection(0, (200, 50, 350, 200), 2, "car", 0.90),
        ]
        result = compute_reliability(detections)
        assert result["valid"] is True
        assert result["reliability"] > 0.5

    def test_low_confidence_detections(self):
        detections = [
            Detection(0, (10, 10, 100, 100), 2, "car", 0.26),
            Detection(0, (200, 50, 350, 200), 2, "car", 0.27),
        ]
        result = compute_reliability(
            detections, low_conf_threshold=0.4
        )
        assert result["valid"] is True
        # Low scores → lower reliability than high-confidence case
        high_result = compute_reliability([
            Detection(0, (10, 10, 100, 100), 2, "car", 0.95),
        ])
        assert result["reliability"] < high_result["reliability"]

    def test_reliability_in_range(self):
        detections = [
            Detection(0, (10, 10, 100, 100), 7, "truck", 0.6),
        ]
        result = compute_reliability(detections)
        assert 0.0 <= result["reliability"] <= 1.0


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (NMS) tests
# ---------------------------------------------------------------------------

class TestNonMaximumSuppression:

    def test_compute_iou_identical_boxes(self):
        box = (10.0, 10.0, 100.0, 100.0)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_compute_iou_non_overlapping(self):
        box1 = (0.0, 0.0, 50.0, 50.0)
        box2 = (60.0, 60.0, 100.0, 100.0)
        assert compute_iou(box1, box2) == pytest.approx(0.0)

    def test_compute_iou_partial_overlap(self):
        # 100x100 box: area 10000
        # shift by 50 in x: intersection is 50x100 = 5000
        # union is 10000 + 10000 - 5000 = 15000
        # IoU = 5000 / 15000 = 1/3
        box1 = (0.0, 0.0, 100.0, 100.0)
        box2 = (50.0, 0.0, 150.0, 100.0)
        assert compute_iou(box1, box2) == pytest.approx(1.0 / 3.0)

    def test_apply_nms_overlapping_same_class(self):
        # Two heavily overlapping car boxes (IoU > 0.8)
        d1 = Detection(0, (10.0, 10.0, 100.0, 100.0), 2, "car", 0.90)
        d2 = Detection(0, (12.0, 11.0, 102.0, 101.0), 2, "car", 0.70)
        kept = apply_nms([d1, d2], iou_threshold=0.45)
        assert len(kept) == 1
        assert kept[0].confidence == pytest.approx(0.90)

    def test_apply_nms_non_overlapping(self):
        d1 = Detection(0, (10.0, 10.0, 50.0, 50.0), 2, "car", 0.85)
        d2 = Detection(0, (100.0, 100.0, 200.0, 200.0), 2, "car", 0.80)
        kept = apply_nms([d1, d2], iou_threshold=0.45)
        assert len(kept) == 2

    def test_apply_nms_per_class(self):
        # Two overlapping boxes of different classes: car and truck
        d_car = Detection(0, (10.0, 10.0, 100.0, 100.0), 2, "car", 0.88)
        d_truck = Detection(0, (12.0, 12.0, 102.0, 102.0), 7, "truck", 0.85)
        # Per-class NMS retains both
        kept = apply_nms([d_car, d_truck], iou_threshold=0.45, per_class=True)
        assert len(kept) == 2

        # Class-agnostic NMS suppresses the lower-confidence one
        kept_agnostic = apply_nms([d_car, d_truck], iou_threshold=0.45, per_class=False)
        assert len(kept_agnostic) == 1
        assert kept_agnostic[0].class_name == "car"

    def test_apply_nms_empty(self):
        assert apply_nms([]) == []

    def test_apply_nms_conf_threshold(self):
        d1 = Detection(0, (0.0, 0.0, 50.0, 50.0), 2, "car", 0.90)
        d2 = Detection(0, (60.0, 60.0, 100.0, 100.0), 2, "car", 0.30)
        kept = apply_nms([d1, d2], conf_threshold=0.50)
        assert len(kept) == 1
        assert kept[0].confidence == pytest.approx(0.90)

    def test_apply_nms_invalid_threshold_raises(self):
        d = Detection(0, (0.0, 0.0, 50.0, 50.0), 2, "car", 0.90)
        with pytest.raises(ValueError, match="iou_threshold must be in"):
            apply_nms([d], iou_threshold=1.5)

        with pytest.raises(ValueError, match="conf_threshold must be in"):
            apply_nms([d], conf_threshold=-0.1)

    def test_apply_nms_invalid_bbox_raises(self):
        # x2 < x1
        bad_det = Detection(0, (50.0, 10.0, 20.0, 100.0), 2, "car", 0.90)
        with pytest.raises(ValueError, match="Invalid bounding box coordinates"):
            apply_nms([bad_det])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

