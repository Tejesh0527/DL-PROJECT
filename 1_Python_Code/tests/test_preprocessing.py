"""
tests/test_preprocessing.py
============================
Comprehensive unit tests for detector/preprocessing.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from detector.preprocessing import (
    letterbox,
    normalise,
    bgr_to_rgb,
    hwc_to_chw,
    preprocess_frame,
    scale_boxes,
    validate_frame,
)


class TestLetterbox:

    def test_letterbox_landscape_image(self):
        # 480x640 landscape
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        lb, scale, (top, left) = letterbox(img, target_size=(640, 640))
        assert lb.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert left == 0
        assert top == 80

    def test_letterbox_portrait_image(self):
        # 640x320 portrait
        img = np.full((640, 320, 3), 120, dtype=np.uint8)
        lb, scale, (top, left) = letterbox(img, target_size=(640, 640))
        assert lb.shape == (640, 640, 3)
        assert scale == pytest.approx(1.0)
        assert top == 0
        assert left == 160

    def test_letterbox_square_image(self):
        img = np.full((300, 300, 3), 150, dtype=np.uint8)
        lb, scale, (top, left) = letterbox(img, target_size=(640, 640))
        assert lb.shape == (640, 640, 3)
        assert scale == pytest.approx(640 / 300)
        assert top == 0
        assert left == 0

    def test_letterbox_custom_color(self):
        img = np.full((200, 400, 3), 50, dtype=np.uint8)
        lb, scale, (top, left) = letterbox(img, target_size=(400, 400), color=(114, 114, 114))
        assert lb.shape == (400, 400, 3)
        # Check padding color at top-left corner
        assert np.array_equal(lb[0, 0], [114, 114, 114])

    def test_letterbox_auto_stride(self):
        img = np.full((350, 500, 3), 100, dtype=np.uint8)
        lb, scale, (top, left) = letterbox(img, target_size=(640, 640), auto=True, stride=32)
        # Dimensions must be multiples of 32
        assert lb.shape[0] % 32 == 0
        assert lb.shape[1] % 32 == 0


class TestNormalizationAndTransforms:

    def test_normalise_range_and_dtype(self):
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        norm = normalise(img)
        assert norm.dtype == np.float32
        assert norm[0, 0, 0] == pytest.approx(0.0)
        assert norm[0, 0, 1] == pytest.approx(128 / 255.0)
        assert norm[0, 0, 2] == pytest.approx(1.0)

    def test_bgr_to_rgb(self):
        bgr = np.array([[[255, 128, 0]]], dtype=np.uint8)  # Blue=255, Green=128, Red=0
        rgb = bgr_to_rgb(bgr)
        assert rgb[0, 0, 0] == 0    # Red
        assert rgb[0, 0, 1] == 128  # Green
        assert rgb[0, 0, 2] == 255  # Blue

    def test_hwc_to_chw(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        chw = hwc_to_chw(img)
        assert chw.shape == (3, 100, 200)


class TestPreprocessFrame:

    def test_preprocess_frame_valid(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tensor, scale, padding = preprocess_frame(frame, target_size=(640, 640))
        assert tensor.ndim == 4
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert 0.0 <= np.min(tensor) and np.max(tensor) <= 1.0
        assert scale > 0
        assert isinstance(padding, tuple) and len(padding) == 2

    def test_preprocess_frame_invalid_raises(self):
        with pytest.raises(ValueError, match="3-D NumPy array"):
            preprocess_frame(None)

        with pytest.raises(ValueError, match="3-D NumPy array"):
            preprocess_frame(np.zeros((100, 100), dtype=np.uint8))


class TestScaleBoxes:

    def test_scale_boxes_inversion(self):
        orig_shape = (480, 640)
        scale = 1.0
        padding = (80, 0)  # pad_top=80, pad_left=0
        # Box in letterboxed 640x640 space: top starts at 80 (y=0 in orig)
        boxes = np.array([
            [10.0, 80.0, 100.0, 180.0]
        ])
        scaled = scale_boxes(boxes, scale, padding, orig_shape)
        # Expected in original: y1 = 80 - 80 = 0, y2 = 180 - 80 = 100
        assert scaled[0, 0] == pytest.approx(10.0)
        assert scaled[0, 1] == pytest.approx(0.0)
        assert scaled[0, 2] == pytest.approx(100.0)
        assert scaled[0, 3] == pytest.approx(100.0)

    def test_scale_boxes_clipping(self):
        orig_shape = (480, 640)
        scale = 1.0
        padding = (0, 0)
        boxes = np.array([
            [-50.0, -20.0, 700.0, 500.0]
        ])
        scaled = scale_boxes(boxes, scale, padding, orig_shape)
        assert scaled[0, 0] == pytest.approx(0.0)
        assert scaled[0, 1] == pytest.approx(0.0)
        assert scaled[0, 2] == pytest.approx(640.0)
        assert scaled[0, 3] == pytest.approx(480.0)

    def test_scale_boxes_empty(self):
        boxes = np.empty((0, 4))
        scaled = scale_boxes(boxes, 1.0, (0, 0), (480, 640))
        assert scaled.shape == (0, 4)


class TestValidateFrame:

    def test_validate_frame_valid(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        validate_frame(frame)  # Should not raise

    def test_validate_frame_none_raises(self):
        with pytest.raises(ValueError, match="frame is None"):
            validate_frame(None)

    def test_validate_frame_non_array_raises(self):
        with pytest.raises(TypeError, match="frame must be np.ndarray"):
            validate_frame([[1, 2], [3, 4]])

    def test_validate_frame_2d_raises(self):
        with pytest.raises(ValueError, match="frame must have 3 dimensions"):
            validate_frame(np.zeros((100, 100), dtype=np.uint8))

    def test_validate_frame_wrong_channels_raises(self):
        with pytest.raises(ValueError, match="frame must have 3 channels"):
            validate_frame(np.zeros((100, 100, 4), dtype=np.uint8))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

