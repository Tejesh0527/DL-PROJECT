"""
detector/preprocessing.py
==========================
Image preprocessing utilities for the RATFA detector pipeline.

Preprocessing matches the input specification of the reference detector
(640 × 640 letterbox, normalised to [0, 1]) as described in the paper for
F-YOLO (which inherits its preprocessing from YOLOv6).
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image to target_size with letterbox padding.

    Preserves aspect ratio by padding with a constant colour.

    Parameters
    ----------
    image : np.ndarray
        BGR input image (H × W × 3).
    target_size : (int, int)
        (height, width) target dimensions.
    color : (int, int, int)
        Padding colour in BGR.
    auto : bool
        If True, pad to nearest stride multiple.
    stride : int
        Stride used when auto=True.

    Returns
    -------
    img_lb : np.ndarray
        Letterboxed image.
    scale : float
        Scale factor applied to the original image.
    padding : (int, int)
        (pad_top, pad_left) offsets in pixels.
    """
    h0, w0 = image.shape[:2]
    h1, w1 = target_size

    scale = min(h1 / h0, w1 / w0)
    new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))

    if (new_h, new_w) != (h0, w0):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_h = h1 - new_h
    pad_w = w1 - new_w
    if auto:
        pad_h = pad_h % stride
        pad_w = pad_w % stride

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_lb = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img_lb, scale, (top, left)


def normalise(image: np.ndarray) -> np.ndarray:
    """Normalise a BGR uint8 image to float32 in [0, 1]."""
    return image.astype(np.float32) / 255.0


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB channel order."""
    return image[:, :, ::-1]


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """Transpose H × W × C → C × H × W."""
    return np.transpose(image, (2, 0, 1))


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Full preprocessing pipeline: letterbox → normalise → CHW.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 frame (H × W × 3).
    target_size : (int, int)
        Target (height, width).

    Returns
    -------
    tensor : np.ndarray
        float32 array of shape (1, 3, H, W) ready for model input.
    scale : float
        Letterbox scale factor.
    padding : (int, int)
        (pad_top, pad_left) offsets.
    """
    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be a 3-D NumPy array (H, W, C).")

    lb, scale, padding = letterbox(frame, target_size)
    img = bgr_to_rgb(lb)
    img = normalise(img)
    img = hwc_to_chw(img)
    tensor = np.expand_dims(img, axis=0)  # (1, 3, H, W)
    return tensor, scale, padding


def scale_boxes(
    boxes: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    """Map boxes from letterbox space back to original image coordinates.

    Parameters
    ----------
    boxes : np.ndarray
        Array of shape (N, 4) in (x1, y1, x2, y2) format.
    scale : float
        Scale factor returned by letterbox().
    padding : (int, int)
        (pad_top, pad_left) returned by letterbox().
    orig_shape : (int, int)
        Original (height, width) of the frame.

    Returns
    -------
    np.ndarray
        Clipped boxes in original image coordinates.
    """
    if boxes.size == 0:
        return boxes

    pad_top, pad_left = padding
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    h_orig, w_orig = orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_orig)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_orig)
    return boxes


def validate_frame(frame: np.ndarray) -> None:
    """Raise ValueError if frame is not a valid BGR image."""
    if frame is None:
        raise ValueError("frame is None.")
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be np.ndarray; got {type(frame)}.")
    if frame.ndim != 3:
        raise ValueError(f"frame must have 3 dimensions; got {frame.ndim}.")
    if frame.shape[2] != 3:
        raise ValueError(f"frame must have 3 channels; got {frame.shape[2]}.")

