"""
detector/vehicle_detection.py
==============================
Vehicle detection: annotation, JSON export, and batch utilities.

This module wraps the detector interface with higher-level operations:
  • drawing annotated bounding boxes
  • saving annotated images
  • exporting detection results to JSON
  • computing frame-level reliability descriptor (Eq. 4–5 in the paper)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector.fyolo_detector import BaseDetector, Detection
from detector.preprocessing import validate_frame

logger = logging.getLogger(__name__)

# Colour palette for vehicle classes (BGR)
CLASS_COLOURS: Dict[int, Tuple[int, int, int]] = {
    2: (0, 255, 0),    # car       – green
    3: (255, 128, 0),  # motorcycle – orange
    5: (255, 0, 0),    # bus       – blue
    7: (0, 0, 255),    # truck     – red
}
DEFAULT_COLOUR = (200, 200, 200)


# ---------------------------------------------------------------------------
# Detection runner
# ---------------------------------------------------------------------------


def detect_vehicles(
    detector: BaseDetector,
    frame: np.ndarray,
    frame_number: int = 0,
    timestamp_ms: float = 0.0,
) -> List[Detection]:
    """Run the detector on a single frame and return vehicle detections.

    Parameters
    ----------
    detector : BaseDetector
        Loaded detector instance.
    frame : np.ndarray
        BGR image.
    frame_number : int
        Frame index.
    timestamp_ms : float
        Frame timestamp (ms).

    Returns
    -------
    List[Detection]
    """
    if not detector.is_loaded:
        raise RuntimeError("Detector is not loaded.  Call load_model() first.")
    validate_frame(frame)
    return detector.detect(frame, frame_number=frame_number, timestamp_ms=timestamp_ms)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR source image.
    detections : List[Detection]
        Detection results for this frame.
    line_thickness : int
        Box line width in pixels.
    font_scale : float
        OpenCV font scale.

    Returns
    -------
    np.ndarray
        Annotated BGR image.
    """
    validate_frame(frame)
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        colour = CLASS_COLOURS.get(det.class_id, DEFAULT_COLOUR)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, line_thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(
            annotated, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_annotated_image(
    annotated: np.ndarray,
    output_path: str,
) -> None:
    """Write an annotated image to disk.

    Parameters
    ----------
    annotated : np.ndarray
        BGR annotated image.
    output_path : str
        Destination file path (creates parent directories as needed).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    success = cv2.imwrite(output_path, annotated)
    if not success:
        raise OSError(f"Failed to write annotated image to {output_path}")
    logger.info("Saved annotated image → %s", output_path)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def detections_to_json(detections: List[Detection]) -> List[Dict]:
    """Convert detection list to JSON-serialisable list of dicts."""
    return [d.to_dict() for d in detections]


def save_detections_json(
    detections: List[Detection],
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Save detection results as JSON.

    Parameters
    ----------
    detections : List[Detection]
        Detection results.
    output_path : str
        Destination .json path.
    metadata : dict, optional
        Additional metadata to embed (e.g. source file, detector name).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "total_detections": len(detections),
        "detections": detections_to_json(detections),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Saved %d detections → %s", len(detections), output_path
    )


# ---------------------------------------------------------------------------
# Frame-level reliability descriptor  (Equations 3–5)
# ---------------------------------------------------------------------------


def compute_reliability(
    detections: List[Detection],
    low_conf_threshold: float = 0.4,
    alpha1: float = 0.4,
    alpha2: float = 0.3,
    alpha3: float = 0.3,
    min_detections: int = 1,
    class_history: Optional[Dict[int, List[int]]] = None,
) -> Dict:
    """Compute frame-level reliability descriptor r_t.

    Implements the reliability descriptor from Equations 3–5 of the paper:

      Detections: {(b_i, k_i, s_i)}_{i=1}^N          (Eq. 3)
      mu_s = mean(s_i), f_low = fraction(s_i < threshold)   (Eq. 4)
      r_t = alpha1 * mu_s + alpha2 * (1 - f_low) + alpha3 * phi  (Eq. 5)

    where phi is a class-consistency term in [0, 1].

    Parameters
    ----------
    detections : List[Detection]
        Frame detections.
    low_conf_threshold : float
        Confidence below which a detection is counted as low-confidence.
    alpha1, alpha2, alpha3 : float
        Weighting coefficients (must sum to 1; validated here).
    min_detections : int
        Minimum number of detections required for a valid estimate;
        below this, a neutral reliability of 0.5 is returned.
    class_history : dict, optional
        {track_id: [class_id, ...]} from recent frames; used for phi.
        If None, phi defaults to 0.5 (neutral).

    Returns
    -------
    dict
        {
          "reliability": float,          # r_t in [0, 1]
          "mean_confidence": float,      # mu_s
          "low_conf_fraction": float,    # f_low
          "class_consistency": float,    # phi
          "num_detections": int,
          "valid": bool                  # False → neutral fallback used
        }
    """
    # Validate coefficients
    coeff_sum = alpha1 + alpha2 + alpha3
    if abs(coeff_sum - 1.0) > 1e-6:
        logger.warning(
            "Reliability coefficients sum to %.4f, not 1.0; normalising.",
            coeff_sum,
        )
        alpha1 /= coeff_sum
        alpha2 /= coeff_sum
        alpha3 /= coeff_sum

    n = len(detections)
    if n < min_detections:
        return {
            "reliability": 0.5,
            "mean_confidence": 0.5,
            "low_conf_fraction": 0.5,
            "class_consistency": 0.5,
            "num_detections": n,
            "valid": False,
        }

    scores = np.array([d.confidence for d in detections], dtype=np.float32)
    mu_s = float(np.mean(scores))
    f_low = float(np.mean(scores < low_conf_threshold))

    # Class consistency phi: fraction of detections whose class matches the
    # modal class seen in recent class_history.
    if class_history and n > 0:
        modal_class_counts: Dict[int, int] = {}
        for hist in class_history.values():
            if hist:
                modal = max(set(hist), key=hist.count)
                modal_class_counts[modal] = modal_class_counts.get(modal, 0) + 1
        if modal_class_counts:
            dominant = max(modal_class_counts, key=modal_class_counts.get)
            phi = float(
                sum(1 for d in detections if d.class_id == dominant) / n
            )
        else:
            phi = 0.5
    else:
        phi = 0.5  # neutral when no history

    reliability = float(
        np.clip(alpha1 * mu_s + alpha2 * (1.0 - f_low) + alpha3 * phi, 0.0, 1.0)
    )

    return {
        "reliability": reliability,
        "mean_confidence": mu_s,
        "low_conf_fraction": f_low,
        "class_consistency": phi,
        "num_detections": n,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Non-Maximum Suppression (NMS)
# ---------------------------------------------------------------------------


def compute_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
) -> float:
    """Calculate Intersection-over-Union (IoU) of two boxes in (x1, y1, x2, y2) format."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.45,
    conf_threshold: Optional[float] = None,
    per_class: bool = True,
) -> List[Detection]:
    """Apply Non-Maximum Suppression (NMS) to a list of detections.

    Parameters
    ----------
    detections : List[Detection]
        Input detection candidates.
    iou_threshold : float
        IoU threshold above which overlapping lower-confidence boxes are suppressed.
        Must be in [0.0, 1.0].
    conf_threshold : float, optional
        Minimum confidence required to retain a detection.
    per_class : bool
        If True (default), NMS is applied independently per class category.
        If False, class-agnostic NMS is performed.

    Returns
    -------
    List[Detection]
        Filtered detections sorted by descending confidence.

    Raises
    ------
    ValueError
        If iou_threshold or conf_threshold is outside [0.0, 1.0], or if bbox has x2 < x1 or y2 < y1.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0.0, 1.0]; got {iou_threshold}")

    if not detections:
        return []

    filtered = detections
    if conf_threshold is not None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in [0.0, 1.0]; got {conf_threshold}")
        filtered = [d for d in detections if d.confidence >= conf_threshold]

    if not filtered:
        return []

    for d in filtered:
        x1, y1, x2, y2 = d.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bounding box coordinates (x2 < x1 or y2 < y1): {d.bbox}")

    if per_class:
        class_buckets: Dict[int, List[Detection]] = {}
        for d in filtered:
            class_buckets.setdefault(d.class_id, []).append(d)
        groups = list(class_buckets.values())
    else:
        groups = [filtered]

    keep: List[Detection] = []
    for group in groups:
        sorted_group = sorted(group, key=lambda d: d.confidence, reverse=True)
        active: List[Detection] = []
        for candidate in sorted_group:
            suppress = False
            for selected in active:
                if compute_iou(candidate.bbox, selected.bbox) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                active.append(candidate)
        keep.extend(active)

    keep.sort(key=lambda d: d.confidence, reverse=True)
    return keep

