"""
detector/fyolo_detector.py
==========================
F-YOLO Detector Interface for RATFA.

Integration Status
------------------
F-YOLO (Jia et al., "F-YOLO: A Fuzzy Feature Fusion Approach for YOLO-Based
Traffic Object Detection", 2024) is the reference detector specified by the
RATFA research paper.

BLOCKED: The F-YOLO source repository and pretrained checkpoint are not
publicly available. This module defines the interface that *will* be used once
the dependency becomes available, plus a clearly-labelled temporary stub for
software-testing purposes.

Architecture of F-YOLO (from paper description):
  - Backbone: YOLOv6 base
  - Fuzzy Fusion Module (FFM): converts feature values into fuzzy membership
    degrees, fuses multi-scale features with fuzzy aggregation.
  - Fuzzy Attention (FA): applies fuzzy weights to spatial/channel features.
  - GhostBlock pooling (GPBlock): global context enhancement via efficient
    pooling with Ghost convolutions.
  Trainable Gaussian membership parameters (centres c and widths σ) are the
  targets of the RATFA adaptation controller.

Do NOT substitute YOLOv5, YOLOv8, or YOLOv11 and label the result F-YOLO.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """Single object detection result."""

    frame_number: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    class_id: int
    class_name: str
    confidence: float
    timestamp_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "frame_number": self.frame_number,
            "bbox": {
                "x1": float(self.bbox[0]),
                "y1": float(self.bbox[1]),
                "x2": float(self.bbox[2]),
                "y2": float(self.bbox[3]),
            },
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "timestamp_ms": float(self.timestamp_ms),
        }


# ---------------------------------------------------------------------------
# Abstract base interface
# ---------------------------------------------------------------------------


class BaseDetector(ABC):
    """Abstract detector interface.

    All concrete detectors (F-YOLO, stub, etc.) must implement this interface
    so that the rest of the RATFA pipeline is detector-agnostic.
    """

    # Vehicle class IDs (COCO): car=2, motorcycle=3, bus=5, truck=7
    VEHICLE_CLASS_IDS: List[int] = [2, 3, 5, 7]
    VEHICLE_CLASS_NAMES: Dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError(f"conf_threshold must be in [0, 1]; got {conf_threshold}")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1]; got {iou_threshold}")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights.  Raises RuntimeError if loading fails."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> List[Detection]:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H × W × 3).
        frame_number : int
            Frame index (0-based) in the video sequence.
        timestamp_ms : float
            Frame timestamp in milliseconds.

        Returns
        -------
        List[Detection]
            Filtered vehicle detections.
        """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True when the model weights are loaded and inference is ready."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier for the backend (e.g. 'F-YOLO', 'stub')."""


# ---------------------------------------------------------------------------
# F-YOLO detector (integration BLOCKED)
# ---------------------------------------------------------------------------


class FYOLODetector(BaseDetector):
    """F-YOLO detector wrapper.

    Status: BLOCKED – pretrained checkpoint and source not available.
    This class prepares the integration interface.  When the checkpoint
    becomes available:
      1. Place the F-YOLO source under detector/fyolo_src/.
      2. Set checkpoint_path to the actual .pt file.
      3. Remove the NotImplementedError in load_model().
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        imgsz: int = 640,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        super().__init__(conf_threshold, iou_threshold)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return "F-YOLO (BLOCKED)"

    def load_model(self) -> None:
        """Attempt to load the F-YOLO checkpoint.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        RuntimeError
            If the F-YOLO source is not installed.
        """
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"[F-YOLO] Checkpoint not found: {self.checkpoint_path}\n"
                "BLOCKED: Obtain the F-YOLO pretrained weights and place them "
                "at the path specified in config/config.yaml under "
                "detector.fyolo.checkpoint."
            )

        # F-YOLO source integration point.
        # TODO: once fyolo_src is available, uncomment and adapt:
        # from detector.fyolo_src.models.fyolo import FYOLOModel
        # self._model = FYOLOModel(self.config_path)
        # self._model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        # self._model.eval()
        # self._loaded = True
        raise RuntimeError(
            "[F-YOLO] Integration BLOCKED.\n"
            "The F-YOLO source code (FFM, FA, GPBlock) is required but could "
            "not be located in the public domain at the time of implementation.\n"
            "Steps to unblock:\n"
            "  1. Obtain the original F-YOLO source from the paper authors.\n"
            "  2. Place it under detector/fyolo_src/.\n"
            "  3. Remove the RuntimeError in FYOLODetector.load_model().\n"
            "  4. Implement the inference call in FYOLODetector.detect()."
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> List[Detection]:
        if not self._loaded:
            raise RuntimeError("Call load_model() before detect().")
        # TODO: implement actual F-YOLO inference once source is available.
        raise NotImplementedError("F-YOLO inference not yet implemented (BLOCKED).")


# ---------------------------------------------------------------------------
# Temporary stub detector (for software testing ONLY)
# ---------------------------------------------------------------------------


class StubDetector(BaseDetector):
    """Clearly-labelled temporary stub detector.

    WARNING: This is NOT F-YOLO.  It generates synthetic bounding boxes
    to allow the rest of the RATFA pipeline to be tested without actual
    inference.  Results are deterministic but meaningless as detection outputs.

    Remove or replace with FYOLODetector once the actual checkpoint is
    available.
    """

    def __init__(
        self,
        imgsz: int = 640,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        seed: int = 42,
    ):
        super().__init__(conf_threshold, iou_threshold)
        self.imgsz = imgsz
        self.device = device
        self._rng = np.random.default_rng(seed)
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend_name(self) -> str:
        return "StubDetector (TEST-ONLY – NOT F-YOLO)"

    def load_model(self) -> None:
        logger.warning(
            "StubDetector loaded.  This is a TEST-ONLY placeholder "
            "and is NOT the actual F-YOLO model."
        )
        self._loaded = True

    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp_ms: float = 0.0,
    ) -> List[Detection]:
        """Return 0–3 synthetic vehicle detections.

        Boxes are deterministic relative to the frame number so that
        temporal-consistency tests produce meaningful (if synthetic) IoU values.
        """
        if not self._loaded:
            raise RuntimeError("Call load_model() before detect().")
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a 3-D NumPy array (H, W, C).")

        h, w = frame.shape[:2]
        rng = np.random.default_rng(frame_number)   # deterministic per frame

        num_detections = rng.integers(0, 4)  # 0–3 vehicles
        detections: List[Detection] = []

        for i in range(num_detections):
            # Generate a plausible bounding box
            x1 = float(rng.uniform(0, w * 0.7))
            y1 = float(rng.uniform(0, h * 0.7))
            bw = float(rng.uniform(w * 0.1, w * 0.3))
            bh = float(rng.uniform(h * 0.1, h * 0.3))
            x2 = min(x1 + bw, float(w))
            y2 = min(y1 + bh, float(h))
            conf = float(rng.uniform(self.conf_threshold, 1.0))
            class_id = int(rng.choice(self.VEHICLE_CLASS_IDS))
            detections.append(
                Detection(
                    frame_number=frame_number,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=self.VEHICLE_CLASS_NAMES[class_id],
                    confidence=conf,
                    timestamp_ms=timestamp_ms,
                )
            )

        logger.debug(
            "StubDetector: frame %d → %d synthetic detections",
            frame_number,
            len(detections),
        )
        return detections


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_detector(config: dict) -> BaseDetector:
    """Factory: create and return the appropriate detector.

    Parameters
    ----------
    config : dict
        Contents of config.yaml['detector'].

    Returns
    -------
    BaseDetector
        Loaded detector instance.
    """
    backend = config.get("backend", "stub").lower()
    conf_th = config.get("conf_threshold", 0.25)
    iou_th = config.get("iou_threshold", 0.45)

    if backend == "fyolo":
        fyolo_cfg = config.get("fyolo", {})
        detector = FYOLODetector(
            checkpoint_path=fyolo_cfg.get("checkpoint", ""),
            config_path=fyolo_cfg.get("config", ""),
            imgsz=fyolo_cfg.get("imgsz", 640),
            device=fyolo_cfg.get("device", "cpu"),
            conf_threshold=conf_th,
            iou_threshold=iou_th,
        )
    elif backend == "stub":
        stub_cfg = config.get("stub", {})
        detector = StubDetector(
            imgsz=stub_cfg.get("imgsz", 640),
            device=stub_cfg.get("device", "cpu"),
            conf_threshold=conf_th,
            iou_threshold=iou_th,
        )
    else:
        raise ValueError(f"Unknown detector backend: '{backend}'")

    detector.load_model()
    logger.info("Detector ready: %s", detector.backend_name)
    return detector

