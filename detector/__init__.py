"""
RATFA – Detector Package
========================
Provides the detector interface and vehicle-detection utilities.

Integration status
------------------
F-YOLO (Jia et al., 2024):
  BLOCKED – The original F-YOLO source repository and pretrained checkpoint
  are not publicly available at the time of this implementation.

YOLOv8-based Detector (Production Ready):
  ✅ Available and functional using Ultralytics YOLOv8
  ✅ Fuzzy Feature Fusion module implemented for RATFA integration
  ✅ Supports real traffic video processing
  ✅ Compatible with ByteTrack for temporal tracking

Set config.detector.backend = "yolo" to use the production YOLOv8 detector.
"""
from detector.fyolo_detector import BaseDetector, Detection, FYOLODetector, StubDetector
from detector.yolo_detector import FuzzyYOLOv8Detector, FuzzyFeatureFusion, create_detector

__all__ = ['BaseDetector', 'Detection', 'FYOLODetector', 'StubDetector',
           'FuzzyYOLOv8Detector', 'FuzzyFeatureFusion', 'create_detector']