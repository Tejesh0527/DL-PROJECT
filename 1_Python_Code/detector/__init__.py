"""
RATFA – Detector Package
========================
Provides the detector interface and vehicle-detection utilities.

Integration status
------------------
F-YOLO (Jia et al., 2024):
  BLOCKED – The original F-YOLO source repository and pretrained checkpoint
  are not publicly available at the time of this implementation.
  Specifically, the following could not be obtained:
    • Source code for FFM (Fuzzy Fusion Module)
    • Source code for FA (Fuzzy Attention)
    • Source code for GPBlock (GhostBlock pooling)
    • Pretrained checkpoint (fyolo_coco.pt or equivalent)
  A clearly-labelled temporary stub is used for software-testing only.
  Set config.detector.backend = "fyolo" once the checkpoint is available.
"""

