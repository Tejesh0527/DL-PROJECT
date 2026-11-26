# RATFA Checkpoints Directory

This directory is designated for pretrained neural network weights.

## Expected Reference Model: F-YOLO
- **Expected Checkpoint**: `checkpoints/fyolo_coco.pt`
- **Config Path**: `config/fyolo.yaml`
- **Reference**: Jia et al., "F-YOLO: A Fuzzy Feature Fusion Approach for YOLO-Based Traffic Object Detection", IEEE T-FUZZ, 2024.

## Current Status: BLOCKED
Official weights have not been released publicly by the authors.  
To unblock genuine F-YOLO execution:
1. Place the official `fyolo_coco.pt` weight file in this directory (`checkpoints/fyolo_coco.pt`).
2. Place the F-YOLO source repository under `detector/fyolo_src/`.
3. In `config/config.yaml`, set:
   ```yaml
   detector:
     backend: "fyolo"
   ```
4. Run `python main.py --mode image --input data/input_images/clear_road.jpg`.

