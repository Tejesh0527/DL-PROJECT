# Datasets used in the RATFA experiments

## 1. UA-DETRAC (`UA-DETRAC/`)
- Source: UA-DETRAC vehicle detection and tracking benchmark (Wen et al., 2015), public mirror
  https://huggingface.co/datasets/abhineet123/ua_detrac (licence CC-BY-4.0 mirror; original CC-BY-NC-SA).
- Content: 960x540 traffic-camera frames; `annotations.csv` per sequence with
  `filename, width, height, xmin, ymin, xmax, ymax, target_id` (target_id = vehicle track ID, -1 = ignore region).
- `train/` 8 sequences, every 15th frame (465 frames):
  - training: MVI_20011, MVI_39761, MVI_40131, MVI_40191, MVI_40732, MVI_40962 (345 frames)
  - validation: MVI_63521, MVI_20051 (120 frames)
- `test/` 4 sequences, first 150 consecutive frames each (600 frames): MVI_39031, MVI_40701, MVI_40852, MVI_40903.
- Used for: detector training/validation, scene-classifier training (train frames only), per-condition fuzzy
  calibration (train frames only), detection and tracking tests (test frames).

## 2. BDD100K validation subset (`BDD100K/`)
- Source: BDD100K (Yu et al., 2020), public mirror https://huggingface.co/datasets/dgural/bdd100k (BSD-3 licence).
- `bdd100k_ratfa_subset.json`: per image `file, condition, weather, timeofday, width, height, boxes`
  (vehicle boxes = car, bus, truck in pixel xyxy).
- 373 images: clear = clear + daytime (120), rain = rainy + daytime (120), dark = clear + night (120),
  fog = foggy (13, which is all foggy images in BDD100K val). Random selection with seed 42.
- Used for: scene-classifier training (70 %) and testing (30 %, stratified, seed 42); cross-domain detection test.

## 3. Synthetic adverse conditions
UA-DETRAC has no per-frame weather labels, so rain, fog and low-light versions are generated with the
deterministic corruptions in `1_Python_Code/experiments/corruptions.py`:
- calibration / training: severity 0.8, seeds 1000+ / 2000+ / 5000+
- test: severity 1.0, seed = sequence_index * 1000 + frame_index

## 4. COCO
Not downloaded. It is used only through the COCO-pretrained YOLOv8n weights
(`1_Python_Code/checkpoints/yolov8n_coco_pretrained.pt`).

## How to reproduce the results
1. Copy this folder's `UA-DETRAC/` and `BDD100K/` into `1_Python_Code/datasets/`
   (or run `python experiments/download_datasets.py` to download the same subsets again).
2. In `1_Python_Code/`:
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python experiments/train_detector.py --epochs 5     # baseline + F-YOLO training
python experiments/train_scene_classifier.py        # scene classifier training + test
python experiments/calibrate_conditions.py          # per-condition fuzzy targets
python experiments/run_experiments.py               # all tests
python experiments/make_figures.py                  # tables + figures
```
Settings: image size 512, batch 16, 5 epochs, seed 42, CPU (Apple M1), Python 3.9, Ultralytics 8.4.
