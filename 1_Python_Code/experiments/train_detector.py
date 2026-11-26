"""Train (fine-tune) the detectors on the UA-DETRAC training subset.

Two models are trained with identical settings so that they can be compared:
  * baseline : COCO-pretrained YOLOv8n, fine-tuned for the single class "vehicle"
  * fyolo    : same network + F-YOLO fuzzy modules (GPBlock, FFM, FA)

Train / val split is by sequence (no frame of a val sequence is seen in training).
Usage:  python experiments/train_detector.py [--epochs 8] [--imgsz 512]
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".ultralytics"))

from detector.fyolo_model import make_fuzzy_trainer  # noqa: E402

DETRAC = os.path.join(ROOT, "datasets", "UA-DETRAC")
YOLO_DIR = os.path.join(ROOT, "datasets", "UA-DETRAC-yolo")
VAL_SEQS = {"MVI_63521", "MVI_20051"}
PRETRAINED = os.path.join(ROOT, "checkpoints", "yolov8n_coco_pretrained.pt")


def build_yolo_dataset():
    shutil.rmtree(YOLO_DIR, ignore_errors=True)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(YOLO_DIR, sub, split), exist_ok=True)
    counts = {"train": 0, "val": 0}
    for seq_dir in sorted(glob.glob(os.path.join(DETRAC, "train", "*"))):
        seq = os.path.basename(seq_dir)
        split = "val" if seq in VAL_SEQS else "train"
        boxes = {}
        for r in csv.DictReader(open(os.path.join(seq_dir, "annotations.csv"))):
            boxes.setdefault(r["filename"], []).append(r)
        for img in sorted(glob.glob(os.path.join(seq_dir, "*.jpg"))):
            fn = os.path.basename(img)
            stem = f"{seq}_{fn[:-4]}"
            # UA-DETRAC ignore regions (target_id == -1) are masked out with grey
            im = cv2.imread(img)
            for r in boxes.get(fn, []):
                if int(r["target_id"]) < 0:
                    x1, y1, x2, y2 = (int(float(r[k])) for k in ("xmin", "ymin", "xmax", "ymax"))
                    im[max(0, y1):y2, max(0, x1):x2] = 114
            cv2.imwrite(os.path.join(YOLO_DIR, "images", split, stem + ".jpg"), im)
            with open(os.path.join(YOLO_DIR, "labels", split, stem + ".txt"), "w") as f:
                for r in (r for r in boxes.get(fn, []) if int(r["target_id"]) >= 0):
                    W, H = float(r["width"]), float(r["height"])
                    x1, y1, x2, y2 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                    f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            counts[split] += 1
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n")
    print("YOLO dataset:", counts)
    return yaml_path


def train(kind, data_yaml, epochs, imgsz, batch):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    args = dict(model=PRETRAINED, data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                device="cpu", workers=0, seed=42, deterministic=True, plots=True,
                project=os.path.join(ROOT, "runs"), name=f"train_{kind}", exist_ok=True,
                amp=False, verbose=False, mosaic=1.0, close_mosaic=2, patience=100)
    if kind == "fyolo":
        trainer_cls = make_fuzzy_trainer(os.path.join(YOLO_DIR, "images", "train", "*.jpg"),
                                         imgsz=imgsz)
    else:
        trainer_cls = DetectionTrainer
    trainer = trainer_cls(overrides=args)
    trainer.train()
    best = os.path.join(ROOT, "runs", f"train_{kind}", "weights", "best.pt")
    dst = os.path.join(ROOT, "checkpoints", f"{kind}_detrac.pt")
    shutil.copy(best, dst)
    # validation metrics of the best model
    m = YOLO(dst).val(data=data_yaml, imgsz=imgsz, batch=batch, device="cpu", workers=0,
                      verbose=False, plots=False, project=os.path.join(ROOT, "runs"),
                      name=f"val_{kind}", exist_ok=True)
    res = {"model": kind, "epochs": epochs, "imgsz": imgsz,
           "val_mAP50": float(m.box.map50), "val_mAP50_95": float(m.box.map),
           "val_precision": float(m.box.mp), "val_recall": float(m.box.mr)}
    os.makedirs(os.path.join(ROOT, "outputs", "training"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "outputs", "training", f"{kind}_val_metrics.json"), "w"),
              indent=2)
    csv_src = os.path.join(ROOT, "runs", f"train_{kind}", "results.csv")
    if os.path.exists(csv_src):
        shutil.copy(csv_src, os.path.join(ROOT, "outputs", "training", f"{kind}_training_curve.csv"))
    print(res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=["baseline", "fyolo"])
    a = ap.parse_args()
    data_yaml = build_yolo_dataset()
    for k in a.models:
        train(k, data_yaml, a.epochs, a.imgsz, a.batch)
