"""
evaluate.py - evaluate the configured detector on a COCO-format dataset split.

Examples
--------
  # 1) create a reproducible split once
  python evaluate.py --annotations path/to/instances.json --make-splits splits.json

  # 2) evaluate on the validation or test split
  python evaluate.py --images-dir path/to/images --annotations path/to/instances.json \
                     --split-file splits.json --split test

  # optional: per-condition breakdown from a CSV with columns  file_name,condition
  python evaluate.py ... --condition-csv conditions.csv

IMPORTANT: with `backend: "stub"` in config.yaml the detector outputs synthetic
random boxes.  This script REFUSES to run in that case (its numbers would be
meaningless) unless you pass --allow-stub, in which case every output is
clearly stamped as meaningless.  Real numbers need real F-YOLO weights.

No training is performed here: this project contains no trainer.  The splits
are for evaluation (use 'val' to tune thresholds/lambda, 'test' once at the end).
"""
import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict

import cv2

from data_tools.coco_dataset import load_coco_vehicles, make_splits, save_splits, load_splits
from data_tools.evaluation import evaluate_detections
from detector.fyolo_detector import create_detector
from detector.vehicle_detection import detect_vehicles
from utils.config_loader import load_config

logger = logging.getLogger("ratfa.evaluate")


def build_parser():
    p = argparse.ArgumentParser(description="RATFA dataset evaluation")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--annotations", required=True, help="COCO-format annotation JSON")
    p.add_argument("--images-dir", help="Folder containing the images")
    p.add_argument("--make-splits", metavar="OUT_JSON", help="Create train/val/test split file and exit")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-file", help="Split JSON created with --make-splits")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--condition-csv", help="CSV (file_name,condition) for per-condition metrics")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--output", default="outputs/evaluation")
    p.add_argument("--allow-stub", action="store_true",
                   help="Run even with the stub detector (results are meaningless)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    gt = load_coco_vehicles(args.annotations)

    if args.make_splits:
        splits = make_splits(list(gt.keys()), tuple(args.ratios), args.seed)
        save_splits(splits, args.make_splits, args.seed)
        print({k: len(v) for k, v in splits.items()}, "->", args.make_splits)
        return 0

    if not args.images_dir:
        print("--images-dir is required for evaluation", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    detector = create_detector(cfg.get("detector", {}))
    is_stub = "stub" in str(getattr(detector, "backend_name", "")).lower()
    if is_stub and not args.allow_stub:
        print("REFUSING: the configured detector is the synthetic StubDetector, so metrics would be "
              "meaningless. Install real F-YOLO weights (see checkpoints/README.md) or pass --allow-stub "
              "to test the evaluation code only.", file=sys.stderr)
        return 3

    ids = sorted(gt.keys())
    if args.split_file:
        wanted = set(load_splits(args.split_file)[args.split])
        ids = [i for i in ids if i in wanted]
    if args.max_images:
        ids = ids[: args.max_images]

    cond = {}
    if args.condition_csv:
        with open(args.condition_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cond[row["file_name"]] = row["condition"]

    preds, gts, by_cond = {}, {}, defaultdict(list)
    for n, iid in enumerate(ids):
        g = gt[iid]
        frame = cv2.imread(os.path.join(args.images_dir, g.file_name))
        if frame is None:
            logger.warning("Could not read %s; skipped.", g.file_name)
            continue
        dets = detect_vehicles(detector, frame, frame_number=n, timestamp_ms=0.0)
        preds[iid] = [(d.bbox, d.class_id, d.confidence) for d in dets]
        gts[iid] = g.boxes
        by_cond[cond.get(g.file_name, "unlabelled")].append(iid)

    result = {"split": args.split if args.split_file else "all", "num_images": len(gts),
              "detector": getattr(detector, "backend_name", "?"),
              "overall": evaluate_detections(preds, gts, args.iou)}
    if args.condition_csv:
        result["by_condition"] = {
            c: evaluate_detections({i: preds[i] for i in v}, {i: gts[i] for i in v}, args.iou)
            for c, v in by_cond.items()
        }
    if is_stub:
        result["WARNING"] = "STUB DETECTOR: these numbers are MEANINGLESS (synthetic random boxes)."
    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, f"evaluation_{result['split']}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    o = result["overall"]
    print(f"images={result['num_images']}  mAP@{args.iou}={o['mAP']:.4f}  P={o['precision']:.4f}  "
          f"R={o['recall']:.4f}  F1={o['f1']:.4f}  -> {out}")
    if is_stub:
        print(result["WARNING"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
