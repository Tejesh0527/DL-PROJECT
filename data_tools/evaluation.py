"""
data_tools/evaluation.py
========================
Standard detection metrics at a fixed IoU threshold (default 0.5), numpy only:
per-class AP (all-point interpolation), mAP, and precision / recall / F1 at the
detector's operating confidence threshold.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from detector.vehicle_detection import compute_iou

Box = Tuple[float, float, float, float]


def _match(
    preds: List[Tuple[int, Box, float]],      # (image_idx, box, conf) sorted by conf desc
    gts: Dict[int, List[Box]],                # image_idx -> boxes of this class
    iou_thr: float,
) -> List[int]:
    """Return a 1/0 true-positive flag per prediction (greedy by confidence)."""
    used = {k: [False] * len(v) for k, v in gts.items()}
    flags = []
    for img, box, _ in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts.get(img, [])):
            if used[img][j]:
                continue
            iou = compute_iou(box, g)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_thr and best_j >= 0:
            used[img][best_j] = True
            flags.append(1)
        else:
            flags.append(0)
    return flags


def average_precision(tp: Sequence[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    tp = np.array(tp, dtype=np.float64)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detections(
    predictions: Dict[int, List[Tuple[Box, int, float]]],   # image_idx -> [(box, class_id, conf)]
    ground_truth: Dict[int, List[Tuple[float, float, float, float, int]]],
    iou_thr: float = 0.5,
) -> Dict:
    """Compute per-class AP, mAP and overall precision/recall/F1."""
    classes = sorted({g[4] for gl in ground_truth.values() for g in gl}
                     | {p[1] for pl in predictions.values() for p in pl})
    per_class = {}
    tot_tp = tot_fp = tot_gt = 0
    for c in classes:
        gts = {i: [g[:4] for g in gl if g[4] == c] for i, gl in ground_truth.items()}
        n_gt = sum(len(v) for v in gts.values())
        preds = sorted(
            [(i, p[0], p[2]) for i, pl in predictions.items() for p in pl if p[1] == c],
            key=lambda t: -t[2],
        )
        flags = _match(preds, gts, iou_thr)
        tp = int(sum(flags)); fp = len(flags) - tp
        per_class[c] = {
            "ap": average_precision(flags, n_gt), "num_gt": n_gt,
            "num_pred": len(flags), "tp": tp, "fp": fp, "fn": n_gt - tp,
        }
        tot_tp += tp; tot_fp += fp; tot_gt += n_gt
    aps = [v["ap"] for v in per_class.values() if not np.isnan(v["ap"])]
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    recall = tot_tp / tot_gt if tot_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "iou_threshold": iou_thr,
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "precision": precision, "recall": recall, "f1": f1,
        "per_class": per_class,
    }
