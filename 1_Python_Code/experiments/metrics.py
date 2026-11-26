"""Detection, calibration and tracking metrics used in the experiments."""
from typing import Dict, List

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_frame(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, thr: float = 0.5):
    """Greedy (score-ordered) matching; returns a TP flag per prediction."""
    order = np.argsort(-scores)
    tp = np.zeros(len(pred), bool)
    if len(gt) == 0 or len(pred) == 0:
        return tp
    ious = iou_matrix(pred[order], gt)
    used = np.zeros(len(gt), bool)
    for k, i in enumerate(order):
        j = int(np.argmax(np.where(used, -1, ious[k])))
        if ious[k, j] >= thr and not used[j]:
            used[j] = True
            tp[i] = True
    return tp


def detection_metrics(frames: List[Dict], conf_thr: float = 0.25, n_bins: int = 10) -> Dict:
    """frames: list of {pred: Nx4, scores: N, gt: Mx4}. Returns AP50, P, R, F1, ECE."""
    all_s, all_tp, n_gt = [], [], 0
    for f in frames:
        tp = match_frame(f["pred"], f["scores"], f["gt"])
        all_s.append(f["scores"]); all_tp.append(tp); n_gt += len(f["gt"])
    s = np.concatenate(all_s) if all_s else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    order = np.argsort(-s)
    s, tp = s[order], tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = float(np.trapz(np.interp(x, mrec, mpre), x))
    keep = s >= conf_thr
    P = float(tp[keep].mean()) if keep.any() else 0.0
    R = float(tp[keep].sum() / max(n_gt, 1))
    F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
    # Expected calibration error of the confidence scores (TP = correct)
    ece, bins = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s > lo) & (s <= hi)
        if m.any():
            ece += m.mean() * abs(tp[m].mean() - s[m].mean())
    return {"AP50": ap, "precision": P, "recall": R, "F1": F1, "ECE": float(ece), "n_gt": n_gt}


def tracking_metrics(sequences: List[Dict]) -> Dict:
    """sequences: list of {frames: [ {gt_ids, gt_boxes, trk_ids, trk_boxes} ]}. MOTA / IDF1."""
    import motmetrics as mm
    accs, names = [], []
    for k, seq in enumerate(sequences):
        acc = mm.MOTAccumulator(auto_id=True)
        for f in seq["frames"]:
            gt_b, tr_b = np.asarray(f["gt_boxes"]), np.asarray(f["trk_boxes"])
            d = 1 - iou_matrix(gt_b, tr_b) if len(gt_b) and len(tr_b) else np.empty((len(gt_b), len(tr_b)))
            if d.size:
                d[d > 0.5] = np.nan
            acc.update(list(f["gt_ids"]), list(f["trk_ids"]), d)
        accs.append(acc); names.append(str(k))
    mh = mm.metrics.create()
    summ = mh.compute_many(accs, names=names, metrics=["mota", "idf1", "num_switches", "motp"],
                           generate_overall=True)
    o = summ.loc["OVERALL"]
    return {"MOTA": float(o["mota"]), "IDF1": float(o["idf1"]), "ID_switches": int(o["num_switches"]),
            "MOTP_dist": float(o["motp"])}
