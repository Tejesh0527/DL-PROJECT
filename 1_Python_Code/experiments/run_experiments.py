"""Testing / evaluation of YOLO baseline, F-YOLO and F-YOLO + RATFA.

Experiment 1 - UA-DETRAC test sequences (video, 4 sequences x 150 frames) under
               clear / rain / fog / low-light conditions:
               detection (AP50, P, R, F1), calibration (ECE),
               tracking with ByteTrack (MOTA, IDF1, ID switches), runtime (FPS, latency).
               Ablation: RATFA without confidence gate (gamma = 1).
Experiment 2 - BDD100K real images (cross-domain, real weather labels).

Usage:  python experiments/run_experiments.py
"""
import csv
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".ultralytics"))

from detector.fyolo_detector import Detection  # noqa: E402
from detector.vehicle_detection import compute_reliability  # noqa: E402
from experiments.corruptions import CONDITIONS, apply  # noqa: E402
from experiments.metrics import detection_metrics, tracking_metrics  # noqa: E402
from fuzzy.ratfa_controller import RATFAController  # noqa: E402
from scene_analysis.condition_smoother import ConditionSmoother  # noqa: E402
from scene_analysis.scene_classifier import SceneClassifier  # noqa: E402
from tracking.temporal_consistency import TemporalConsistencyTracker  # noqa: E402

IMGSZ, CONF, IOU, OP_CONF = 512, 0.05, 0.6, 0.25
OUT = os.path.join(ROOT, "outputs")
TEST_DIR = os.path.join(ROOT, "datasets", "UA-DETRAC", "test")
METHODS = ["YOLOv8n baseline", "F-YOLO (static)", "F-YOLO + RATFA", "RATFA w/o gate"]


def load_models():
    from ultralytics import YOLO
    base = YOLO(os.path.join(ROOT, "checkpoints", "baseline_detrac.pt"))
    fy = YOLO(os.path.join(ROOT, "checkpoints", "fyolo_detrac.pt"))
    dummy = np.zeros((540, 960, 3), np.uint8)
    for m in (base, fy):
        m.predict(dummy, imgsz=IMGSZ, conf=CONF, verbose=False, device="cpu")
    return base, fy


def make_tracker():
    from ultralytics.trackers.byte_tracker import BYTETracker
    args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
                           new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)
    return BYTETracker(args)


def load_sequence(seq_dir):
    gt = {}
    for r in csv.DictReader(open(os.path.join(seq_dir, "annotations.csv"))):
        gt.setdefault(r["filename"], []).append(
            [float(r["xmin"]), float(r["ymin"]), float(r["xmax"]), float(r["ymax"]), int(r["target_id"])])
    frames = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    return frames, gt


def in_ignore(boxes, ign):
    """True for boxes whose centre lies inside a UA-DETRAC ignore region."""
    if len(boxes) == 0 or len(ign) == 0:
        return np.zeros(len(boxes), bool)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    return ((cx[:, None] >= ign[None, :, 0]) & (cx[:, None] <= ign[None, :, 2]) &
            (cy[:, None] >= ign[None, :, 1]) & (cy[:, None] <= ign[None, :, 3])).any(1)


def predict(model, img):
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False, device="cpu")[0]
    return r.boxes.cpu().numpy()


def run_sequence(method, models, ctrl_cfg, scene_clf, frames, gt, condition, seq_seed, video=None):
    base, fy = models
    model = base if method == "YOLOv8n baseline" else fy
    ctrl = None
    if method in ("F-YOLO + RATFA", "RATFA w/o gate"):
        ctrl = RATFAController(fy.predictor.model.model, ctrl_cfg, use_gate=(method == "F-YOLO + RATFA"))
    smoother, tc = ConditionSmoother(lambda_e=0.7), TemporalConsistencyTracker()
    tracker = make_tracker()
    det_frames, trk_frames, lat = [], [], []
    for t, path in enumerate(frames):
        img = apply(cv2.imread(path), condition, seed=seq_seed * 1000 + t, severity=1.0)
        t0 = time.perf_counter()
        if ctrl is not None:
            e_t, q_t = scene_clf.predict_proba_img(img)
            e_bar = smoother.update(e_t)
        boxes = predict(model, img)
        xyxy, conf = boxes.xyxy, boxes.conf
        ops = [Detection(t, tuple(b), 0, "vehicle", float(c)) for b, c in zip(xyxy, conf) if c >= OP_CONF]
        tc_out = tc.update(ops)
        if ctrl is not None:
            rel = compute_reliability(ops, class_history=tc_out.class_history)
            ctrl.step(e_bar, q_t, rel["reliability"], tc_out.tau, frame=t)   # affects frame t+1
        tracks = tracker.update(boxes, img)
        lat.append(time.perf_counter() - t0)
        g_all = np.array(gt.get(os.path.basename(path), []), dtype=np.float32).reshape(-1, 5)
        g, ign = g_all[g_all[:, 4] >= 0], g_all[g_all[:, 4] < 0, :4]
        keep = ~in_ignore(xyxy, ign)                   # UA-DETRAC protocol: drop ignored-region boxes
        tracks = tracks.reshape(-1, tracks.shape[-1] if tracks.ndim == 2 else 8)
        tkeep = ~in_ignore(tracks[:, :4], ign)
        det_frames.append({"pred": xyxy[keep], "scores": conf[keep], "gt": g[:, :4]})
        trk_frames.append({"gt_ids": g[:, 4].astype(int).tolist(), "gt_boxes": g[:, :4],
                           "trk_ids": tracks[tkeep, 4].astype(int).tolist(),
                           "trk_boxes": tracks[tkeep, :4]})
        if video is not None:
            vis = img.copy()
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr[:4])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID {int(tr[4])} {tr[5]:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            if ctrl is not None:
                lg = ctrl.log[-1]
                txt = (f"{method} | scene={CONDITIONS[int(np.argmax(e_bar))]} q={lg['q']:.2f} "
                       f"r={lg['r']:.2f} tau={lg['tau']:.2f} gamma={lg['gamma']:.2f}")
            else:
                txt = method
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(vis, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            video.write(vis)
    log = list(ctrl.log) if ctrl is not None else None
    if ctrl is not None:
        ctrl.reset()
    return det_frames, {"frames": trk_frames}, lat, log


def experiment_detrac(models, ctrl_cfg, scene_clf, conditions=CONDITIONS):
    seqs = sorted(glob.glob(os.path.join(TEST_DIR, "*")))
    results, gate_logs = [], {}
    os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)
    for condition in conditions:
        for method in METHODS:
            all_det, all_trk, all_lat = [], [], []
            for si, sd in enumerate(seqs):
                frames, gt = load_sequence(sd)
                video = None
                if method == "F-YOLO + RATFA" and si == 1:
                    h, w = cv2.imread(frames[0]).shape[:2]
                    video = cv2.VideoWriter(os.path.join(OUT, "videos", f"ratfa_{condition}_{os.path.basename(sd)}.mp4"),
                                            cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                d, tr, lat, log = run_sequence(method, models, ctrl_cfg, scene_clf, frames, gt,
                                               condition, si, video)
                if video is not None:
                    video.release()
                all_det += d; all_trk.append(tr); all_lat += lat
                if log is not None:
                    gate_logs[f"{method}|{condition}|{os.path.basename(sd)}"] = log
            row = {"dataset": "UA-DETRAC", "condition": condition, "method": method}
            row.update(detection_metrics(all_det)); row.update(tracking_metrics(all_trk))
            row["latency_ms"] = 1000 * float(np.mean(all_lat)); row["FPS"] = 1.0 / float(np.mean(all_lat))
            results.append(row)
            print(f"{condition:6s} {method:18s} AP50={row['AP50']:.3f} F1={row['F1']:.3f} "
                  f"MOTA={row['MOTA']:.3f} IDF1={row['IDF1']:.3f} ECE={row['ECE']:.3f} FPS={row['FPS']:.1f}",
                  flush=True)
    return results, gate_logs


def experiment_bdd(models, ctrl_cfg, scene_clf):
    """Single images: no temporal evidence, so tau = 1 and one full gated step (eta = 1)."""
    base, fy = models
    bdd = os.path.join(ROOT, "datasets", "BDD100K")
    recs = json.load(open(os.path.join(bdd, "bdd100k_ratfa_subset.json")))
    per = {(c, m): [] for c in CONDITIONS for m in METHODS[:3]}
    ctrl = RATFAController(fy.predictor.model.model, ctrl_cfg, eta=1.0)
    for r in recs:
        img = cv2.imread(os.path.join(bdd, "images", r["file"]))
        gt = np.array(r["boxes"], np.float32).reshape(-1, 4)
        for m, model in (("YOLOv8n baseline", base), ("F-YOLO (static)", fy)):
            b = predict(model, img)
            per[(r["condition"], m)].append({"pred": b.xyxy, "scores": b.conf, "gt": gt})
        e, q = scene_clf.predict_proba_img(img)
        b = predict(fy, img)
        ops = [Detection(0, tuple(x), 0, "vehicle", float(c)) for x, c in zip(b.xyxy, b.conf) if c >= OP_CONF]
        ctrl.step(e, q, compute_reliability(ops)["reliability"], 1.0)
        b = predict(fy, img)
        ctrl.reset()
        per[(r["condition"], "F-YOLO + RATFA")].append({"pred": b.xyxy, "scores": b.conf, "gt": gt})
    rows = []
    for (c, m), frames in per.items():
        row = {"dataset": "BDD100K", "condition": c, "method": m, "n_images": len(frames)}
        row.update(detection_metrics(frames))
        rows.append(row)
        print(f"BDD {c:6s} {m:18s} AP50={row['AP50']:.3f} F1={row['F1']:.3f}", flush=True)
    return rows


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader(); w.writerows(rows)


def main():
    """python run_experiments.py [detrac <condition>|bdd|merge]  (no args = everything)"""
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "parameter_reports"), exist_ok=True)
    mode = sys.argv[1:] or ["all"]
    if mode[0] in ("all", "detrac", "bdd"):
        models = load_models()
        ctrl_cfg = json.load(open(os.path.join(ROOT, "checkpoints", "ratfa_condition_params.json")))
        scene_clf = SceneClassifier.load(os.path.join(ROOT, "checkpoints", "scene_classifier.pkl"))
    if mode[0] in ("all", "detrac"):
        conds = mode[1:] or CONDITIONS
        for c in conds:
            rows, logs = experiment_detrac(models, ctrl_cfg, scene_clf, [c])
            json.dump({"rows": rows, "logs": logs},
                      open(os.path.join(OUT, "results", f"_part_{c}.json"), "w"))
    if mode[0] in ("all", "bdd"):
        bdd = experiment_bdd(models, ctrl_cfg, scene_clf)
        json.dump(bdd, open(os.path.join(OUT, "results", "bdd100k_results.json"), "w"), indent=1)
        write_csv(bdd, os.path.join(OUT, "results", "bdd100k_results.csv"))
    if mode[0] in ("all", "merge"):
        rows, logs = [], {}
        for c in CONDITIONS:
            part = json.load(open(os.path.join(OUT, "results", f"_part_{c}.json")))
            rows += part["rows"]; logs.update(part["logs"])
        json.dump(rows, open(os.path.join(OUT, "results", "ua_detrac_results.json"), "w"), indent=1)
        write_csv(rows, os.path.join(OUT, "results", "ua_detrac_results.csv"))
        json.dump(logs, open(os.path.join(OUT, "parameter_reports", "ratfa_gate_logs.json"), "w"))
        for c in CONDITIONS:
            os.remove(os.path.join(OUT, "results", f"_part_{c}.json"))


if __name__ == "__main__":
    main()
