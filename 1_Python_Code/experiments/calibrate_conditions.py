"""Per-condition fuzzy membership targets theta_c for the RATFA controller.

For every condition c the trained F-YOLO model is run on UA-DETRAC *training*
frames corrupted with condition c (calibration severity 0.8, seeds disjoint
from the test split). For every fuzzy unit we record the standardised
descriptor z. Its mean m_c and std s_c describe how the feature distribution
shifts under c; the clear-weather memberships are re-centred accordingly:
    c_k(c) = m_c + s_c * c_k(0),     sigma_k(c) = s_c * sigma_k(0)
"""
import glob
import json
import os
import sys

import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".ultralytics"))
from detector.fyolo_model import _letterbox, fuzzy_sets  # noqa: E402
from experiments.corruptions import CONDITIONS, apply  # noqa: E402


@torch.no_grad()
def main(imgsz=512):
    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "checkpoints", "fyolo_detrac.pt")).model.float().eval()
    sets = fuzzy_sets(model)
    frames = sorted(glob.glob(os.path.join(ROOT, "datasets", "UA-DETRAC", "train", "*", "*.jpg")))[::4]
    captured = {}

    def hook(name):
        def f(mod, inp, out):
            captured.setdefault(name, []).append(out.detach().reshape(-1))
        return f

    out = {}
    for c in CONDITIONS:
        captured.clear()
        hooks = []
        for n, s in sets.items():   # capture z = standardise(d) via a wrapper
            orig = s.standardise

            def wrapped(d, _o=orig, _n=n):
                z = _o(d)
                captured.setdefault(_n, []).append(z.detach().reshape(-1))
                return z
            s.standardise = wrapped
            hooks.append((s, orig))
        for i, p in enumerate(frames):
            img = apply(cv2.imread(p), c, seed=5000 + i, severity=0.8)
            model(_letterbox(img, imgsz)[None])
        for s, orig in hooks:
            s.standardise = orig
        out[c] = {}
        for n, s in sets.items():
            z = torch.cat(captured[n])
            m, sd = float(z.mean()), float(z.std().clamp(0.25, 4.0))
            out[c][n] = {"shift": m, "scale": sd,
                         "centres": (m + sd * s.centres.detach()).tolist(),
                         "sigmas": (sd * s.sigmas.detach().abs()).tolist()}
    # express every condition relative to clear weather, so theta_clear == theta_0
    for c in CONDITIONS:
        for n, s in sets.items():
            m = out[c][n]["shift"] - out["clear"][n]["shift"]
            sd = out[c][n]["scale"] / out["clear"][n]["scale"]
            mc = out["clear"][n]["shift"]
            out[c][n].update(rel_shift=m, rel_scale=sd,
                             centres=(mc + m + sd * (s.centres.detach() - mc)).tolist(),
                             sigmas=(sd * s.sigmas.detach().abs()).tolist())
        print(c, {n.split('.')[1]: round(v["rel_shift"], 2) for n, v in out[c].items()})
    os.makedirs(os.path.join(ROOT, "outputs", "parameter_reports"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "checkpoints", "ratfa_condition_params.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(ROOT, "outputs", "parameter_reports",
                                     "ratfa_condition_params.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
