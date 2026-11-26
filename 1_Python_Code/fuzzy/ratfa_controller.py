"""RATFA controller with the multiplicative confidence gate.

    gamma_t = q_t * r_t * tau_t                                 (confidence gate)
    theta*_t = sum_c  e_bar_t[c] * theta_c                       (scene-conditioned target)
    theta_t  = clip(theta_{t-1} + eta * gamma_t * (theta*_t - theta_{t-1}),
                    theta_0 - b, theta_0 + b)                    (bounded, gated update)

theta are the Gaussian membership centres and widths of every fuzzy unit
(FFM / FA) in the F-YOLO-style detector. theta_c are per-condition membership
parameters obtained by re-centring the clear-weather memberships on the
descriptor distribution observed under condition c (see
experiments/calibrate_conditions.py). If gamma_t < gate_min the update is
skipped and the current parameters are kept.
"""
from typing import Dict, Optional

import numpy as np
import torch

from detector.fyolo_model import fuzzy_sets
from scene_analysis.scene_classifier import CONDITIONS


class RATFAController:
    def __init__(self, model: torch.nn.Module, condition_params: Dict[str, Dict[str, Dict]],
                 eta: float = 0.5, bound: float = 1.0, gate_min: float = 0.1,
                 use_gate: bool = True):
        self.sets = fuzzy_sets(model)
        self.theta0 = {n: (s.centres.detach().clone(), s.sigmas.detach().clone())
                       for n, s in self.sets.items()}
        self.targets = {c: {n: (torch.tensor(v["centres"]), torch.tensor(v["sigmas"]))
                            for n, v in condition_params[c].items()} for c in CONDITIONS}
        self.eta, self.bound, self.gate_min, self.use_gate = eta, bound, gate_min, use_gate
        self.log = []

    def reset(self):
        with torch.no_grad():
            for n, s in self.sets.items():
                s.centres.copy_(self.theta0[n][0])
                s.sigmas.copy_(self.theta0[n][1])
        self.log = []

    @torch.no_grad()
    def step(self, e_bar: np.ndarray, q: float, r: float, tau: float, frame: Optional[int] = None):
        gamma = float(q * r * tau) if self.use_gate else 1.0
        applied = gamma >= self.gate_min
        delta_norm = 0.0
        if applied:
            for n, s in self.sets.items():
                c_t = sum(float(e_bar[i]) * self.targets[c][n][0] for i, c in enumerate(CONDITIONS))
                s_t = sum(float(e_bar[i]) * self.targets[c][n][1] for i, c in enumerate(CONDITIONS))
                c0, s0 = self.theta0[n]
                new_c = s.centres + self.eta * gamma * (c_t.to(s.centres) - s.centres)
                new_s = s.sigmas + self.eta * gamma * (s_t.to(s.sigmas) - s.sigmas)
                new_c = torch.max(torch.min(new_c, c0 + self.bound), c0 - self.bound)
                new_s = torch.max(torch.min(new_s, s0 * 2.0), s0 * 0.5).clamp_min(1e-3)
                delta_norm += float((new_c - s.centres).norm() + (new_s - s.sigmas).norm())
                s.centres.copy_(new_c)
                s.sigmas.copy_(new_s)
        self.log.append({"frame": frame, "q": q, "r": r, "tau": tau, "gamma": gamma,
                         "applied": bool(applied), "delta_norm": delta_norm,
                         "e_bar": [float(x) for x in e_bar]})
        return gamma, applied
