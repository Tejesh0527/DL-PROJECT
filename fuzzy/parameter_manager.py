"""
fuzzy/parameter_manager.py
===========================
Fuzzy parameter manager for RATFA – bounded update with confidence gating.

Implements Equations (7) through (11) from the paper:

  Eq. (7):  g_t = f(ē_t, r_t, τ_t)          ← combined control signal
  Eq. (8):  μ(x; c, σ) = exp(-(x-c)²/(2σ²))  ← Gaussian membership
  Eq. (9):  Δθ = Φ(g_t);  θ_proposed = clip(θ + Δθ, θ_min, θ_max)
  Eq. (10): γ_t = q_t · r_t · τ_t            ← multiplicative confidence gate γ_t ∈ [0,1]
  Eq. (11): θ_new = clip(θ + γ_t·Δθ, θ_min, θ_max)

The confidence gate α reduces the adaptation magnitude when:
  • scene-condition confidence q_t is low
  • detection reliability r_t is low
  • temporal consistency τ_t is low

Backbone weights are NEVER modified.
Only a bounded subset of Gaussian membership parameters is eligible.

IMPORTANT:
  This module implements the CONTROLLER INTERFACE for adaptation.
  Actual F-YOLO parameter tensors are not modified here because the model
  integration is BLOCKED.  The controller logic is fully testable with the
  mock parameter structures from parameter_inspector.py.
"""

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fuzzy.parameter_inspector import FuzzyLayerParameters, FuzzyParameterReport
from fuzzy.membership import gaussian_membership

logger = logging.getLogger(__name__)

SIGMA_MIN = 1e-6   # widths must remain strictly positive after any update


# ---------------------------------------------------------------------------
# Control signal
# ---------------------------------------------------------------------------


@dataclass
class ControlSignal:
    """RATFA control signal for a single frame (Eq. 7)."""
    smoothed_condition: np.ndarray   # ē_t, shape (4,)
    reliability: float               # r_t ∈ [0, 1]
    temporal_consistency: float      # τ_t ∈ [0, 1]
    scene_confidence: float          # q_t ∈ [0, 1]

    def __post_init__(self) -> None:
        """Validate input ranges strictly in [0, 1]."""
        for name, val in [
            ("reliability", self.reliability),
            ("temporal_consistency", self.temporal_consistency),
            ("scene_confidence", self.scene_confidence),
        ]:
            if not isinstance(val, (int, float, np.floating)):
                raise TypeError(f"{name} must be a number; got {type(val)}")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]; got {val}")

    def combined_gate(self) -> float:
        """Multiplicative confidence gate γ_t = q_t * r_t * τ_t (Eq. 10).

        Returns
        -------
        float
            Gate value γ_t in [0, 1]. If ANY input is zero, returns strictly 0.0.
        """
        gate = float(self.scene_confidence * self.reliability * self.temporal_consistency)
        return float(np.clip(gate, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Update record
# ---------------------------------------------------------------------------


@dataclass
class ParameterUpdateRecord:
    """Log entry for one RATFA parameter update step."""
    frame_number: int
    gate_alpha: float
    reliability: float
    temporal_consistency: float
    scene_confidence: float
    layers_updated: List[str] = field(default_factory=list)
    delta_norms: Dict[str, float] = field(default_factory=dict)
    update_applied: bool = True

    def to_dict(self) -> Dict:
        return {
            "frame_number": self.frame_number,
            "gate_alpha": self.gate_alpha,
            "reliability": self.reliability,
            "temporal_consistency": self.temporal_consistency,
            "scene_confidence": self.scene_confidence,
            "layers_updated": self.layers_updated,
            "delta_norms": self.delta_norms,
            "update_applied": self.update_applied,
        }


# ---------------------------------------------------------------------------
# Parameter manager
# ---------------------------------------------------------------------------


class FuzzyParameterManager:
    """Manages bounded updates of fuzzy membership parameters (RATFA Eqs. 9–11).

    Parameters
    ----------
    reference_report : FuzzyParameterReport
        The reference (trained) fuzzy parameters θ_ref.  These are never
        modified; they define the allowed update range.
    max_centre_frac : float
        Maximum fractional change allowed for each centre:
        |θ_new - θ_ref| ≤ max_centre_frac × |θ_ref|
    max_sigma_frac : float
        Maximum fractional change allowed for each sigma.
    gate_threshold : float
        Minimum gate value α required to apply any update.
        Below this, the reference parameters are kept unchanged.
    rate_limit : float
        Maximum absolute step per update (rate limiter, Eq. paper text).
        Prevents abrupt parameter jumps.
    """

    def __init__(
        self,
        reference_report: FuzzyParameterReport,
        max_centre_frac: float = 0.10,
        max_sigma_frac: float = 0.15,
        gate_threshold: float = 0.1,
        rate_limit: float = 0.05,
    ):
        self.reference_report = reference_report
        self.max_centre_frac = max_centre_frac
        self.max_sigma_frac = max_sigma_frac
        self.gate_threshold = gate_threshold
        self.rate_limit = rate_limit

        # Working copy of parameters (what RATFA currently proposes to use)
        self._current_params: List[FuzzyLayerParameters] = deepcopy(
            reference_report.layers
        )
        # Clip bounds derived once from reference values
        self._bounds = self._compute_bounds()

        self._update_log: List[ParameterUpdateRecord] = []

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def _compute_bounds(
        self,
    ) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Compute (min, max) bounds for each layer's centres and sigmas."""
        bounds = []
        for layer in self.reference_report.layers:
            c_ref = layer.centres
            s_ref = layer.sigmas
            c_range = np.abs(c_ref) * self.max_centre_frac
            s_range = np.abs(s_ref) * self.max_sigma_frac

            bounds.append({
                "centre_min": c_ref - c_range,
                "centre_max": c_ref + c_range,
                "sigma_min": np.maximum(s_ref - s_range, SIGMA_MIN),
                "sigma_max": s_ref + s_range,
            })
        return bounds

    # ------------------------------------------------------------------
    # Linear mapping (Φ): control signal → parameter delta (Eq. 9)
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_mapping(
        control_signal: ControlSignal,
        num_params: int,
        scale: float = 0.01,
    ) -> np.ndarray:
        """Linear low-dimensional mapping Φ(g_t) → Δθ.

        As recommended in the paper: "A practical first version should use a
        linear mapping with bounded outputs rather than a high-capacity
        recurrent controller."

        The mapping uses the dominant condition in ē_t to determine the sign
        and direction of the update.  The magnitude is proportional to the
        combined gate.

        Parameters
        ----------
        control_signal : ControlSignal
        num_params : int
            Number of parameter elements (K centres or K sigmas).
        scale : float
            Base step magnitude.

        Returns
        -------
        np.ndarray
            Unconstrained delta of shape (num_params,).
        """
        # Dominant condition index drives the update direction
        dominant_idx = int(np.argmax(control_signal.smoothed_condition))
        # Map: clear→small, rain→widen, fog→shift_up, dark→shift_down (heuristic)
        direction_map = {0: 0.0, 1: +1.0, 2: +0.5, 3: -0.5}
        direction = direction_map.get(dominant_idx, 0.0)
        gate = control_signal.combined_gate()
        delta = np.full(num_params, direction * scale * gate, dtype=np.float64)
        return delta

    # ------------------------------------------------------------------
    # Core update step
    # ------------------------------------------------------------------

    def update(
        self,
        control_signal: ControlSignal,
        frame_number: int = 0,
    ) -> List[FuzzyLayerParameters]:
        """Apply one RATFA adaptation step.

        Implements Eqs. (9)–(11):
          Δθ = Φ(g_t)                                              (Eq. 9a)
          θ_proposed = clip(θ_ref + Δθ, θ_min, θ_max)            (Eq. 9b)
          α = ψ(q_t, r_t, τ_t)                                    (Eq. 10)
          θ_new = clip(θ_current + α·Δθ, θ_min, θ_max)           (Eq. 11)

        Parameters
        ----------
        control_signal : ControlSignal
        frame_number : int

        Returns
        -------
        List[FuzzyLayerParameters]
            Updated (or unchanged) fuzzy parameters for each layer.
        """
        alpha = control_signal.combined_gate()
        record = ParameterUpdateRecord(
            frame_number=frame_number,
            gate_alpha=alpha,
            reliability=control_signal.reliability,
            temporal_consistency=control_signal.temporal_consistency,
            scene_confidence=control_signal.scene_confidence,
        )

        if alpha < self.gate_threshold:
            # Evidence insufficient – keep reference parameters
            logger.debug(
                "Frame %d: gate α=%.4f < threshold %.4f; keeping reference params.",
                frame_number, alpha, self.gate_threshold,
            )
            record.update_applied = False
            self._update_log.append(record)
            return deepcopy(self._current_params)

        updated_layers: List[FuzzyLayerParameters] = []
        for i, (layer, bounds) in enumerate(
            zip(self._current_params, self._bounds)
        ):
            # --- Centres ---
            delta_c = self._linear_mapping(control_signal, layer.num_functions)
            # Apply rate limit
            delta_c = np.clip(delta_c, -self.rate_limit, self.rate_limit)
            # Apply confidence gate
            new_centres = np.clip(
                layer.centres + alpha * delta_c,
                bounds["centre_min"],
                bounds["centre_max"],
            )

            # --- Sigmas (using log-width to guarantee positivity) ---
            delta_s = self._linear_mapping(
                control_signal, layer.num_functions, scale=0.005
            )
            delta_s = np.clip(delta_s, -self.rate_limit, self.rate_limit)
            new_sigmas = np.clip(
                layer.sigmas + alpha * delta_s,
                bounds["sigma_min"],
                bounds["sigma_max"],
            )
            # Ensure strictly positive
            new_sigmas = np.maximum(new_sigmas, SIGMA_MIN)

            delta_norm_c = float(np.linalg.norm(new_centres - layer.centres))
            delta_norm_s = float(np.linalg.norm(new_sigmas - layer.sigmas))
            record.delta_norms[layer.layer_name + ".centres"] = delta_norm_c
            record.delta_norms[layer.layer_name + ".sigmas"] = delta_norm_s
            record.layers_updated.append(layer.layer_name)

            updated_layer = FuzzyLayerParameters(
                layer_name=layer.layer_name,
                centres=new_centres,
                sigmas=new_sigmas,
                metadata=layer.metadata,
            )
            updated_layers.append(updated_layer)

        self._current_params = updated_layers
        self._update_log.append(record)
        logger.debug(
            "Frame %d: α=%.4f, %d layers updated.",
            frame_number, alpha, len(updated_layers),
        )
        return deepcopy(updated_layers)

    # ------------------------------------------------------------------
    # No-op verification (paper requirement)
    # ------------------------------------------------------------------

    def no_op_test(self) -> bool:
        """Verify that zero-delta update reproduces reference parameters.

        Returns True if the current parameters match the reference within
        numerical tolerance.  Should be run after reset().
        """
        self.reset()
        for layer, ref_layer in zip(
            self._current_params, self.reference_report.layers
        ):
            if not np.allclose(layer.centres, ref_layer.centres, atol=1e-9):
                return False
            if not np.allclose(layer.sigmas, ref_layer.sigmas, atol=1e-9):
                return False
        return True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset working parameters to the reference values."""
        self._current_params = deepcopy(self.reference_report.layers)
        logger.debug("FuzzyParameterManager: parameters reset to reference.")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_update_log(self) -> List[Dict]:
        return [r.to_dict() for r in self._update_log]

    def save_update_log(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_update_log(), f, indent=2)
        logger.info("Parameter update log saved → %s", path)

    def save_current_parameters(self, path: str) -> None:
        """Save current parameter state to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "current_parameters": [p.to_dict() for p in self._current_params]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Current parameters saved → %s", path)

