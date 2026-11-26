"""
fuzzy/parameter_inspector.py
==============================
Safe fuzzy parameter inspection utilities for RATFA.

Purpose
-------
This module inspects the fuzzy membership parameters of an F-YOLO model
(or placeholder structures during testing).  It does NOT modify any weights.

F-YOLO Integration Status: BLOCKED
------------------------------------
The actual F-YOLO source code is not available.  When it becomes available,
the inspection functions in this module will locate and extract:
  • Gaussian membership centres (c) for each fuzzy layer
  • Gaussian membership widths (σ) for each fuzzy layer
  • Parameter tensor shapes
  • Initial (reference) parameter values

Until then, the inspector works with mock parameter structures to demonstrate
the inspection interface.

IMPORTANT:
  • Do NOT modify detector backbone weights.
  • Do NOT implement online adaptation here (see parameter_manager.py).
  • Do NOT invent actual F-YOLO parameter values.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SIGMA_MIN = 1e-6   # Hard lower bound: widths must remain strictly positive


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FuzzyLayerParameters:
    """Fuzzy membership parameters for a single layer.

    In F-YOLO, each fuzzy block (FFM or FA) contains a set of K Gaussian
    membership functions, each characterised by a centre c and a width σ.
    """
    layer_name: str
    centres: np.ndarray          # shape (K,)
    sigmas: np.ndarray           # shape (K,)  all > 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.centres = np.asarray(self.centres, dtype=np.float64)
        self.sigmas = np.asarray(self.sigmas, dtype=np.float64)
        if self.centres.shape != self.sigmas.shape:
            raise ValueError(
                f"centres and sigmas must have the same shape in layer "
                f"'{self.layer_name}'; got {self.centres.shape} vs {self.sigmas.shape}"
            )
        if np.any(self.sigmas <= 0):
            raise ValueError(
                f"All sigmas must be > 0 in layer '{self.layer_name}'; "
                f"got min(σ) = {self.sigmas.min()}"
            )

    @property
    def num_functions(self) -> int:
        return len(self.centres)

    def to_dict(self) -> Dict:
        return {
            "layer_name": self.layer_name,
            "num_functions": self.num_functions,
            "centres": self.centres.tolist(),
            "sigmas": self.sigmas.tolist(),
            "metadata": self.metadata,
        }


@dataclass
class FuzzyParameterReport:
    """Report summarising fuzzy parameter inspection for a whole model."""
    model_name: str
    integration_status: str    # "available" | "blocked" | "mock"
    layers: List[FuzzyLayerParameters] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def total_parameters(self) -> int:
        """Total number of inspected fuzzy parameters (centres + sigmas)."""
        return sum(2 * layer.num_functions for layer in self.layers)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "integration_status": self.integration_status,
            "total_fuzzy_parameters": self.total_parameters(),
            "num_fuzzy_layers": len(self.layers),
            "layers": [layer.to_dict() for layer in self.layers],
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Model        : {self.model_name}",
            f"Status       : {self.integration_status}",
            f"Fuzzy layers : {len(self.layers)}",
            f"Total params : {self.total_parameters()}",
        ]
        for layer in self.layers:
            lines.append(
                f"  [{layer.layer_name}] K={layer.num_functions} "
                f"c={np.round(layer.centres, 4)} "
                f"σ={np.round(layer.sigmas, 4)}"
            )
        for note in self.notes:
            lines.append(f"NOTE: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parameter inspector
# ---------------------------------------------------------------------------


class FuzzyParameterInspector:
    """Inspect fuzzy membership parameters from an F-YOLO model.

    When F-YOLO is available:
        inspector = FuzzyParameterInspector(model=fyolo_model)
        report = inspector.inspect()

    When BLOCKED (current status):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()   # returns mock report with status=blocked
    """

    # Expected layer name patterns in F-YOLO (from paper description).
    # These will be verified against the actual model when available.
    EXPECTED_LAYER_PATTERNS = [
        "ffm",          # Fuzzy Fusion Module layers
        "fa",           # Fuzzy Attention layers
        "gpblock",      # GhostBlock pooling layers (may also contain fuzzy params)
        "fuzzy",        # generic fuzzy parameter containers
        "membership",
    ]

    def __init__(self, model: Any = None):
        """
        Parameters
        ----------
        model : Any
            Loaded F-YOLO model (PyTorch nn.Module).  Pass None when BLOCKED.
        """
        self.model = model
        self._report: Optional[FuzzyParameterReport] = None

    def inspect(self) -> FuzzyParameterReport:
        """Inspect and return a parameter report.

        Returns
        -------
        FuzzyParameterReport
            Contains layer names, centres, sigmas, and integration status.
        """
        if self.model is None:
            return self._blocked_report()

        try:
            return self._inspect_pytorch_model()
        except Exception as exc:
            logger.error("Parameter inspection failed: %s", exc)
            report = self._blocked_report()
            report.notes.append(f"Inspection error: {exc}")
            return report

    def _blocked_report(self) -> FuzzyParameterReport:
        """Return a clearly-labelled mock report for BLOCKED status."""
        report = FuzzyParameterReport(
            model_name="F-YOLO (BLOCKED)",
            integration_status="blocked",
            notes=[
                "F-YOLO source and checkpoint are not available.",
                "Parameter inspection cannot proceed without the actual model.",
                "Mock parameters below are for interface testing ONLY.",
                "Do NOT use these values as actual F-YOLO parameters.",
            ],
        )
        # Provide mock layers for interface testing
        mock_layers = [
            FuzzyLayerParameters(
                layer_name="ffm.fuzzy_layer_mock",
                centres=np.array([0.2, 0.5, 0.8]),
                sigmas=np.array([0.15, 0.20, 0.15]),
                metadata={"source": "mock", "verified": False},
            ),
            FuzzyLayerParameters(
                layer_name="fa.fuzzy_attention_mock",
                centres=np.array([0.3, 0.7]),
                sigmas=np.array([0.20, 0.20]),
                metadata={"source": "mock", "verified": False},
            ),
        ]
        report.layers = mock_layers
        logger.warning(
            "FuzzyParameterInspector: returning MOCK report (F-YOLO BLOCKED)"
        )
        return report

    def _inspect_pytorch_model(self) -> FuzzyParameterReport:
        """Attempt inspection of a PyTorch model for fuzzy parameters.

        This requires the actual F-YOLO source.  It searches named parameters
        for membership-function-related tensors.
        """
        import torch  # only imported when actually available

        report = FuzzyParameterReport(
            model_name=type(self.model).__name__,
            integration_status="available",
        )

        found_any = False
        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            if any(pat in name_lower for pat in self.EXPECTED_LAYER_PATTERNS):
                # Determine if this parameter is a centre or sigma
                param_data = param.detach().cpu().numpy().flatten()
                if "centre" in name_lower or "center" in name_lower or ".c" in name_lower:
                    # Try to find paired sigma
                    sigma_name = name.replace("centre", "sigma").replace(
                        "center", "sigma"
                    )
                    sigma_param = dict(self.model.named_parameters()).get(sigma_name)
                    sigma_data = (
                        np.abs(sigma_param.detach().cpu().numpy().flatten())
                        if sigma_param is not None
                        else np.full_like(param_data, 0.2)
                    )
                    sigma_data = np.maximum(sigma_data, SIGMA_MIN)
                    layer = FuzzyLayerParameters(
                        layer_name=name,
                        centres=param_data,
                        sigmas=sigma_data,
                        metadata={"shape": list(param.shape)},
                    )
                    report.layers.append(layer)
                    found_any = True
                    logger.info("Inspected fuzzy layer: %s", name)

        if not found_any:
            report.integration_status = "partial"
            report.notes.append(
                "No fuzzy membership parameters found matching expected patterns. "
                "The F-YOLO model may use different naming conventions.  "
                "Inspect the model's named_parameters() manually."
            )
        return report

    def validate_parameters(self, report: FuzzyParameterReport) -> List[str]:
        """Validate a parameter report; return list of issues.

        Parameters
        ----------
        report : FuzzyParameterReport

        Returns
        -------
        List[str]
            List of validation issues.  Empty list ⟹ all checks pass.
        """
        issues = []
        for layer in report.layers:
            if np.any(layer.sigmas <= 0):
                issues.append(
                    f"Layer '{layer.layer_name}': invalid sigma ≤ 0 detected."
                )
            if np.any(np.isnan(layer.centres)) or np.any(np.isinf(layer.centres)):
                issues.append(
                    f"Layer '{layer.layer_name}': NaN or Inf in centres."
                )
            if np.any(np.isnan(layer.sigmas)) or np.any(np.isinf(layer.sigmas)):
                issues.append(
                    f"Layer '{layer.layer_name}': NaN or Inf in sigmas."
                )
        return issues

