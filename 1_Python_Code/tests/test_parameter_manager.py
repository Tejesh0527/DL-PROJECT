"""
tests/test_parameter_manager.py
=================================
Tests for the fuzzy parameter manager and inspector.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from fuzzy.parameter_inspector import (
    FuzzyParameterInspector,
    FuzzyLayerParameters,
    FuzzyParameterReport,
)
from fuzzy.parameter_manager import FuzzyParameterManager, ControlSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reference_report() -> FuzzyParameterReport:
    report = FuzzyParameterReport(
        model_name="test_model",
        integration_status="mock",
    )
    report.layers = [
        FuzzyLayerParameters(
            layer_name="ffm.fuzzy",
            centres=np.array([0.2, 0.5, 0.8]),
            sigmas=np.array([0.15, 0.20, 0.15]),
        ),
        FuzzyLayerParameters(
            layer_name="fa.fuzzy",
            centres=np.array([0.3, 0.7]),
            sigmas=np.array([0.20, 0.20]),
        ),
    ]
    return report


def make_control_signal(q=0.8, r=0.7, tau=0.75) -> ControlSignal:
    return ControlSignal(
        smoothed_condition=np.array([0.5, 0.2, 0.2, 0.1]),
        reliability=r,
        temporal_consistency=tau,
        scene_confidence=q,
    )


# ---------------------------------------------------------------------------
# FuzzyLayerParameters tests
# ---------------------------------------------------------------------------

class TestFuzzyLayerParameters:

    def test_valid_construction(self):
        layer = FuzzyLayerParameters(
            layer_name="test",
            centres=np.array([0.3, 0.5]),
            sigmas=np.array([0.2, 0.2]),
        )
        assert layer.num_functions == 2

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            FuzzyLayerParameters(
                layer_name="test",
                centres=np.array([0.3, 0.5, 0.7]),
                sigmas=np.array([0.2, 0.2]),   # wrong length
            )

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError):
            FuzzyLayerParameters(
                layer_name="test",
                centres=np.array([0.3]),
                sigmas=np.array([0.0]),   # invalid
            )

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError):
            FuzzyLayerParameters(
                layer_name="test",
                centres=np.array([0.3]),
                sigmas=np.array([-0.1]),
            )

    def test_to_dict(self):
        layer = FuzzyLayerParameters("test", np.array([0.5]), np.array([0.2]))
        d = layer.to_dict()
        assert "centres" in d
        assert "sigmas" in d
        assert d["num_functions"] == 1


# ---------------------------------------------------------------------------
# FuzzyParameterInspector tests
# ---------------------------------------------------------------------------

class TestFuzzyParameterInspector:

    def test_blocked_report_when_model_none(self):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()
        assert report.integration_status == "blocked"
        assert len(report.layers) > 0  # mock layers provided

    def test_blocked_report_has_notes(self):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()
        assert len(report.notes) > 0

    def test_mock_sigmas_positive(self):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()
        for layer in report.layers:
            assert np.all(layer.sigmas > 0)

    def test_validation_passes_for_valid_report(self):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()
        issues = inspector.validate_parameters(report)
        assert issues == [], f"Unexpected validation issues: {issues}"

    def test_validation_catches_invalid_sigma(self):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()
        # Manually corrupt a sigma
        bad_layer = FuzzyLayerParameters.__new__(FuzzyLayerParameters)
        bad_layer.layer_name = "bad"
        bad_layer.centres = np.array([0.5])
        bad_layer.sigmas = np.array([-1.0])  # bypass __post_init__ for test
        bad_layer.metadata = {}
        report.layers.append(bad_layer)
        issues = inspector.validate_parameters(report)
        assert len(issues) > 0

    def test_summary_contains_model_name(self):
        inspector = FuzzyParameterInspector(model=None)
        report = inspector.inspect()
        summary = report.summary()
        assert "F-YOLO" in summary or "BLOCKED" in summary


# ---------------------------------------------------------------------------
# ControlSignal tests (Multiplicative Confidence Gate: gamma_t = q_t * r_t * tau_t)
# ---------------------------------------------------------------------------

class TestControlSignal:

    def test_gate_all_ones(self):
        sig = make_control_signal(q=1.0, r=1.0, tau=1.0)
        assert sig.combined_gate() == pytest.approx(1.0)

    def test_gate_any_zero_yields_zero(self):
        # q = 0
        sig_q0 = make_control_signal(q=0.0, r=1.0, tau=1.0)
        assert sig_q0.combined_gate() == pytest.approx(0.0)

        # r = 0
        sig_r0 = make_control_signal(q=1.0, r=0.0, tau=1.0)
        assert sig_r0.combined_gate() == pytest.approx(0.0)

        # tau = 0
        sig_tau0 = make_control_signal(q=1.0, r=1.0, tau=0.0)
        assert sig_tau0.combined_gate() == pytest.approx(0.0)

        # all zero
        sig_all0 = make_control_signal(q=0.0, r=0.0, tau=0.0)
        assert sig_all0.combined_gate() == pytest.approx(0.0)

    def test_gate_multiplicative_exact_value(self):
        q, r, tau = 0.8, 0.7, 0.75
        sig = make_control_signal(q=q, r=r, tau=tau)
        expected = q * r * tau
        assert sig.combined_gate() == pytest.approx(expected, abs=1e-9)

    def test_gate_low_confidence_suppression(self):
        sig = make_control_signal(q=0.1, r=0.5, tau=0.5)
        gate = sig.combined_gate()
        assert gate == pytest.approx(0.025, abs=1e-9)
        assert gate < 0.1  # Below standard gate_threshold

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            make_control_signal(q=1.5, r=0.5, tau=0.5)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            make_control_signal(q=-0.1, r=0.5, tau=0.5)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            make_control_signal(q=0.5, r=1.2, tau=0.5)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            make_control_signal(q=0.5, r=0.5, tau=-0.2)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be a number"):
            make_control_signal(q="invalid", r=0.5, tau=0.5)

    def test_gate_boundary_values(self):
        for q in [0.0, 1.0]:
            for r in [0.0, 1.0]:
                for tau in [0.0, 1.0]:
                    sig = make_control_signal(q=q, r=r, tau=tau)
                    assert sig.combined_gate() == pytest.approx(q * r * tau)


# ---------------------------------------------------------------------------
# FuzzyParameterManager tests
# ---------------------------------------------------------------------------

class TestFuzzyParameterManager:

    def setup_method(self):
        self.report = make_reference_report()
        self.manager = FuzzyParameterManager(
            reference_report=self.report,
            max_centre_frac=0.10,
            max_sigma_frac=0.15,
            gate_threshold=0.1,
        )

    def test_no_op_test_passes_after_reset(self):
        assert self.manager.no_op_test() is True

    def test_update_returns_correct_number_of_layers(self):
        sig = make_control_signal()
        updated = self.manager.update(sig, frame_number=1)
        assert len(updated) == len(self.report.layers)

    def test_updated_sigmas_remain_positive(self):
        sig = make_control_signal()
        for _ in range(20):
            updated = self.manager.update(sig, frame_number=1)
        for layer in updated:
            assert np.all(layer.sigmas > 0), f"Non-positive sigma: {layer.sigmas}"

    def test_centres_within_bounds(self):
        sig = make_control_signal()
        for _ in range(20):
            updated = self.manager.update(sig)
        for i, (layer, ref) in enumerate(
            zip(updated, self.report.layers)
        ):
            max_change = np.abs(ref.centres) * self.manager.max_centre_frac
            assert np.all(
                np.abs(layer.centres - ref.centres) <= max_change + 1e-9
            ), f"Layer {i}: centres exceeded bounds"

    def test_low_gate_suppresses_update(self):
        """Below gate_threshold, parameters should not change."""
        self.manager.reset()
        sig_low = ControlSignal(
            smoothed_condition=np.array([0.25, 0.25, 0.25, 0.25]),
            reliability=0.0,
            temporal_consistency=0.0,
            scene_confidence=0.0,
        )
        updated = self.manager.update(sig_low, frame_number=0)
        for layer, ref in zip(updated, self.report.layers):
            assert np.allclose(layer.centres, ref.centres, atol=1e-9)

    def test_update_log_recorded(self):
        sig = make_control_signal()
        self.manager.update(sig, frame_number=0)
        log = self.manager.get_update_log()
        assert len(log) >= 1
        assert "gate_alpha" in log[0]

    def test_save_update_log(self, tmp_path):
        sig = make_control_signal()
        self.manager.update(sig, frame_number=0)
        path = str(tmp_path / "log.json")
        self.manager.save_update_log(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_reset_restores_reference(self):
        sig = make_control_signal()
        self.manager.update(sig, frame_number=1)
        self.manager.reset()
        for layer, ref in zip(self.manager._current_params, self.report.layers):
            assert np.allclose(layer.centres, ref.centres, atol=1e-9)
            assert np.allclose(layer.sigmas, ref.sigmas, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

