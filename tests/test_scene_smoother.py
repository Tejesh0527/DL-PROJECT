"""
tests/test_scene_smoother.py
==============================
Unit tests for the causal temporal scene smoother (Eq. 2).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from scene_analysis.condition_smoother import ConditionSmoother


class TestConditionSmootherInit:

    def test_default_construction(self):
        smoother = ConditionSmoother()
        assert smoother.lambda_e == 0.7
        assert smoother.num_conditions == 4
        assert smoother.smoothed_vector() is None

    def test_custom_lambda(self):
        smoother = ConditionSmoother(lambda_e=0.9)
        assert smoother.lambda_e == 0.9

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            ConditionSmoother(lambda_e=1.5)

    def test_invalid_lambda_negative_raises(self):
        with pytest.raises(ValueError):
            ConditionSmoother(lambda_e=-0.1)


class TestConditionSmootherUpdate:

    def test_first_update_equals_input(self):
        """First frame: ē_0 = e_0 (no previous state)."""
        smoother = ConditionSmoother(lambda_e=0.7)
        vec = np.array([0.7, 0.1, 0.1, 0.1])
        result = smoother.update(vec)
        # After normalisation, proportions are preserved
        assert np.allclose(result, vec / vec.sum(), atol=1e-9)

    def test_causal_equation(self):
        """Verify Eq. (2): ē_t = λ·ē_{t-1} + (1-λ)·e_t."""
        lambda_e = 0.7
        smoother = ConditionSmoother(lambda_e=lambda_e)
        e0 = np.array([1.0, 0.0, 0.0, 0.0])
        e1 = np.array([0.0, 1.0, 0.0, 0.0])
        e_bar_0 = smoother.update(e0)
        e_bar_1 = smoother.update(e1)
        expected_unnorm = lambda_e * e_bar_0 + (1 - lambda_e) * e1
        expected = expected_unnorm / expected_unnorm.sum()
        assert np.allclose(e_bar_1, expected, atol=1e-9)

    def test_output_sums_to_one(self):
        """Smoothed vector must always sum to 1."""
        smoother = ConditionSmoother(lambda_e=0.5)
        for _ in range(10):
            vec = np.random.default_rng(42).dirichlet(np.ones(4))
            result = smoother.update(vec)
            assert abs(result.sum() - 1.0) < 1e-9

    def test_output_non_negative(self):
        smoother = ConditionSmoother(lambda_e=0.5)
        vec = np.array([0.6, 0.2, 0.1, 0.1])
        result = smoother.update(vec)
        assert np.all(result >= 0.0)

    def test_invalid_input_dimension_raises(self):
        smoother = ConditionSmoother()
        with pytest.raises(ValueError):
            smoother.update(np.array([0.5, 0.5]))   # wrong length

    def test_negative_input_raises(self):
        smoother = ConditionSmoother()
        with pytest.raises(ValueError):
            smoother.update(np.array([-0.1, 0.4, 0.4, 0.3]))


class TestConditionSmootherReset:

    def test_reset_clears_state(self):
        smoother = ConditionSmoother(lambda_e=0.7)
        vec = np.array([0.7, 0.1, 0.1, 0.1])
        smoother.update(vec)
        assert smoother.smoothed_vector() is not None
        smoother.reset()
        assert smoother.smoothed_vector() is None

    def test_after_reset_first_update_equals_input(self):
        smoother = ConditionSmoother(lambda_e=0.7)
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        smoother.update(v1)
        smoother.reset()
        result = smoother.update(v2)
        assert np.allclose(result, v2 / v2.sum(), atol=1e-9)


class TestConditionSmootherBatchProcessing:

    def test_process_sequence_length(self):
        smoother = ConditionSmoother(lambda_e=0.7)
        vecs = [np.array([0.7, 0.1, 0.1, 0.1]) for _ in range(5)]
        results = smoother.process_sequence(vecs)
        assert len(results) == 5

    def test_process_sequence_causal(self):
        """process_sequence must produce same result as sequential updates."""
        smoother = ConditionSmoother(lambda_e=0.6)
        vecs = [np.random.default_rng(i).dirichlet(np.ones(4)) for i in range(8)]
        seq_results = smoother.process_sequence(vecs)

        # Compare against manual sequential calls
        smoother.reset()
        manual_results = [smoother.update(v) for v in vecs]
        for s, m in zip(seq_results, manual_results):
            assert np.allclose(s, m, atol=1e-12)

    def test_lambda_effect(self):
        """Higher lambda → slower response (ē closer to previous state)."""
        vecs = [np.array([1.0, 0.0, 0.0, 0.0])] * 3 + \
               [np.array([0.0, 0.0, 0.0, 1.0])] * 3

        smoother_slow = ConditionSmoother(lambda_e=0.95)
        smoother_fast = ConditionSmoother(lambda_e=0.1)

        slow_results = smoother_slow.process_sequence(vecs)
        fast_results = smoother_fast.process_sequence(vecs)

        # At the last frame, the dark component should rise faster
        # for the low-lambda (fast) smoother
        assert fast_results[-1][3] > slow_results[-1][3]


class TestConditionSmootherDictInterface:

    def test_update_from_dict(self):
        smoother = ConditionSmoother()
        d = {"clear": 0.7, "rain": 0.1, "fog": 0.1, "dark": 0.1}
        result = smoother.update_from_dict(d)
        assert result.shape == (4,)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_smoothed_dict_keys(self):
        smoother = ConditionSmoother()
        smoother.update(np.array([0.5, 0.2, 0.2, 0.1]))
        d = smoother.smoothed_dict()
        assert set(d.keys()) == {"clear", "rain", "fog", "dark"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

