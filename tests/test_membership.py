"""
tests/test_membership.py
=========================
Unit tests for the Gaussian fuzzy membership function (Eq. 8).
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from fuzzy.membership import (
    gaussian_membership,
    evaluate_membership_set,
    gaussian_membership_gradient_c,
    gaussian_membership_gradient_sigma,
)


# ---------------------------------------------------------------------------
# Test: scalar inputs
# ---------------------------------------------------------------------------

class TestGaussianMembershipScalar:

    def test_at_centre_returns_one(self):
        """μ(c; c, σ) = 1 for any valid c and σ."""
        result = gaussian_membership(0.5, c=0.5, sigma=0.2)
        assert math.isclose(result, 1.0, abs_tol=1e-9), f"Expected 1.0, got {result}"

    def test_decreases_away_from_centre(self):
        """μ decreases monotonically as x moves away from c."""
        c, sigma = 0.5, 0.2
        mu_at_c = gaussian_membership(c, c, sigma)
        mu_near = gaussian_membership(c + 0.1, c, sigma)
        mu_far = gaussian_membership(c + 0.3, c, sigma)
        assert mu_at_c > mu_near > mu_far

    def test_symmetric_about_centre(self):
        """μ is symmetric: μ(c+d; c, σ) == μ(c-d; c, σ)."""
        c, sigma, d = 0.5, 0.2, 0.15
        mu_right = gaussian_membership(c + d, c, sigma)
        mu_left = gaussian_membership(c - d, c, sigma)
        assert math.isclose(mu_right, mu_left, abs_tol=1e-9)

    def test_range_zero_to_one(self):
        """μ ∈ [0, 1] for all inputs."""
        for x in [-1.0, 0.0, 0.5, 1.0, 10.0]:
            mu = gaussian_membership(x, c=0.5, sigma=0.2)
            assert 0.0 <= mu <= 1.0, f"μ={mu} out of [0,1] for x={x}"

    def test_different_centres(self):
        """Different centres shift the peak location."""
        sigma = 0.2
        mu_c05 = gaussian_membership(0.5, c=0.5, sigma=sigma)
        mu_c08 = gaussian_membership(0.8, c=0.8, sigma=sigma)
        assert math.isclose(mu_c05, 1.0, abs_tol=1e-9)
        assert math.isclose(mu_c08, 1.0, abs_tol=1e-9)

    def test_width_effect(self):
        """Smaller σ → steeper falloff; larger σ → broader bell."""
        c, x = 0.5, 0.7
        mu_narrow = gaussian_membership(x, c=c, sigma=0.1)
        mu_wide = gaussian_membership(x, c=c, sigma=0.5)
        assert mu_narrow < mu_wide, (
            f"Narrow width should give smaller μ at x≠c; "
            f"narrow={mu_narrow:.4f}, wide={mu_wide:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: invalid sigma
# ---------------------------------------------------------------------------

class TestGaussianMembershipInvalidSigma:

    def test_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            gaussian_membership(0.5, c=0.5, sigma=0.0)

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            gaussian_membership(0.5, c=0.5, sigma=-0.1)

    def test_array_with_zero_sigma_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            gaussian_membership(
                np.array([0.1, 0.5, 0.9]),
                c=0.5,
                sigma=np.array([0.2, 0.0, 0.2]),  # second sigma is 0
            )


# ---------------------------------------------------------------------------
# Test: array inputs
# ---------------------------------------------------------------------------

class TestGaussianMembershipArray:

    def test_array_input_returns_array(self):
        x = np.linspace(0.0, 1.0, 11)
        result = gaussian_membership(x, c=0.5, sigma=0.2)
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape

    def test_array_values_in_range(self):
        x = np.linspace(-2.0, 2.0, 100)
        result = gaussian_membership(x, c=0.0, sigma=0.5)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_array_peak_at_centre(self):
        c = 0.5
        x = np.linspace(0.0, 1.0, 1001)
        result = gaussian_membership(x, c=c, sigma=0.2)
        peak_idx = np.argmax(result)
        assert math.isclose(x[peak_idx], c, abs_tol=0.002)


# ---------------------------------------------------------------------------
# Test: evaluate_membership_set
# ---------------------------------------------------------------------------

class TestEvaluateMembershipSet:

    def test_multiple_centres(self):
        centres = np.array([0.2, 0.5, 0.8])
        sigmas = np.array([0.15, 0.20, 0.15])
        result = evaluate_membership_set(0.5, centres, sigmas)
        assert result.shape == (3,)
        # Peak of middle function at x=0.5
        assert math.isclose(result[1], 1.0, abs_tol=1e-9)
        # Side functions should have lower membership at x=0.5
        assert result[0] < 1.0
        assert result[2] < 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            evaluate_membership_set(
                0.5,
                np.array([0.2, 0.5]),
                np.array([0.15]),   # wrong length
            )


# ---------------------------------------------------------------------------
# Test: gradients
# ---------------------------------------------------------------------------

class TestGaussianGradients:

    def test_gradient_c_zero_at_centre(self):
        """∂μ/∂c = 0 when x = c (peak of Gaussian)."""
        grad = gaussian_membership_gradient_c(x=0.5, c=0.5, sigma=0.2)
        assert math.isclose(grad, 0.0, abs_tol=1e-9)

    def test_gradient_sigma_positive_away_from_centre(self):
        """∂μ/∂σ > 0 when x ≠ c (wider σ → higher μ away from c)."""
        grad = gaussian_membership_gradient_sigma(x=0.7, c=0.5, sigma=0.2)
        assert grad > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

