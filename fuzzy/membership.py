"""
fuzzy/membership.py
====================
Gaussian fuzzy membership function (Equation 8 of the RATFA paper).

Equation (8):
  μ(x; c, σ) = exp(-(x - c)² / (2σ²))

Terminology
-----------
  c (centre):
    The input value at which the membership function reaches its peak (μ = 1).
    Shifting c moves the region of strongest activation along the input axis.
    In F-YOLO, centres are trainable: the RATFA controller adjusts them within
    bounded limits to adapt to current scene conditions.

  σ (width / sigma):
    Controls how broadly the membership function responds around c.
    - Small σ : narrow bell → only inputs very close to c get high membership.
      Risk: brittle; slight deviation causes rapid drop in membership.
    - Large σ : wide bell → inputs far from c still have appreciable membership.
      Risk: blurs distinctions between different input ranges.
    σ must remain strictly positive (σ > 0); a zero or negative width produces
    an invalid membership function.

IMPORTANT CAVEAT
----------------
Implementing this mathematical function does NOT mean that the actual F-YOLO
Fuzzy Fusion Module (FFM), Fuzzy Attention (FA), or GhostBlock (GPBlock)
components have been integrated.  Those require the original F-YOLO source code
and checkpoint, both of which are currently BLOCKED.
"""

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Hard lower bound on sigma to prevent division by zero
SIGMA_MIN = 1e-9


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def gaussian_membership(
    x: Union[float, np.ndarray],
    c: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Gaussian fuzzy membership function (Eq. 8).

    μ(x; c, σ) = exp(-(x - c)² / (2σ²))

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) at which to evaluate the membership function.
    c : float or np.ndarray
        Centre of the Gaussian: the input at which μ = 1.
        Must be broadcastable with x.
    sigma : float or np.ndarray
        Width of the Gaussian: σ > 0 is enforced.
        Must be broadcastable with x.

    Returns
    -------
    float or np.ndarray
        Membership degree(s) in [0, 1].
        Returns the same type as the input x.

    Raises
    ------
    ValueError
        If any element of sigma ≤ 0.

    Examples
    --------
    >>> gaussian_membership(0.5, c=0.5, sigma=0.2)
    1.0
    >>> gaussian_membership(0.0, c=0.5, sigma=0.2)
    0.0439...
    >>> import numpy as np
    >>> gaussian_membership(np.linspace(0, 1, 5), c=0.5, sigma=0.2)
    array([0.0439..., 0.3247..., 1.0, 0.3247..., 0.0439...])
    """
    scalar_input = np.isscalar(x)

    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Validate sigma > 0
    if np.any(sigma <= 0):
        bad = sigma[sigma <= 0] if sigma.ndim > 0 else sigma
        raise ValueError(
            f"sigma must be strictly positive (σ > 0); got invalid value(s): {bad}"
        )

    result = np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))
    result = np.clip(result, 0.0, 1.0)  # guard against floating-point overshoot

    if result.ndim == 0:
        return float(result) if scalar_input else result
    return result


# ---------------------------------------------------------------------------
# Convenience: evaluate a set of Gaussian centres/widths at a single input
# ---------------------------------------------------------------------------


def evaluate_membership_set(
    x: float,
    centres: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Evaluate multiple Gaussian membership functions at a single input x.

    Parameters
    ----------
    x : float
        Input value.
    centres : np.ndarray
        Array of centres, shape (K,).
    sigmas : np.ndarray
        Array of widths, shape (K,).  All must be > 0.

    Returns
    -------
    np.ndarray
        Membership values shape (K,) in [0, 1].
    """
    centres = np.asarray(centres, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if centres.shape != sigmas.shape:
        raise ValueError(
            f"centres and sigmas must have the same shape; "
            f"got {centres.shape} vs {sigmas.shape}"
        )
    return gaussian_membership(x, centres, sigmas)


# ---------------------------------------------------------------------------
# Gradient of the membership function (useful for parameter inspection)
# ---------------------------------------------------------------------------


def gaussian_membership_gradient_c(
    x: float,
    c: float,
    sigma: float,
) -> float:
    """Partial derivative of μ with respect to centre c.

    ∂μ/∂c = μ(x; c, σ) × (x - c) / σ²
    """
    mu = gaussian_membership(x, c, sigma)
    return float(mu * (x - c) / (sigma ** 2))


def gaussian_membership_gradient_sigma(
    x: float,
    c: float,
    sigma: float,
) -> float:
    """Partial derivative of μ with respect to width σ.

    ∂μ/∂σ = μ(x; c, σ) × (x - c)² / σ³
    """
    mu = gaussian_membership(x, c, sigma)
    return float(mu * (x - c) ** 2 / (sigma ** 3))

