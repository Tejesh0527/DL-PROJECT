"""
scene_analysis/condition_smoother.py
======================================
Causal exponential temporal smoother for scene-condition vectors.

Implements Equation (2) from the RATFA paper:
  ē_t = λ_e * ē_{t-1} + (1 - λ_e) * e_t

Causality guarantee: only the current and previous smoothed vectors are used.
Future frames are NEVER accessed.

Effect of λ_e
-------------
  λ_e → 0  : no smoothing; ē_t = e_t (immediate response to new estimates)
  λ_e → 1  : heavy smoothing; very slow response (ē_t barely changes)
  λ_e = 0.7: recommended default; retains ~70 % of previous history,
              30 % weight on new observation; suitable for typical frame rates.

Practical guidance
------------------
Choose λ_e on a validation sequence by measuring condition-transition lag vs.
noise suppression.  It must be fixed before the held-out test evaluation.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

NUM_CONDITIONS = 4   # [clear, rain, fog, dark]


class ConditionSmoother:
    """Causal exponential smoother for scene-condition vectors (Eq. 2).

    Parameters
    ----------
    lambda_e : float
        Smoothing coefficient in [0, 1].  Higher ⟹ more smoothing.
    num_conditions : int
        Length of the condition vector (default 4 for RATFA).
    """

    def __init__(
        self,
        lambda_e: float = 0.7,
        num_conditions: int = NUM_CONDITIONS,
    ):
        self._validate_lambda(lambda_e)
        self.lambda_e = float(lambda_e)
        self.num_conditions = int(num_conditions)
        self._smoothed: Optional[np.ndarray] = None   # ē_{t-1}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_lambda(value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"lambda_e must be in [0, 1]; got {value}"
            )

    def _validate_vector(self, vec: np.ndarray) -> None:
        if vec.ndim != 1 or len(vec) != self.num_conditions:
            raise ValueError(
                f"Input vector must have shape ({self.num_conditions},); "
                f"got {vec.shape}"
            )
        if np.any(vec < 0):
            raise ValueError(
                "Condition vector must have non-negative entries; "
                f"got {vec}"
            )

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, e_t: np.ndarray) -> np.ndarray:
        """Apply one step of causal exponential smoothing.

        Implements Eq. (2):  ē_t = λ_e * ē_{t-1} + (1 - λ_e) * e_t

        Parameters
        ----------
        e_t : np.ndarray
            Current scene-condition vector of shape (num_conditions,).
            Values must be non-negative (they represent membership degrees).

        Returns
        -------
        np.ndarray
            Smoothed condition vector ē_t of shape (num_conditions,).
        """
        e_t = np.asarray(e_t, dtype=np.float64)
        self._validate_vector(e_t)

        if self._smoothed is None:
            # First frame: initialise with the current observation
            self._smoothed = e_t.copy()
        else:
            # Eq. (2): causal exponential smoothing
            self._smoothed = (
                self.lambda_e * self._smoothed + (1.0 - self.lambda_e) * e_t
            )

        # Re-normalise to maintain a valid distribution
        total = self._smoothed.sum()
        if total > 1e-9:
            self._smoothed = self._smoothed / total
        else:
            self._smoothed = np.ones(self.num_conditions) / self.num_conditions

        return self._smoothed.copy()

    def update_from_dict(self, condition_dict: dict) -> np.ndarray:
        """Convenience wrapper accepting a dict with keys [clear, rain, fog, dark].

        Parameters
        ----------
        condition_dict : dict
            Must contain keys "clear", "rain", "fog", "dark".

        Returns
        -------
        np.ndarray
            Smoothed condition vector.
        """
        try:
            vec = np.array(
                [
                    condition_dict["clear"],
                    condition_dict["rain"],
                    condition_dict["fog"],
                    condition_dict["dark"],
                ],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise KeyError(
                f"condition_dict must contain 'clear', 'rain', 'fog', 'dark'; "
                f"missing key: {exc}"
            )
        return self.update(vec)

    def smoothed_vector(self) -> Optional[np.ndarray]:
        """Return the current smoothed vector ē_t, or None if not yet updated."""
        return self._smoothed.copy() if self._smoothed is not None else None

    def smoothed_dict(self) -> Optional[dict]:
        """Return the smoothed vector as a dict, or None if not yet updated."""
        if self._smoothed is None:
            return None
        return {
            "clear": float(self._smoothed[0]),
            "rain": float(self._smoothed[1]),
            "fog": float(self._smoothed[2]),
            "dark": float(self._smoothed[3]),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset smoothing state.

        Call this whenever a new video begins to avoid carrying over state
        from a previous sequence.
        """
        self._smoothed = None
        logger.debug("ConditionSmoother reset.")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_sequence(
        self, vectors: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Process a sequence of condition vectors in causal order.

        Parameters
        ----------
        vectors : list of np.ndarray
            Condition vectors for frames 0, 1, ..., T-1.

        Returns
        -------
        list of np.ndarray
            Smoothed vectors for each frame.
        """
        self.reset()
        return [self.update(v) for v in vectors]

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_lambda(self, lambda_e: float) -> None:
        """Update the smoothing coefficient (useful for tuning)."""
        self._validate_lambda(lambda_e)
        self.lambda_e = float(lambda_e)
        logger.info("ConditionSmoother lambda_e updated to %.4f", self.lambda_e)

    def __repr__(self) -> str:
        state = (
            f"smoothed={np.round(self._smoothed, 4)}"
            if self._smoothed is not None
            else "not initialised"
        )
        return (
            f"ConditionSmoother(lambda_e={self.lambda_e}, "
            f"num_conditions={self.num_conditions}, {state})"
        )

