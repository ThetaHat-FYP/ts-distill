"""
Variance fidelity — is the synthetic data as spread out as the real data?

Reports the relative difference in per-channel variance, scale-free so channels
and datasets can be compared directly.

Like trend, this cannot be fixed by a partial spectral blend. Parseval's
theorem ties total variance to total spectral energy, so blending amplitudes
part-way toward the real spectrum lands variance part-way too, never exactly
right. Expect `variance_diff` to improve only at alpha = 1.

Uses the FULL real sequence rather than truncating to the synthetic length:
variance is a global property, so more observations give a better estimate.
"""

import numpy as np

from ts_distill.metrics.base import BaseTemporalMetric


class VarianceMetric(BaseTemporalMetric):
    """
    Relative variance preservation metric.

    Computes the per-channel relative absolute difference between the variance
    of the real training sequence and the variance of the synthetic sequence:

        variance_diff[c] = |Var(real[:, c]) - Var(synthetic[:, c])| / Var(real[:, c])

    The result is scale-free (bounded [0, ∞)) and comparable across channels
    and datasets regardless of the original signal amplitude.

    Unlike other metrics in this module, the FULL real training sequence is
    used (not truncated to synthetic length) because variance is a global
    property that benefits from as many observations as possible.

    Args:
        eps (float): Small constant added to the real variance denominator to
                     prevent division by zero on near-constant channels.
                     Default: 1e-10. (Near-zero variance cannot occur after
                     StandardScaler unless a channel is exactly constant, but
                     this guard makes the class robust to edge cases.)
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel relative variance difference.

        Args:
            real      (np.ndarray): Shape (T, C) — full real training sequence.
            synthetic (np.ndarray): Shape (M, C) — distilled synthetic sequence.

        Returns:
            np.ndarray: Shape (C,) — per-channel |Var(real) - Var(syn)| / Var(real).
                        0 means perfect variance preservation; larger = worse.
        """
        # Population variance (ddof=0) — both arrays are treated as populations,
        # not samples drawn from a larger distribution.
        var_real = np.var(real,      axis=0)   # shape (C,)
        var_syn  = np.var(synthetic, axis=0)   # shape (C,)

        return np.abs(var_real - var_syn) / (var_real + self.eps)
