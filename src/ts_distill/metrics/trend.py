import numpy as np
from statsmodels.tsa.seasonal import STL

from ts_distill.metrics.base import BaseTemporalMetric


class TrendMetric(BaseTemporalMetric):
    """
    Trend fidelity metric using STL decomposition (Seasonal-Trend via LOESS).

    Decomposes both real and synthetic sequences into trend, seasonal, and
    residual components, then measures the per-channel L2 distance between
    the trend components.  A large trend error indicates that the synthetic
    sequence fails to reproduce the slow-moving structure of the real data.

    Both sequences are truncated to min(T, M) before decomposition so that
    the trend components have the same length and can be directly compared.

    STL safety requirement: sequence length >= 2 * period.
    With the default n_synthetic=384 and max period=96: 384 >= 192. Safe.

    Args:
        period (int): Seasonal period for STL decomposition.
                      Use 24  for ETTh1/ETTh2 (hourly → one day).
                      Use 96  for ETTm1/ETTm2 (15-min → one day).
    """

    def __init__(self, period: int = 24) -> None:
        self.period = period

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel STL trend L2 distance (normalised by N).

        Args:
            real      (np.ndarray): Shape (T, C).
            synthetic (np.ndarray): Shape (M, C).

        Returns:
            np.ndarray: Shape (C,) — per-channel trend error / N.
                        Smaller values indicate better trend preservation.

        Raises:
            ValueError: If the truncated sequence length is less than 2 * period,
                        making STL decomposition unreliable.
        """
        N = min(len(real), len(synthetic))

        if N < 2 * self.period:
            raise ValueError(
                f"Truncated sequence length ({N}) is less than 2 * period "
                f"({2 * self.period}). STL requires at least two full seasonal "
                f"cycles. Increase n_synthetic or reduce period."
            )

        real_t = real[:N]

        n_channels = real.shape[1]
        results    = np.zeros(n_channels, dtype=np.float64)

        for c in range(n_channels):
            # robust=True uses iteratively reweighted least squares, making
            # the decomposition less sensitive to outliers in normalised data.
            trend_real = STL(real_t[:, c],       period=self.period, robust=True).fit().trend
            trend_syn  = STL(synthetic[:N, c],   period=self.period, robust=True).fit().trend

            results[c] = np.linalg.norm(trend_real - trend_syn) / N

        return results
