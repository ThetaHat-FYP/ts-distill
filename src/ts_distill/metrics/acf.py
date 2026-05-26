import numpy as np
from statsmodels.tsa.stattools import acf

from ts_distill.metrics.base import BaseTemporalMetric


class ACFMetric(BaseTemporalMetric):
    """
    Autocorrelation Function (ACF) fidelity metric.

    Measures how well the synthetic sequence reproduces the autocorrelation
    structure of the real data, split into two lag bands:

      Short lags (1 .. period)        — captures seasonal / periodic structure.
      Long  lags (period+1 .. n_lags) — captures slow decay / long-range dependence.

    This split directly operationalises H1:
      "Current methods preserve dominant periodic structures but struggle
       with long-range dependencies."

    Both sequences are truncated to min(T, M) before comparison so that the
    ACF is estimated from the same number of observations.

    Args:
        period (int): Dominant seasonal period of the dataset.
                      Use 24  for ETTh1/ETTh2 (hourly → one day).
                      Use 96  for ETTm1/ETTm2 (15-min → one day).
        n_lags (int): Total number of lags to compute. Must satisfy period < n_lags
                      and n_lags < min(len(real), len(synthetic)). Default: 100.
    """

    def __init__(self, period: int = 24, n_lags: int = 100) -> None:
        if period >= n_lags:
            raise ValueError(
                f"period ({period}) must be strictly less than n_lags ({n_lags})."
            )
        self.period = period
        self.n_lags = n_lags

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel ACF short-lag and long-lag MAE errors.

        Args:
            real      (np.ndarray): Shape (T, C) — full real training sequence.
            synthetic (np.ndarray): Shape (M, C) — distilled synthetic sequence.

        Returns:
            np.ndarray: Shape (C, 2).
                [:, 0] = acf_short — MAE over lags 1 .. period
                [:, 1] = acf_long  — MAE over lags period+1 .. n_lags
        """
        min_len = min(len(real), len(synthetic))

        if self.n_lags >= min_len:
            raise ValueError(
                f"n_lags ({self.n_lags}) must be less than the truncated sequence "
                f"length ({min_len}). Reduce n_lags or increase n_synthetic."
            )

        # Truncate real to match synthetic length for a fair comparison.
        real_t = real[:min_len]
        syn_t  = synthetic[:min_len]

        n_channels = real.shape[1]
        results    = np.zeros((n_channels, 2), dtype=np.float64)

        for c in range(n_channels):
            # acf() returns lags 0..n_lags (length n_lags+1). Lag 0 is always
            # 1.0 for both sequences so we drop it — slice from index 1.
            # nan_to_num guards near-constant channels where std≈0 causes acf()
            # to return NaN (division by zero in the normalisation step).
            acf_real = np.nan_to_num(acf(real_t[:, c], nlags=self.n_lags, fft=True)[1:], nan=0.0)
            acf_syn  = np.nan_to_num(acf(syn_t[:, c],  nlags=self.n_lags, fft=True)[1:], nan=0.0)

            diff = np.abs(acf_real - acf_syn)

            # Short lags: indices 0..period-1 in the sliced array = lags 1..period
            acf_short = float(np.mean(diff[: self.period]))
            # Long lags:  indices period.. in the sliced array   = lags period+1..n_lags
            acf_long  = float(np.mean(diff[self.period :]))

            results[c, 0] = acf_short
            results[c, 1] = acf_long

        return results
