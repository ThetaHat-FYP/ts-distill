import numpy as np

from ts_distill.metrics.base import BaseTemporalMetric


class PeriodicityMetric(BaseTemporalMetric):
    """
    Periodicity preservation metric.

    Checks whether the synthetic sequence preserves the DOMINANT frequency of
    the real sequence — not just the overall spectral shape (that is fft_distance).

    Per channel two values are returned:

      freq_rank_error  — absolute distance (in frequency bins) between the
                         dominant frequency index of the real signal and the
                         dominant frequency index of the synthetic signal.
                         0 = same dominant period preserved exactly.

      peak_mag_ratio   — relative error in the magnitude of the real dominant
                         frequency when evaluated in the synthetic spectrum:
                         |mag_real[peak] - mag_syn[peak]| / mag_real[peak].
                         0 = synthetic reproduces the same energy at the dominant
                         frequency; 1 = dominant frequency completely absent.

    DC component (index 0) is excluded from the dominant-frequency search
    because it reflects the mean, not a periodic structure.

    Both sequences are truncated to min(T, M) for a fair comparison.

    Args:
        eps (float): Guard added to the real peak magnitude to prevent
                     division by zero on near-constant channels. Default: 1e-10.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel periodicity preservation errors.

        Args:
            real      (np.ndarray): Shape (T, C).
            synthetic (np.ndarray): Shape (M, C).

        Returns:
            np.ndarray: Shape (C, 2).
                [:, 0] = freq_rank_error — dominant frequency bin distance (int).
                [:, 1] = peak_mag_ratio  — relative magnitude error at real's peak.
        """
        N = min(len(real), len(synthetic))
        real_t = real[:N]
        syn_t  = synthetic[:N]

        n_channels = real.shape[1]
        results    = np.zeros((n_channels, 2), dtype=np.float64)

        for c in range(n_channels):
            mag_real = np.abs(np.fft.rfft(real_t[:, c]))
            mag_syn  = np.abs(np.fft.rfft(syn_t[:, c]))

            # Skip DC (index 0) — it encodes the mean, not a periodic structure.
            # argmax on [1:] gives 0-based index into the non-DC part; +1 restores
            # the original rfft index.
            peak_real = int(np.argmax(mag_real[1:])) + 1
            peak_syn  = int(np.argmax(mag_syn[1:]))  + 1

            freq_rank_error = abs(peak_real - peak_syn)

            # Evaluate the synthetic magnitude at THE REAL dominant frequency
            # to measure how much of the real periodic energy survived.
            mag_at_real_peak_syn = mag_syn[peak_real]
            peak_mag_ratio = abs(mag_real[peak_real] - mag_at_real_peak_syn) / (
                mag_real[peak_real] + self.eps
            )

            results[c, 0] = float(freq_rank_error)
            results[c, 1] = float(peak_mag_ratio)

        return results
