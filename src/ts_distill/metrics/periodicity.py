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

    Whole-dataset target
    --------------------
    The synthetic sequence (length M) is compared against the *average*
    magnitude spectrum of the real sequence, computed over every
    non-overlapping length-M segment of the full real training data. Comparing
    against just the first M real rows would pick one arbitrary window's
    dominant frequency; averaging captures the dataset's true dominant period.

    Args:
        eps (float): Guard added to the real peak magnitude to prevent
                     division by zero on near-constant channels. Default: 1e-10.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel periodicity preservation errors.

        The real spectrum is the mean magnitude spectrum over all
        non-overlapping length-M segments of the real sequence.

        Args:
            real      (np.ndarray): Shape (T, C) — full real training sequence.
            synthetic (np.ndarray): Shape (M, C) — distilled synthetic sequence.

        Returns:
            np.ndarray: Shape (C, 2).
                [:, 0] = freq_rank_error — dominant frequency bin distance (int).
                [:, 1] = peak_mag_ratio  — relative magnitude error at real's peak.
        """
        M          = len(synthetic)
        n_channels = real.shape[1]
        n_segs     = len(real) // M

        # ── Mean real magnitude spectrum over all length-M segments ──────────
        if n_segs == 0:
            mag_real_all = np.abs(np.fft.rfft(real[:M], axis=0))
        else:
            seg_mags = np.zeros((M // 2 + 1, n_channels), dtype=np.float64)
            for i in range(n_segs):
                seg = real[i * M : (i + 1) * M]
                seg_mags += np.abs(np.fft.rfft(seg, axis=0))
            mag_real_all = seg_mags / n_segs

        results = np.zeros((n_channels, 2), dtype=np.float64)

        for c in range(n_channels):
            mag_real = mag_real_all[:, c]
            mag_syn  = np.abs(np.fft.rfft(synthetic[:, c]))

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
