import numpy as np

from ts_distill.metrics.base import BaseTemporalMetric


class FrequencyMetric(BaseTemporalMetric):
    """
    Frequency-domain fidelity metric using FFT magnitude spectrum distance.

    Computes the per-channel L2 distance between the magnitude spectra of the
    real and synthetic sequences, normalised by sequence length so that the
    result is comparable across datasets with different sequence lengths.

    Only the magnitude spectrum (not phase) is compared because phase alignment
    between independently-generated sequences is arbitrary and not meaningful
    for characterising frequency content preservation.

    Both sequences are truncated to min(T, M) before comparison for a fair
    evaluation on the same number of observations.
    """

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel FFT magnitude L2 distance (normalised by N).

        Args:
            real      (np.ndarray): Shape (T, C).
            synthetic (np.ndarray): Shape (M, C).

        Returns:
            np.ndarray: Shape (C,) — per-channel FFT distance / N.
                        Smaller values indicate better frequency preservation.
        """
        N = min(len(real), len(synthetic))

        real_t = real[:N]
        syn_t  = synthetic[:N]

        n_channels = real.shape[1]
        results    = np.zeros(n_channels, dtype=np.float64)

        for c in range(n_channels):
            # rfft is used (real-valued FFT) for efficiency.
            # Output length is N // 2 + 1 — identical for both sequences
            # because N is the same.
            mag_real = np.abs(np.fft.rfft(real_t[:, c]))
            mag_syn  = np.abs(np.fft.rfft(syn_t[:, c]))

            results[c] = np.linalg.norm(mag_real - mag_syn) / N

        return results
