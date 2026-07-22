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

    Whole-dataset target
    ---------------------
    The synthetic sequence (length M) is compared against the *average*
    magnitude spectrum of the real sequence, computed over every
    non-overlapping length-M segment of the full real training data.

    This matters: the real sequence has T rows (e.g. ~34 000) while synthetic
    has only M (e.g. 384).  Comparing synthetic against just the first M real
    rows would score it against one arbitrary window.  Averaging the spectrum
    across all T // M segments gives a stable target that represents the whole
    dataset's seasonal energy, and matches the target used by the FFT
    amplitude post-fix so the two are directly comparable.
    """

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute per-channel FFT magnitude L2 distance (normalised by M).

        The real spectrum is the mean magnitude spectrum over all
        non-overlapping length-M segments of the real sequence.

        Args:
            real      (np.ndarray): Shape (T, C) — full real training sequence.
            synthetic (np.ndarray): Shape (M, C) — distilled synthetic sequence.

        Returns:
            np.ndarray: Shape (C,) — per-channel FFT distance / M.
                        Smaller values indicate better frequency preservation.
        """
        M          = len(synthetic)
        n_channels = real.shape[1]
        n_segs     = len(real) // M

        # ── Mean real magnitude spectrum over all length-M segments ──────────
        # rfft output length is M // 2 + 1 — identical for real segments and
        # synthetic because both use window length M.
        if n_segs == 0:
            # Real sequence shorter than synthetic — fall back to what exists.
            mag_real = np.abs(np.fft.rfft(real[:M], axis=0))
        else:
            seg_mags = np.zeros((M // 2 + 1, n_channels), dtype=np.float64)
            for i in range(n_segs):
                seg = real[i * M : (i + 1) * M]
                seg_mags += np.abs(np.fft.rfft(seg, axis=0))
            mag_real = seg_mags / n_segs

        # ── Per-channel L2 distance to the synthetic spectrum ────────────────
        results = np.zeros(n_channels, dtype=np.float64)
        for c in range(n_channels):
            mag_syn    = np.abs(np.fft.rfft(synthetic[:, c]))
            results[c] = np.linalg.norm(mag_real[:, c] - mag_syn) / M

        return results
