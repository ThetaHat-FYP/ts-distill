"""
Cross-channel fidelity — are the relationships BETWEEN channels preserved?

Every other metric in this package scores each channel in isolation. A
multivariate forecaster also exploits how channels move together (in ETT, oil
temperature tracks load), and synthetic data can reproduce each channel's
temporal shape perfectly while destroying those couplings.

Compares Pearson correlation matrices via Frobenius norm, scaled by channel
count so datasets of different width stay comparable.

This metric can DEGRADE as post-fix strength rises: `FFTAmplitudePostFix`
processes each channel independently, so pushing every channel toward the real
spectrum can pull them out of step with each other. Worth reporting as the
honest cost of a per-channel correction.
"""

import numpy as np


class CrossCorrelationMetric:
    """
    Cross-feature correlation preservation metric.

    Measures whether the synthetic sequence preserves the inter-channel
    relationships of the real training data, using the Frobenius norm of the
    difference between the two Pearson correlation matrices.

    Why this matters
    ----------------
    Temporal metrics (ACF, FFT, trend, variance) evaluate each channel in
    isolation.  A multivariate forecasting model exploits correlations between
    channels (e.g. oil temperature correlating with electrical load in ETT
    datasets).  Synthetic data that destroys these relationships degrades the
    model's ability to learn cross-feature patterns, even if per-channel
    temporal structure is perfectly preserved.

    Metric definition
    -----------------
        corr_real = Pearson correlation matrix of real training data   → (C, C)
        corr_syn  = Pearson correlation matrix of synthetic data        → (C, C)
        cross_corr_error = Frobenius(corr_real - corr_syn) / C

    Dividing by C scales the result to be comparable across datasets with
    different numbers of channels (C=7 for ETT, C=21 for weather).

    The FULL sequences are used (no truncation): correlation structure is a
    global property that benefits from all available observations.

    This class does NOT inherit from BaseTemporalMetric because it returns a
    scalar (not per-channel), making the standard (C,) contract inapplicable.
    The MetricAggregator handles it as a special case.

    Near-constant channels (std ≈ 0) produce NaN from np.corrcoef.
    np.nan_to_num replaces those entries with 0.0, treating a collapsed
    channel as uncorrelated with everything else.
    """

    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """
        Compute the Frobenius-norm cross-feature correlation error.

        Args:
            real      (np.ndarray): Shape (T, C) — full real training sequence.
            synthetic (np.ndarray): Shape (M, C) — distilled synthetic sequence.

        Returns:
            float: Frobenius(corr_real - corr_syn) / C.
                   0 = identical correlation structure; larger = worse.
        """
        C = real.shape[1]

        # np.corrcoef expects (C, T) — features as rows, observations as columns.
        corr_real = np.corrcoef(real.T)       # (C, C)
        corr_syn  = np.corrcoef(synthetic.T)  # (C, C)

        # Replace NaN with 0 on near-constant channels (std ≈ 0 after scaling
        # is theoretically impossible after StandardScaler but can occur in
        # pathological synthetic sequences that collapsed to a constant).
        corr_real = np.nan_to_num(corr_real, nan=0.0)
        corr_syn  = np.nan_to_num(corr_syn,  nan=0.0)

        frob = np.linalg.norm(corr_real - corr_syn, 'fro')
        return float(frob / C)
