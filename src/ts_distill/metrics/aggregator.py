"""
Metric orchestration — runs every fidelity metric and builds the result rows.

Single entry point for scoring one distillation run. Calls all six metric
classes and returns two things at once:

  feature_rows   one row per CHANNEL, for per-channel analysis
  main_row       one row per RUN, channels averaged, for the results table

Both carry the same columns in a fixed order (MAIN_COLUMNS / FEATURE_COLUMNS)
so results appended across many runs always line up into one CSV.

Utility numbers (real_mse, transfer_mse) are passed IN rather than computed
here — this class only measures fidelity; `ts_distill.evaluation` measures
utility. Keeping both in the same row is what makes the utility-vs-fidelity
trade-off visible in a single table.
"""

from typing import Dict, List, Tuple

import numpy as np

from ts_distill.metrics.acf import ACFMetric
from ts_distill.metrics.frequency import FrequencyMetric
from ts_distill.metrics.trend import TrendMetric
from ts_distill.metrics.variance import VarianceMetric
from ts_distill.metrics.periodicity import PeriodicityMetric
from ts_distill.metrics.cross_correlation import CrossCorrelationMetric


class MetricAggregator:
    """
    Orchestrates all four temporal metrics and assembles the result rows for
    both the feature-wise table and the aggregate main table.

    Feature-wise table  — one row per channel per experiment.
    Main table          — one row per experiment (metrics averaged across channels).

    Usage
    -----
    aggregator = MetricAggregator(period=24, n_lags=100)

    feature_rows, main_row = aggregator.compute_all(
        real                = real_np,          # shape (T, C)
        synthetic           = synthetic_np,     # shape (M, C)
        real_mse            = 0.0123,
        transfer_mse        = 0.0456,
        dataset             = 'ETTh1',
        model               = 'DLinear',
        distillation_method = 'MTT',
    )

    Args:
        period (int):  Seasonal period for ACFMetric and TrendMetric.
                       24 for ETTh datasets, 96 for ETTm datasets.
        n_lags (int):  Total ACF lags to compute (must be > period). Default: 100.
        eps    (float): Denominator guard for VarianceMetric. Default: 1e-10.
    """

    # Exact column order for both output tables.
    # The feature-wise table inserts 'feature' between 'model' and 'distillation_method'.
    MAIN_COLUMNS: List[str] = [
        'dataset', 'model', 'distillation_method', 'n_distill_steps',
        'real_mse', 'transfer_mse',
        'acf_short', 'acf_long', 'fft_distance', 'trend_error', 'variance_diff',
        'freq_rank_error', 'peak_mag_ratio', 'cross_corr_error',
    ]
    FEATURE_COLUMNS: List[str] = [
        'dataset', 'feature', 'model', 'distillation_method', 'n_distill_steps',
        'real_mse', 'transfer_mse',
        'acf_short', 'acf_long', 'fft_distance', 'trend_error', 'variance_diff',
        'freq_rank_error', 'peak_mag_ratio', 'cross_corr_error',
    ]

    def __init__(
        self,
        period: int   = 24,
        n_lags: int   = 100,
        eps:    float = 1e-10,
    ) -> None:
        self._acf          = ACFMetric(period=period, n_lags=n_lags)
        self._freq         = FrequencyMetric()
        self._trend        = TrendMetric(period=period)
        self._variance     = VarianceMetric(eps=eps)
        self._periodicity  = PeriodicityMetric(eps=eps)
        self._cross_corr   = CrossCorrelationMetric()

    def compute_all(
        self,
        real:                np.ndarray,
        synthetic:           np.ndarray,
        real_mse:            float,
        transfer_mse:        float,
        dataset:             str,
        model:               str,
        distillation_method: str,
        n_distill_steps:     int = 0,
    ) -> Tuple[List[Dict], Dict]:
        """
        Run all four metric classes and assemble both table outputs.

        Args:
            real                (np.ndarray): Shape (T, C) normalised real train sequence.
            synthetic           (np.ndarray): Shape (M, C) distilled synthetic sequence.
            real_mse            (float):      Test MSE of model trained on real data.
            transfer_mse        (float):      Test MSE of model trained on synthetic data.
            dataset             (str):        Dataset name, e.g. 'ETTh1'.
            model               (str):        Model name,   e.g. 'DLinear'.
            distillation_method (str):        Algorithm,    e.g. 'MTT'.

        Returns:
            Tuple[List[Dict], Dict]:
                feature_rows — list of C dicts, one per channel, keyed by FEATURE_COLUMNS.
                main_row     — single dict keyed by MAIN_COLUMNS (channels averaged).
        """
        # ── Run all six metrics ───────────────────────────────────────────────
        # acf_results   : (C, 2) — col 0 = acf_short, col 1 = acf_long
        # fft_results   : (C,)
        # trend_results : (C,)
        # var_results   : (C,)
        # period_results: (C, 2) — col 0 = freq_rank_error, col 1 = peak_mag_ratio
        # cross_corr    : float  — single scalar (not per-channel)
        acf_results    = self._acf.compute(real, synthetic)
        fft_results    = self._freq.compute(real, synthetic)
        trend_results  = self._trend.compute(real, synthetic)
        var_results    = self._variance.compute(real, synthetic)
        period_results = self._periodicity.compute(real, synthetic)
        cross_corr_val = self._cross_corr.compute(real, synthetic)

        n_channels = real.shape[1]

        # ── Build feature-wise rows (one dict per channel) ────────────────────
        feature_rows: List[Dict] = []
        for c in range(n_channels):
            row: Dict = {
                'dataset':             dataset,
                'feature':             c,
                'model':               model,
                'distillation_method': distillation_method,
                'n_distill_steps':     n_distill_steps,
                'real_mse':            real_mse,
                'transfer_mse':        transfer_mse,
                'acf_short':           float(acf_results[c, 0]),
                'acf_long':            float(acf_results[c, 1]),
                'fft_distance':        float(fft_results[c]),
                'trend_error':         float(trend_results[c]),
                'variance_diff':       float(var_results[c]),
                'freq_rank_error':     float(period_results[c, 0]),
                'peak_mag_ratio':      float(period_results[c, 1]),
                'cross_corr_error':    cross_corr_val,   # same scalar for all channels
            }
            feature_rows.append(row)

        # ── Build main row (channels averaged) ────────────────────────────────
        main_row: Dict = {
            'dataset':             dataset,
            'model':               model,
            'distillation_method': distillation_method,
            'n_distill_steps':     n_distill_steps,
            'real_mse':            real_mse,
            'transfer_mse':        transfer_mse,
            'acf_short':           float(np.mean(acf_results[:, 0])),
            'acf_long':            float(np.mean(acf_results[:, 1])),
            'fft_distance':        float(np.mean(fft_results)),
            'trend_error':         float(np.mean(trend_results)),
            'variance_diff':       float(np.mean(var_results)),
            'freq_rank_error':     float(np.mean(period_results[:, 0])),
            'peak_mag_ratio':      float(np.mean(period_results[:, 1])),
            'cross_corr_error':    cross_corr_val,
        }

        return feature_rows, main_row
