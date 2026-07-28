"""
Temporal fidelity metrics — is the synthetic data still a time series?

Distillation optimises synthetic data for TRAINING UTILITY and damages temporal
structure as a side effect. These metrics quantify that damage, and are what
make the damage (and any repair) measurable rather than anecdotal.

Six metrics, three groups by what the FFT post-fix can do to them:

  REPAIRABLE   FrequencyMetric    spectrum distance  (post-fix target)
               ACFMetric          autocorrelation    (improves for free)
               PeriodicityMetric  dominant cycle

  NOT REPAIRABLE by a partial blend
               TrendMetric        slow drift  (lives in DC / lowest bins)
               VarianceMetric     spread      (Parseval ties it to total energy)

  CAN DEGRADE
               CrossCorrelationMetric  between-channel structure, because the
               post-fix corrects each channel independently

Use `MetricAggregator` rather than the individual classes — it runs all six and
returns both the per-channel and the channel-averaged rows.
"""

from .base import BaseTemporalMetric
from .acf import ACFMetric
from .frequency import FrequencyMetric
from .trend import TrendMetric
from .variance import VarianceMetric
from .periodicity import PeriodicityMetric
from .cross_correlation import CrossCorrelationMetric
from .aggregator import MetricAggregator

__all__ = [
    'BaseTemporalMetric',
    'ACFMetric',
    'FrequencyMetric',
    'TrendMetric',
    'VarianceMetric',
    'PeriodicityMetric',
    'CrossCorrelationMetric',
    'MetricAggregator',
]
