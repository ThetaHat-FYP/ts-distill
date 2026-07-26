"""
Distillation core
=================
The distillation algorithms and the synthetic-data initializers they start from.

Sub-packages
------------
distillation_algorithm : MTT and its variants (the outer-loop optimisers).
initializer            : Strategies for choosing the starting synthetic sequence.
"""

from ts_distill.distillation_core.distillation_algorithm import (
    BaseDistiller,
    MTTDistiller,
    PhaseAwareMTTDistiller,
)
from ts_distill.distillation_core.initializer import (
    BaseInitializer,
    RandomSampleInitializer,
    GeometrySequenceInitializer,
    UncertaintySampleInitializer,
)

__all__ = [
    'BaseDistiller',
    'MTTDistiller',
    'PhaseAwareMTTDistiller',
    'BaseInitializer',
    'RandomSampleInitializer',
    'GeometrySequenceInitializer',
    'UncertaintySampleInitializer',
]
