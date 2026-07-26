"""
Distillation algorithms
=======================
Outer-loop optimisers that turn a real training set into a small synthetic one.

Stable
------
MTTDistiller           : Matching Training Trajectories — parameter matching
                         against a recorded expert trajectory.
PhaseAwareMTTDistiller : MTT that switches from parameter matching to prediction
                         matching at a detected phase boundary T+.

Experimental
------------
CondTSFDistiller, FRePODistiller, PredictiveMTTDistiller are implemented but not
covered by the reference experiments. Import them directly from their modules,
e.g. ``from ts_distill.distillation_core.distillation_algorithm.frepo import
FRePODistiller``. They are intentionally not re-exported here.
"""

from ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.distillation_algorithm.phase_aware_mtt import (
    PhaseAwareMTTDistiller,
)

__all__ = [
    'BaseDistiller',
    'MTTDistiller',
    'PhaseAwareMTTDistiller',
]
