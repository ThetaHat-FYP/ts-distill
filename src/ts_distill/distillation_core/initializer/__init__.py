"""
Initializers
============
Strategies for choosing the synthetic sequence that distillation starts from.

All initializers expose the same two entry points:

  initialize(shape, random_data_reference)  -> (n_windows, window_size, features)
      For window-based algorithms (FRePO, CondTSF).

  initialize_sequence(raw_train_data, n_synthetic) -> (n_synthetic, features)
      For continuous-sequence algorithms (MTT). Each strategy picks ONE
      contiguous block of `n_synthetic` timesteps; they differ only in the rule
      used to choose that block:

        RandomSampleInitializer      : uniformly random start index.
        GeometrySequenceInitializer  : the medoid block (closest to the dataset
                                       centroid) — deterministic, low variance.
        UncertaintySampleInitializer : the highest-volatility ("hardest") block.

Because the choice of block only sets the starting point of a non-convex
optimisation, strategies are best compared on seed-to-seed variance and on
downstream hybrid-mixing gain, not on a single-seed transfer MSE.
"""

from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.distillation_core.initializer.random_sample_initializer import (
    RandomSampleInitializer,
)
from ts_distill.distillation_core.initializer.geommetry_sequence_initializer import (
    GeometrySequenceInitializer,
)
from ts_distill.distillation_core.initializer.uncertainty_sequence_initializer import (
    UncertaintySampleInitializer,
)

__all__ = [
    'BaseInitializer',
    'RandomSampleInitializer',
    'GeometrySequenceInitializer',
    'UncertaintySampleInitializer',
]
