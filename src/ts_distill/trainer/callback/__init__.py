"""
Training callbacks
==================
Hooks fired at train/epoch boundaries.

SimpleCallback          : Progress logging.
ValLossRecorderCallback : Records the per-epoch validation loss curve, which the
                          phase detector consumes to locate the boundary T+.

Note: a callback that iterates a DataLoader consumes torch RNG. When that would
change downstream sampling, snapshot and restore the RNG state around it with
``torch.get_rng_state()`` / ``torch.set_rng_state()``.
"""

from ts_distill.trainer.callback.base import BaseCallback
from ts_distill.trainer.callback.simple_callback import SimpleCallback
from ts_distill.trainer.callback.val_loss_callback import ValLossRecorderCallback

__all__ = [
    'BaseCallback',
    'SimpleCallback',
    'ValLossRecorderCallback',
]
