"""
Training
========
The training loop used for every model in the pipeline: the expert whose
trajectory is recorded, and the probes trained on distilled or hybrid data.

Sub-packages
------------
trainer  : Trainer — fit/eval loop with optional early stopping.
callback : Hooks fired during training (trajectory recording, val-loss curves).
"""

from ts_distill.trainer.trainer import BaseTrainer, Trainer
from ts_distill.trainer.callback import (
    BaseCallback,
    SimpleCallback,
    ValLossRecorderCallback,
)

__all__ = [
    'BaseTrainer',
    'Trainer',
    'BaseCallback',
    'SimpleCallback',
    'ValLossRecorderCallback',
]
