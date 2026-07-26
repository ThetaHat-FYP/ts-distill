"""
Trainer
=======
Standard fit/eval loop. `Trainer.fit` optionally takes a `val_loader` and a
`patience`, in which case it early-stops and restores the best weights seen.
"""

from ts_distill.trainer.trainer.base import BaseTrainer
from ts_distill.trainer.trainer.trainer import Trainer

__all__ = ['BaseTrainer', 'Trainer']
