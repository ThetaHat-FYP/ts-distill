"""
Callback interface — hooks fired at training boundaries.

Lets code observe a training run without the trainer knowing what it is.
`SimpleRecorder` uses it to capture the expert trajectory MTT matches against;
`ValLossRecorderCallback` uses it to build the validation curve the phase
detector reads.

RNG caution: a callback that iterates a DataLoader consumes torch RNG on every
epoch, shifting all later random draws and changing results elsewhere. If that
matters, snapshot and restore around the work with `torch.get_rng_state()` /
`torch.set_rng_state()`.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch.nn as nn
import torch.optim as optim

class BaseCallback(ABC):
    """
    Interface for objects that need to intervene during training.
    Examples: TrajectoryRecorder, EarlyStopping, TensorBoardLogger.
    """
    @abstractmethod
    def on_train_begin(self, model, **kwargs):
        """Fired once before the first epoch. Reset any per-run state here."""
        pass

    @abstractmethod
    def on_epoch_end(self, model, epoch, loss, **kwargs):
        """Fired after every epoch. Where trajectories and curves get recorded."""
        pass

    @abstractmethod
    def on_train_end(self, model, **kwargs):
        """Fired once after the last epoch (or after early stopping)."""
        pass