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
    def on_train_begin(self, model, **kwargs): pass

    @abstractmethod
    def on_epoch_end(self, model, epoch, loss, **kwargs): pass

    @abstractmethod
    def on_train_end(self, model, **kwargs): pass