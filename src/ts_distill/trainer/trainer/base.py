"""
Trainer interface — one training loop for experts, probes, and baselines.

Every model in the pipeline is trained through this contract, so the expert
whose trajectory is recorded and the probe that scores synthetic data follow an
identical procedure. That is what makes real_mse and transfer_mse comparable:
the data differs, the training does not.

Note the INNER loop of MTT deliberately does NOT use this interface. It needs
differentiable, in-graph updates so gradients can flow back into the synthetic
data, which a standard optimiser step would sever.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch.nn as nn
import torch.optim as optim
from ts_distill.trainer.callback.base import BaseCallback

class BaseTrainer(ABC):
    """
    Standard interface for training ANY model on ANY dataset.
    """
    
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, device: str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    @abstractmethod
    def train_epoch(self, dataloader) -> float:
        """
        Runs a single epoch of training.
        Returns: Average loss for the epoch.
        """
        pass

    @abstractmethod
    def fit(self, dataloader, epochs: int, callbacks: List[BaseCallback] = None):
        """
        The main training loop.
        
        Args:
            dataloader: Iterator for data batches.
            epochs: Number of passes.
            callbacks: List of objects (like TrajectoryRecorder) to call after steps.
        """
        pass