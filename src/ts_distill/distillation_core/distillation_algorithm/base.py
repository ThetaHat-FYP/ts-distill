from abc import ABC, abstractmethod

from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from ts_distill.distillation_core.initializer.base import BaseInitializer

class BaseDistiller(ABC):
    """
    The Main Controller.
    Subclasses: MTTDistiller, DistributionDistiller.
    """
    
    def __init__(self, initializer: BaseInitializer, matcher: BaseTrajectoryMatcher):
        self.initializer = initializer
        self.matcher = matcher

    @abstractmethod
    def distill(self, source_data, n_steps: int):
        """
        Executes the full distillation loop.
        Returns: The final Synthetic Dataset.
        """
        pass