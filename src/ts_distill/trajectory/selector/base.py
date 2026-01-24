from abc import ABC, abstractmethod
from typing import List, Dict

class BaseTrajectorySelector(ABC):
    """Filters a list of trajectories based on a metric."""
    def __init__(self, metric: BaseTrajectoryMetric):
        self.metric = metric

    @abstractmethod
    def select(self, trajectories: List[List[Dict]], k: int) -> List[List[Dict]]:
        """
        Args:
            trajectories: A list of expert paths (each path is a list of weights).
            k: How many to keep.
        Returns:
            The top-k most informative trajectories.
        """
        pass