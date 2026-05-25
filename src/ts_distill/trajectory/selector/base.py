from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseTrajectoryMetric(ABC):
    """
    Scores a single checkpoint or a checkpoint pair for use in selection decisions.
    Subclasses implement domain-specific quality signals (e.g. gradient energy,
    cross-seed variance, per-pair distillation efficiency).
    """

    @abstractmethod
    def score(self, checkpoint: Dict) -> float:
        """Return a scalar quality score for a single checkpoint dict."""
        pass


class BaseTrajectorySelector(ABC):
    """
    Filters or reorders a recorded expert trajectory.

    Two usage patterns are supported:
      1. Multi-trajectory selection — pick the top-k trajectories from a pool
         of expert runs (e.g. select diverse seeds).  Use select().
      2. Intra-trajectory filtering — restrict a single trajectory to a subset
         of its checkpoints (e.g. phase-window selection).  Use filter_trajectory().

    Subclasses must implement at least one of these methods.

    Args:
        metric (BaseTrajectoryMetric | None): Optional quality metric used by
            concrete subclasses that need to score checkpoints.  Pass None for
            selector strategies that don't use an external metric (e.g. epoch-
            range filtering).
    """

    def __init__(self, metric: Optional[BaseTrajectoryMetric] = None):
        self.metric = metric

    @abstractmethod
    def select(self, trajectories: List[List[Dict]], k: int) -> List[List[Dict]]:
        """
        Choose the top-k trajectories from a pool of expert paths.

        Args:
            trajectories: A list of expert paths; each path is a list of
                          checkpoint dicts with keys 'step' and 'weights'.
            k:            Number of trajectories to keep.

        Returns:
            The k most informative trajectory paths.
        """
        pass

    def filter_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Filter a single trajectory to a relevant subset of its checkpoints.

        The default implementation returns the trajectory unchanged.
        Override in subclasses that perform intra-trajectory filtering
        (e.g. phase-window selection, gradient-energy thresholding).

        Args:
            trajectory: Ordered list of checkpoint dicts from one expert run.

        Returns:
            Filtered list of checkpoint dicts.
        """
        return trajectory