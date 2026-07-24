from abc import ABC, abstractmethod
from typing import Dict, List


class BasePhaseDetector(ABC):
    """Interface for detecting the expert-trajectory phase boundary T+."""

    @abstractmethod
    def detect(self, val_losses: List[float]) -> Dict:
        """
        Args:
            val_losses: Per-epoch validation loss recorded during expert
                training (index i = epoch i+1).

        Returns:
            dict with at least a 'boundary_epoch' key (1-indexed epoch, or
            None if no boundary was found).
        """
        pass
