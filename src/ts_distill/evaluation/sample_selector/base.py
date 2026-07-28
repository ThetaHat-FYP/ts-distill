"""
Anchor selector interface — which REAL windows get mixed into synthetic data.

Hybrid mixing adds a fraction of real data back to the distilled set. This
interface decides WHICH real windows to add; the fraction itself is chosen
elsewhere (`predict_r_star` or `find_optimal_ratio`).

Concrete strategies live in `ts_distill.evaluation.hybrid_evaluation.hybrid`:
random, uniform stride, importance-weighted by expert loss, and diversity-based.
"""

from abc import ABC, abstractmethod
import torch

class BaseAnchorSelector(ABC):
    """
    Interface for selecting a subset of 'informative' samples (Anchor Points)
    from the real dataset to mix with synthetic data.
    """

    @abstractmethod
    def select_indices(self, real_data: torch.Tensor, n_samples: int, model=None) -> torch.Tensor:
        """
        Logic to pick the best real data points.
        
        Args:
            real_data: The full real training dataset.
            n_samples: How many anchor points to select.
            model: Optional. Some selectors (like 'HardSampleMining') need a model 
                   to see which points are hard to learn.
                   
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        pass