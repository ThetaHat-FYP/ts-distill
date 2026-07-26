"""
Uncertainty-Sample Initializer
------------------------
Initialises synthetic data by selecting the "hardest" examples from the 
real training data. 

Provides two strategies:
  initialize()          — window-based algorithms.
                          Returns (n_windows, window_size, features).

  initialize_sequence() — continuous-sequence algorithms (MTT).
                          Returns a single (n_synthetic, features) tensor.

Hardness is determined by an optional `scoring_fn` (e.g., a model's loss 
function). If none is provided, it defaults to selecting the sequences with 
the highest temporal volatility (variance).
"""

import torch
from typing import Callable, Optional

# Assuming this is your base class import path
from ts_distill.distillation_core.initializer.base import BaseInitializer

from ts_distill._logging import get_logger

logger = get_logger(__name__)


class UncertaintySampleInitializer(BaseInitializer):
    """
    Initialise synthetic data by selecting the most 'uncertain' or 'hard' data.
    
    This helps the distillation process focus immediately on the most difficult 
    parts of the data manifold.
    """

    def __init__(self, scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        Args:
            scoring_fn: A function that takes a batch of sequences (B, T, C) 
                        and returns a 1D tensor of 'hardness' scores (B,). 
                        Higher score = harder sequence. If None, sequence 
                        volatility (variance) is used.
        """
        super().__init__()
        self.scoring_fn = scoring_fn

    def _default_hardness_score(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Fallback scoring: Calculates the temporal volatility of the sequence.
        Sequences with high step-to-step changes are considered 'harder'.
        """
        # Calculate the absolute difference between consecutive timesteps
        diffs = torch.abs(sequences[:, 1:, :] - sequences[:, :-1, :])
        # Average the differences across time and features for each sequence
        scores = diffs.mean(dim=(1, 2))
        return scores

    def initialize(
        self,
        shape: tuple,
        random_data_reference: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Window-based initialisation for algorithms like FRePO and CondTSF.

        Selects `n_windows` independent windows from a pre-windowed real dataset 
        that have the highest uncertainty/hardness scores.

        Args:
            shape (tuple):                Target shape: (n_windows, window_size, n_features).
            random_data_reference (Tensor): Windowed data to evaluate and sample from,
                                          shape (N, window_size, n_features).

        Returns:
            Tensor of shape `shape` with requires_grad=True.
        """
        if random_data_reference is None:
            logger.info("  Warning: No reference data provided for Uncertainty init. Falling back to noise.")
            data = torch.randn(shape) * 0.1
        else:
            n_samples = shape[0]
            
            # 1. Score all available windows
            if self.scoring_fn:
                scores = self.scoring_fn(random_data_reference)
            else:
                scores = self._default_hardness_score(random_data_reference)
                
            # 2. Find the indices of the 'n_samples' hardest windows
            # topk returns (values, indices)
            _, hardest_indices = torch.topk(scores, k=n_samples)
            
            # 3. Extract and clone
            data = random_data_reference[hardest_indices].clone()

        data.requires_grad_(True)
        return data

    def initialize_sequence(
        self,
        raw_train_data: torch.Tensor,
        n_synthetic: int,
    ) -> torch.Tensor:
        """
        Continuous-sequence initialisation for MTT-style distillation.

        Evaluates all possible sliding windows of length `n_synthetic` and 
        selects the single block with the highest uncertainty/hardness score.

        Args:
            raw_train_data (Tensor): Un-windowed, normalised training data,
                                     shape (T, C).  T must be > n_synthetic.
            n_synthetic (int):       Number of timesteps in the synthetic
                                     sequence (e.g. 384).

        Returns:
            Tensor of shape (n_synthetic, C) with requires_grad=True.

        Raises:
            ValueError: If raw_train_data is too short.
        """
        T, C = raw_train_data.shape
        if T <= n_synthetic:
            raise ValueError(
                f"raw_train_data length ({T}) must be "
                f"greater than n_synthetic ({n_synthetic})."
            )

        # 1. Extract all possible sliding windows efficiently
        # Resulting shape: (Total_Windows, n_synthetic, C)
        windows = raw_train_data.unfold(0, n_synthetic, 1).transpose(1, 2)

        # 2. Score all windows to find the hardest one
        if self.scoring_fn:
            # Note: If memory is an issue with huge datasets, you may need to 
            # batch the scoring_fn evaluation here instead of doing it all at once.
            with torch.no_grad():
                scores = self.scoring_fn(windows)
        else:
            scores = self._default_hardness_score(windows)

        # 3. Find the index of the window with the maximum hardness score
        best_idx = torch.argmax(scores).item()

        # 4. Extract, clone, and prep for the optimiser
        synthetic_seq = raw_train_data[best_idx : best_idx + n_synthetic].clone()
        synthetic_seq.requires_grad_(True)

        logger.info(f"  Uncertainty sequence initialised from real data (Hardest sequence) "
              f"(rows {best_idx}–{best_idx + n_synthetic - 1}), "
              f"shape {tuple(synthetic_seq.shape)}")

        return synthetic_seq