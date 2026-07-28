"""
Geometry initializer — starts distillation from the most TYPICAL window.

Scans every candidate block of `n_synthetic` timesteps and picks the medoid:
the block closest to the dataset's average behaviour. Deterministic, so the
same data always yields the same starting point.

Trade-off against the alternatives. The medoid is the FLATTEST, least eventful
stretch of the series, so it carries little information for a probe to learn
from — single-seed transfer MSE is often worse than a random block. What it
buys is zero seed-to-seed variance, and synthetic data that is complementary
rather than redundant to real data, which shows up as a larger hybrid-mixing
gain. Compare initializers on variance and post-mixing MSE, not on one seed.

Note the file name is misspelled (geommetry); the class name is correct.
"""

import torch
from ts_distill.distillation_core.initializer.base import BaseInitializer

from ts_distill._logging import get_logger

logger = get_logger(__name__)

class GeometrySequenceInitializer(BaseInitializer):
    """Picks the medoid block — the stretch closest to the dataset average."""

    def initialize(self, shape: tuple, real_data_reference: torch.Tensor = None) -> torch.Tensor:
        """Not supported — this strategy is continuous-sequence only (MTT)."""
        raise NotImplementedError("GeometrySequenceInitializer only supports continuous sequence initialization.")

    def initialize_sequence(
        self,
        raw_train_data: torch.Tensor,
        n_synthetic: int,
    ) -> torch.Tensor:
        """
        Geometry-based initialisation for a single sequence.

        Finds the single continuous block of `n_synthetic` timesteps that is 
        geometrically closest to the "average" behavior of the entire dataset 
        (the geometric center or medoid).

        Args:
            raw_train_data (Tensor): Un-windowed, normalised training data,
                                     shape (T, C).  T must be > n_synthetic.
            n_synthetic (int):       Number of timesteps in the synthetic
                                     sequence (e.g. 384).

        Returns:
            Tensor of shape (n_synthetic, C) with requires_grad=True.

        Raises:
            ValueError: If raw_train_data is too short to slice n_synthetic rows.
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
        total_windows = windows.shape[0]

        # Flatten windows for geometric distance calculations: shape (Total_Windows, n_synthetic * C)
        flat_windows = windows.reshape(total_windows, -1)

        # 2. Find the geometric center (the "mean" sequence shape) of the entire dataset
        mean_window = flat_windows.mean(dim=0, keepdim=True)

        # 3. Calculate Euclidean distance from every window to the mean window
        distances = torch.cdist(flat_windows, mean_window).squeeze()

        # 4. Select the index of the window that is geometrically closest to the center
        best_idx = torch.argmin(distances).item()

        # Extract, clone, and prep for the optimiser
        synthetic_seq = raw_train_data[best_idx : best_idx + n_synthetic].clone()
        synthetic_seq.requires_grad_(True)

        logger.info(f"  Geometry sequence initialised from random data (Most representative) "
              f"(rows {best_idx}-{best_idx + n_synthetic - 1}), "
              f"shape {tuple(synthetic_seq.shape)}")

        return synthetic_seq