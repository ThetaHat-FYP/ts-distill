"""
Real-Sample Initializer
------------------------
Initialises synthetic data from real training data.  Provides two strategies
depending on whether the distillation algorithm works with independent windows
or a single continuous sequence:

  initialize()          — window-based algorithms (FRePO, CondTSF).
                          Returns (n_windows, window_size, features).

  initialize_sequence() — continuous-sequence algorithms (MTT).
                          Returns a single (n_synthetic, features) tensor
                          by slicing a random contiguous block from the raw
                          training data.

Starting from real data (rather than random noise) places the synthetic tensor
on the data manifold from step 0, which gives the outer-loop optimiser a
better starting point and typically speeds up convergence.
"""

import torch

from ts_distill.distillation_core.initializer.base import BaseInitializer


class RealSampleInitializer(BaseInitializer):
    """
    Initialise synthetic data by sampling from real training data.

    Both methods return a tensor with requires_grad=True so it can be
    directly handed to an SGD/Adam optimiser in the distillation outer loop.
    """

    def initialize(
        self,
        shape: tuple,
        real_data_reference: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Window-based initialisation for algorithms like FRePO and CondTSF.

        Randomly samples `n_windows` independent windows from a pre-windowed
        real dataset.  Falls back to small Gaussian noise when no reference
        data is provided.

        Args:
            shape (tuple):                Target shape:
                                          (n_windows, window_size, n_features).
            real_data_reference (Tensor): Windowed real data to sample from,
                                          shape (N, window_size, n_features).
                                          Pass None to use random noise.

        Returns:
            Tensor of shape `shape` with requires_grad=True.
        """
        if real_data_reference is None:
            # Small scale keeps early gradients stable when there is no
            # real-data reference to warm-start from.
            data = torch.randn(shape) * 0.1
        else:
            n_samples = shape[0]
            indices   = torch.randint(0, len(real_data_reference), (n_samples,))
            data      = real_data_reference[indices].clone()

        data.requires_grad_(True)
        return data

    def initialize_sequence(
        self,
        raw_train_data: torch.Tensor,
        n_synthetic: int,
    ) -> torch.Tensor:
        """
        Continuous-sequence initialisation for MTT-style distillation.

        Picks a random contiguous block of n_synthetic timesteps from the
        raw (un-windowed) training data.  The block is cloned so the
        returned tensor is an independent leaf variable that the outer-loop
        optimiser can update freely.

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
        if len(raw_train_data) <= n_synthetic:
            raise ValueError(
                f"raw_train_data length ({len(raw_train_data)}) must be "
                f"greater than n_synthetic ({n_synthetic})."
            )

        # Random start so each run samples a different region of the dataset.
        start_idx     = torch.randint(0, len(raw_train_data) - n_synthetic, (1,)).item()
        synthetic_seq = raw_train_data[start_idx : start_idx + n_synthetic].clone()
        synthetic_seq.requires_grad_(True)

        print(f"   Synthetic sequence initialised from real data "
              f"(rows {start_idx}–{start_idx + n_synthetic - 1}), "
              f"shape {tuple(synthetic_seq.shape)}")

        return synthetic_seq
