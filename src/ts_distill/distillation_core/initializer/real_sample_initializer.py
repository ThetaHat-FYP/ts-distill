"""
Real-Sample Initializer
------------------------
Initialises synthetic data by randomly sampling windows from real data.

Role in the pipeline
~~~~~~~~~~~~~~~~~~~~~
The initializer is called once at the start of distillation to set the
starting point of the optimised synthetic tensor.  A real-data initialisation
often converges faster than pure random noise because the synthetic data
already lies on the data manifold.

Note on MTTDistiller
~~~~~~~~~~~~~~~~~~~~~
MTTDistiller currently bypasses the initializer and slices a contiguous block
directly from the raw training sequence (to preserve temporal continuity).
This class is still useful for other distillation algorithms (FRePO, CondTSF)
or for any custom loop that prefers window-level initialisation.
"""

import torch

from ts_distill.distillation_core.initializer.base import BaseInitializer


class RealSampleInitializer(BaseInitializer):
    """
    Initialise synthetic windows by sampling randomly from real data.

    Falls back to small Gaussian noise when no real data reference is given.
    The returned tensor always has requires_grad=True so that it can be
    optimised by the distillation outer loop.
    """

    def initialize(
        self,
        shape: tuple,
        real_data_reference: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            shape (tuple):               Target shape, typically
                                         (n_synthetic_windows, window_size, n_features).
            real_data_reference (Tensor): Windowed real data to sample from,
                                          shape (N, window_size, n_features).
                                          If None, falls back to random noise.

        Returns:
            Tensor of shape `shape` with requires_grad=True.
        """
        if real_data_reference is None:
            # No real data provided — start from small random perturbations
            # around zero.  The small scale (0.1) prevents exploding gradients
            # in the first few outer-loop steps.
            data = torch.randn(shape) * 0.1
        else:
            n_samples = shape[0]
            indices   = torch.randint(0, len(real_data_reference), (n_samples,))
            data      = real_data_reference[indices].clone()

        data.requires_grad_(True)
        return data
