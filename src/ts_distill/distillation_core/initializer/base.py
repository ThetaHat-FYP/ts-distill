from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseInitializer(ABC):
    """
    Interface for creating the starting synthetic tensor before distillation.

    Two initialisation strategies are supported by the framework:

      initialize()          — window-based algorithms (FRePO, CondTSF).
                              Must be implemented by every subclass.

      initialize_sequence() — continuous-sequence algorithms (MTT).
                              Optional: subclasses that support this strategy
                              override this method.  The default raises
                              NotImplementedError so misuse is caught early.
    """

    @abstractmethod
    def initialize(self, shape: tuple, real_data_reference: torch.Tensor = None) -> torch.Tensor:
        """
        Create a window-based synthetic initialisation.

        Args:
            shape (tuple):                (n_windows, window_size, n_features).
            real_data_reference (Tensor): Optional real windowed data to sample from.

        Returns:
            Tensor of shape `shape` with requires_grad=True.
        """
        pass

    def initialize_sequence(self, raw_train_data: torch.Tensor, n_synthetic: int) -> torch.Tensor:
        """
        Create a continuous-sequence synthetic initialisation for MTT-style distillation.

        Subclasses that support this strategy (e.g. RealSampleInitializer) must
        override this method.  The base implementation raises NotImplementedError
        so that using an incompatible initializer with MTT fails loudly.

        Args:
            raw_train_data (Tensor): Un-windowed training data, shape (T, C).
            n_synthetic (int):       Desired sequence length (e.g. 384).

        Returns:
            Tensor of shape (n_synthetic, C) with requires_grad=True.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support initialize_sequence(). "
            "Use RealSampleInitializer or implement this method in your subclass."
        )

    def refine(
        self,
        synthetic_data: torch.Tensor,
        real_data_reference: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Optional post-update refinement/projection of synthetic data.

        Distillers may call this after each outer-loop optimiser step to let an
        initializer enforce weak priors (e.g., frequency constraints).

        The default implementation is a no-op for backward compatibility.

        Args:
            synthetic_data:       Current synthetic tensor (2D or 3D).
            real_data_reference:  Optional real data (windowed or raw) used as a
                                  reference for refinement.
            step:                Current distillation step (0-indexed).
            n_steps:             Total number of distillation steps.

        Returns:
            Potentially updated synthetic_data (same object or a new tensor).
        """
        return synthetic_data