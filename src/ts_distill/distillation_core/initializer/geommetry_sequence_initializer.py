import torch
from ts_distill.distillation_core.initializer.base import BaseInitializer


class GeometrySequenceInitializer(BaseInitializer):
    def _apply_gaussian_noise(
        self,
        sequence: torch.Tensor,
        noise_std: float,
    ) -> torch.Tensor:
        if noise_std <= 0:
            return sequence
        return sequence + torch.randn_like(sequence) * noise_std

    def initialize(self, shape: tuple, real_data_reference: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError("GeometrySequenceInitializer only supports continuous sequence initialization.")

    def initialize_sequence(
        self,
        raw_train_data: torch.Tensor,
        n_synthetic: int,
        noise_std: float = 0.08,
    ) -> torch.Tensor:
        """
        Herding-based initialisation for a single synthetic sequence.

        Iteratively selects timesteps whose running mean best approximates the
        global mean of the training data (Welling 2009 / dataset-distillation
        herding).  Selected indices are sorted to preserve temporal order, then
        concatenated to form a sequence of shape (n_synthetic, C).

        Args:
            raw_train_data (Tensor): Un-windowed, normalised training data,
                                     shape (T, C).  T must be > n_synthetic.
            n_synthetic    (int):    Number of timesteps in the output sequence.
            noise_std      (float):  Std of Gaussian noise added after selection.

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

        device = raw_train_data.device
        global_mean  = raw_train_data.mean(dim=0)          # (C,)
        running_sum  = torch.zeros(C, device=device)
        selected_idx = []

        for step in range(n_synthetic):
            # Score each timestep: how close would the new running mean be
            # to the global mean if we added that timestep?
            candidates = (running_sum.unsqueeze(0) + raw_train_data) / (step + 1)
            scores     = torch.norm(global_mean.unsqueeze(0) - candidates, dim=1)
            best       = int(torch.argmin(scores).item())
            selected_idx.append(best)
            running_sum = running_sum + raw_train_data[best]

        # Sort by time index to preserve temporal structure
        selected_idx.sort()
        synthetic_seq = raw_train_data[selected_idx].clone()
        synthetic_seq = self._apply_gaussian_noise(synthetic_seq, noise_std)
        synthetic_seq.requires_grad_(True)

        print(
            f"  Herding initialised synthetic sequence "
            f"({n_synthetic} timesteps selected from {T}), "
            f"shape {tuple(synthetic_seq.shape)}"
        )

        return synthetic_seq
