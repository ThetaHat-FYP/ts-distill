from typing import Callable, Optional

import torch
import torch.nn as nn

from ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher


class FRePODistiller(BaseDistiller):
    """Feature Regression based Prototype Optimization (FRePo) for time-series.

    Adaptation notes for this codebase:
    - Synthetic windows are prototypes.
    - Inputs are first half of window, targets are second half.
    - A temporary online model is fit on synthetic data each outer step.
    - Synthetic data is optimized using kernel ridge regression from prototype
      features to real-batch targets.
    """

    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory: Callable,
        feature_extractor: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
        synthetic_lr: float = 0.1,
        online_lr: float = 1e-3,
        online_updates: int = 10,
        syn_batch_size: int = 128,
        real_batch_size: int = 256,
        ridge_lambda: float = 1e-3,
        feature_norm_eps: float = 1e-8,
        device: str = "cpu",
    ) -> None:
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.feature_extractor = feature_extractor
        self.synthetic_lr = synthetic_lr
        self.online_lr = online_lr
        self.online_updates = online_updates
        self.syn_batch_size = syn_batch_size
        self.real_batch_size = real_batch_size
        self.ridge_lambda = ridge_lambda
        self.feature_norm_eps = feature_norm_eps
        self.device = device
        self.criterion = nn.MSELoss()

    def distill(
        self,
        source_data: torch.Tensor,
        n_steps: int,
        n_synthetic: Optional[int] = None,
    ) -> torch.Tensor:
        if n_synthetic is None:
            n_synthetic = 50

        synthetic_shape = (n_synthetic, source_data.shape[1], source_data.shape[2])
        synthetic_data = self.initializer.initialize(
            synthetic_shape,
            real_data_reference=source_data,
        ).to(self.device)

        if not synthetic_data.requires_grad:
            synthetic_data.requires_grad_(True)

        optimizer_img = torch.optim.SGD([synthetic_data], lr=self.synthetic_lr, momentum=0.5)

        print(f"FRePo distilling {n_synthetic} samples over {n_steps} steps...")

        for step in range(n_steps):
            optimizer_img.zero_grad()

            model = self.model_factory().to(self.device)
            self._online_fit_on_synthetic(model, synthetic_data)

            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()

            real_batch = self._sample_batch(source_data, self.real_batch_size)
            proto_batch = self._sample_batch(synthetic_data, self.syn_batch_size)

            loss = self._frepo_objective(model, proto_batch, real_batch)
            loss.backward()
            optimizer_img.step()

            if (step + 1) % 5 == 0 or step == 0:
                print(f"[Step {step + 1:>3}/{n_steps}] Loss: {loss.item():.4f}")

        return synthetic_data.detach()

    def _online_fit_on_synthetic(self, model: nn.Module, synthetic_data: torch.Tensor) -> None:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.online_lr)

        for _ in range(self.online_updates):
            # Keep synthetic-data dependency during online fitting.
            batch = self._sample_batch(synthetic_data, self.syn_batch_size)
            seq_len = batch.shape[1] // 2
            x = batch[:, :seq_len, :]
            y = batch[:, seq_len:, :]

            pred = model(x)
            loss = self.criterion(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    def _frepo_objective(
        self,
        model: nn.Module,
        proto_batch: torch.Tensor,
        real_batch: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = proto_batch.shape[1] // 2

        proto_x = proto_batch[:, :seq_len, :]
        proto_y = proto_batch[:, seq_len:, :]

        real_x = real_batch[:, :seq_len, :]
        real_y = real_batch[:, seq_len:, :]

        # True FRePo uses feature space (not final output space).
        phi_p = self._extract_features(model, proto_x)
        phi_r = self._extract_features(model, real_x)

        # Feature normalization improves kernel conditioning.
        phi_p = phi_p / (phi_p.norm(dim=1, keepdim=True) + self.feature_norm_eps)
        phi_r = phi_r / (phi_r.norm(dim=1, keepdim=True) + self.feature_norm_eps)

        y_p = proto_y.reshape(proto_y.shape[0], -1)
        y_r = real_y.reshape(real_y.shape[0], -1)

        k_pp = phi_p @ phi_p.transpose(0, 1)
        k_rp = phi_r @ phi_p.transpose(0, 1)

        eye = torch.eye(k_pp.shape[0], device=k_pp.device, dtype=k_pp.dtype)
        trace_scale = torch.trace(k_pp) / max(k_pp.shape[0], 1)
        k_pp_reg = k_pp + self.ridge_lambda * trace_scale * eye

        try:
            w = torch.linalg.solve(k_pp_reg, y_p)
        except RuntimeError:
            # Fallback when matrix is near-singular.
            w = torch.linalg.pinv(k_pp_reg) @ y_p
        pred_r = k_rp @ w

        return self.criterion(pred_r, y_r)

    def _sample_batch(self, data: torch.Tensor, batch_size: int) -> torch.Tensor:
        n = data.shape[0]
        if n <= batch_size:
            return data
        idx = torch.randperm(n, device=data.device)[:batch_size]
        return data[idx]

    def _extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            feat = self.feature_extractor(model, x)
        elif hasattr(model, "extract_features"):
            feat = model.extract_features(x)
        elif hasattr(model, "get_features"):
            feat = model.get_features(x)
        elif hasattr(model, "forward_features"):
            feat = model.forward_features(x)
        else:
            raise ValueError(
                "FRePO requires feature-space representations. "
                "Provide `feature_extractor` or implement `extract_features` on the model."
            )

        return feat.reshape(feat.shape[0], -1)