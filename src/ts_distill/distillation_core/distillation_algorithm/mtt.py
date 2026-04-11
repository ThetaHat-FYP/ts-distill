from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder


class MTTDistiller(BaseDistiller):
    """Matching Training Trajectories (MTT) distillation for time-series models.

    This implementation follows the original MTT training structure:
    1) sample (theta_t, theta_{t+k}) from expert trajectory
    2) unroll student updates on synthetic data
    3) optimize synthetic data by minimizing normalized parameter distance
    """

    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory: Callable,
        expert_recorder: BaseTrajectoryRecorder,
        expert_epochs: int = 10,
        syn_batch_size: int = 128,
        synthetic_lr: float = 0.01,
        student_lr: float = 0.01,
        student_steps: int = 10,
        device: str = "cpu",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.expert_recorder = expert_recorder
        self.expert_epochs = expert_epochs
        self.syn_batch_size = syn_batch_size
        self.synthetic_lr = synthetic_lr
        self.student_lr = student_lr
        self.student_steps = student_steps
        self.device = device
        self.eps = eps
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

        print(f"Distilling {n_synthetic} samples over {n_steps} steps...")

        for step in range(n_steps):
            optimizer_img.zero_grad()

            start_ckpt, end_ckpt = self.expert_recorder.sample_checkpoint_pair(
                step_gap=self.expert_epochs
            )
            expert_start_weights = start_ckpt["weights"]
            expert_end_weights = end_ckpt["weights"]

            student_model = self.model_factory().to(self.device)

            final_student_params, start_student_params = self._unroll_student(
                student_model=student_model,
                synthetic_data=synthetic_data,
                expert_start_weights=expert_start_weights,
            )

            target_param_list = []
            final_param_list = []
            start_param_list = []

            for name, start_param in start_student_params.items():
                if name not in expert_end_weights:
                    continue
                target_param = expert_end_weights[name].to(self.device).detach()
                final_param = final_student_params[name]
                final_param_list.append(final_param)
                start_param_list.append(start_param)
                target_param_list.append(target_param)

            param_loss = self.matcher.calculate_loss(final_param_list, target_param_list)
            param_dist = self.matcher.calculate_loss(start_param_list, target_param_list)

            # Follow original MTT normalization by start->target distance.
            grand_loss = param_loss / (param_dist + self.eps)

            grand_loss.backward()
            optimizer_img.step()

            if (step + 1) % 5 == 0 or step == 0:
                print(f"[Step {step + 1:>3}/{n_steps}] Loss: {grand_loss.item():.4f}")

        return synthetic_data.detach()

    def _unroll_student(
        self,
        student_model: nn.Module,
        synthetic_data: torch.Tensor,
        expert_start_weights: Dict[str, torch.Tensor],
    ):
        params = {}
        for name, param in student_model.named_parameters():
            if name in expert_start_weights:
                base = expert_start_weights[name].to(self.device)
            else:
                base = param.detach().to(self.device)
            params[name] = base.clone().detach().requires_grad_(True)

        start_params = {k: v for k, v in params.items()}

        num_syn_samples = synthetic_data.shape[0]

        for _ in range(self.student_steps):
            param_list = [params[name] for name in params.keys()]

            batch_size = min(self.syn_batch_size, num_syn_samples)
            indices = torch.randperm(num_syn_samples, device=synthetic_data.device)[:batch_size]
            batch_data = synthetic_data[indices]

            seq_len = batch_data.shape[1] // 2
            inputs = batch_data[:, :seq_len, :]
            targets = batch_data[:, seq_len:, :]

            predictions = torch.func.functional_call(student_model, params, (inputs,))
            loss = self.criterion(predictions, targets)

            grads = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                allow_unused=True,
            )

            params = {
                name: param - self.student_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(params.items(), grads)
            }

        return params, start_params