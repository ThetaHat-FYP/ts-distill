from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
class CondTSFDistiller(BaseDistiller):
    """CondTSF distiller for time-series.

    Core behavior:
    - Student unrolling is purely data-driven on synthetic (input, target) pairs.
    - Teacher guidance is applied only during periodic CondTSF refinement.

    Notes:
    - MTT-style parameter trajectory matching is intentionally not used.
    - This keeps CondTSF separated from MTT by design.
    """

    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory: Callable,
        teacher_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        expert_epochs: int = 10,
        syn_batch_size: int = 128,
        synthetic_lr: float = 0.01,
        student_lr: float = 0.01,
        student_steps: int = 10,
        cond_gap: int = 3,
        beta: float = 0.01,
        device: str = "cpu",
        eps: float = 1e-12,
    ) -> None:
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.teacher_state_dict = teacher_state_dict
        self.expert_epochs = expert_epochs
        self.syn_batch_size = syn_batch_size
        self.synthetic_lr = synthetic_lr
        self.student_lr = student_lr
        self.student_steps = student_steps
        self.cond_gap = max(1, int(cond_gap))
        self.beta = beta
        self.device = device
        self.eps = eps
        self.criterion = nn.MSELoss()
        self._student_init_params: Optional[Dict[str, torch.Tensor]] = None

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

        # Cache one student initialization and reuse it at every outer step for stability.
        if self._student_init_params is None:
            base_student = self.model_factory().to(self.device)
            self._student_init_params = {
                name: param.detach().clone()
                for name, param in base_student.named_parameters()
            }

        print(f"CondTSF distilling {n_synthetic} samples over {n_steps} steps...")

        for step in range(n_steps):
            optimizer_img.zero_grad()

            student_model = self.model_factory().to(self.device)

            # CondTSF objective: optimize synthetic data through unrolled
            # student training loss only (no parameter trajectory loss).
            unroll_loss = self._unroll_student(
                student_model=student_model,
                synthetic_data=synthetic_data,
            )

            unroll_loss.backward()
            optimizer_img.step()

            # CondTSF periodic refinement: blend synthetic future targets toward expert outputs.
            if (step + 1) % self.cond_gap == 0:
                self._apply_cond_refinement(synthetic_data)

            if (step + 1) % 5 == 0 or step == 0:
                print(f"[Step {step + 1:>3}/{n_steps}] Loss: {unroll_loss.item():.4f}")

        return synthetic_data.detach()

    def _unroll_student(
        self,
        student_model: nn.Module,
        synthetic_data: torch.Tensor,
    ) -> torch.Tensor:
        params = {}
        for name, param in student_model.named_parameters():
            if self._student_init_params is not None and name in self._student_init_params:
                base = self._student_init_params[name].to(self.device)
            else:
                base = param.detach().to(self.device)
            params[name] = base.clone().detach().requires_grad_(True)

        num_syn_samples = synthetic_data.shape[0]
        step_losses = []

        for _ in range(self.student_steps):
            param_list = [params[name] for name in params.keys()]

            batch_size = min(self.syn_batch_size, num_syn_samples)
            indices = torch.randperm(num_syn_samples, device=synthetic_data.device)[:batch_size]
            batch_data = synthetic_data[indices]

            seq_len = batch_data.shape[1] // 2
            inputs = batch_data[:, :seq_len, :]
            targets = batch_data[:, seq_len:, :]

            predictions = torch.func.functional_call(student_model, params, (inputs,))
            # CondTSF student training is data-supervised only.
            loss = self.criterion(predictions, targets)
            step_losses.append(loss)

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

        # Weight later unroll steps more than earlier steps.
        weights = torch.linspace(
            0.1,
            1.0,
            steps=len(step_losses),
            device=step_losses[0].device,
            dtype=step_losses[0].dtype,
        )
        weighted = torch.stack([w * l for w, l in zip(weights, step_losses)])
        return weighted.sum() / weights.sum()

    def _apply_cond_refinement(self, synthetic_data: torch.Tensor) -> None:
        if not self.teacher_state_dict:
            return

        teacher_model = self.model_factory().to(self.device)
        teacher_state = {
            name: tensor.to(self.device).detach()
            for name, tensor in self.teacher_state_dict.items()
            if name in dict(teacher_model.named_parameters())
        }

        seq_len = synthetic_data.shape[1] // 2
        inputs = synthetic_data[:, :seq_len, :]

        with torch.no_grad():
            teacher_pred = torch.func.functional_call(teacher_model, teacher_state, (inputs,))
            synthetic_data[:, seq_len:, :] = (
                (1.0 - self.beta) * synthetic_data[:, seq_len:, :] + self.beta * teacher_pred
            )