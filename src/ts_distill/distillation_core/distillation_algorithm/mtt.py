from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder


class MTTDistiller(BaseDistiller):
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
        snapshot_student_steps: int = 50,
        seq_len: int = 96,
        pred_len: int = 96,
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
        self.snapshot_student_steps = snapshot_student_steps
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_size = seq_len + pred_len
        self.device = device
        self.eps = eps
        self.criterion = nn.MSELoss()

    def distill(
        self,
        synthetic_init: torch.Tensor,
        n_steps: int,
        val_data: Optional[torch.Tensor] = None,
        val_snapshot_every: int = 50,
    ) -> torch.Tensor:
        """
        Run the MTT outer loop to optimise the synthetic sequence.

        The caller is responsible for creating the initial synthetic tensor
        (see init_synthetic_sequence() in run_cycle.py).  This keeps
        initialisation strategy separate from the distillation algorithm.

        Args:
            synthetic_init:     Starting synthetic sequence, shape (M, C).
                                Must already have requires_grad=True.
                                Produced by init_synthetic_sequence().
            n_steps:            Number of outer-loop optimisation steps.
            val_data:           Windowed validation windows (N_val, window_size, C).
                                When provided, a temporary student is scored every
                                `val_snapshot_every` steps; the best-scoring
                                snapshot is returned instead of the final one.
            val_snapshot_every: Outer-loop step interval for snapshot evaluation.
        """
        # Move to the distiller's device and keep the gradient tape alive.
        synthetic_data = synthetic_init.to(self.device)
        if not synthetic_data.requires_grad:
            synthetic_data.requires_grad_(True)

        n_synthetic    = synthetic_data.shape[0]
        optimizer_img  = torch.optim.SGD([synthetic_data], lr=self.synthetic_lr, momentum=0.5)

        best_synthetic = synthetic_data.detach().clone()
        best_val_mse   = float('inf')

        print(f"Distilling sequence of length {n_synthetic} over {n_steps} steps...")

        for step in range(n_steps):
            optimizer_img.zero_grad()

            start_ckpt, end_ckpt = self.expert_recorder.sample_checkpoint_pair(
                step_gap=self.expert_epochs
            )
            expert_start_weights = start_ckpt["weights"]
            expert_end_weights   = end_ckpt["weights"]

            student_model = self.model_factory().to(self.device)

            final_student_params, start_student_params = self._unroll_student(
                student_model=student_model,
                synthetic_data=synthetic_data,
                expert_start_weights=expert_start_weights,
            )

            target_param_list = []
            final_param_list  = []
            start_param_list  = []

            for name, start_param in start_student_params.items():
                if name not in expert_end_weights:
                    continue
                target_param = expert_end_weights[name].to(self.device).detach()
                final_param_list.append(final_student_params[name])
                start_param_list.append(start_param)
                target_param_list.append(target_param)

            param_loss = self.matcher.calculate_loss(final_param_list, target_param_list)
            param_dist = self.matcher.calculate_loss(start_param_list, target_param_list)

            grand_loss = param_loss / (param_dist + self.eps)
            grand_loss.backward()
            optimizer_img.step()

            if (step + 1) % 5 == 0 or step == 0:
                print(f"[Step {step + 1:>3}/{n_steps}] Loss: {grand_loss.item():.4f}")

            if val_data is not None and (step + 1) % val_snapshot_every == 0:
                val_mse = self._evaluate_snapshot(synthetic_data.detach(), val_data)
                if val_mse < best_val_mse:
                    best_val_mse   = val_mse
                    best_synthetic = synthetic_data.detach().clone()
                    print(f"   [Snapshot @ step {step + 1}] New best val MSE: {val_mse:.6f} (saved)")
                else:
                    print(f"   [Snapshot @ step {step + 1}] Val MSE: {val_mse:.6f} (best: {best_val_mse:.6f})")

        if val_data is not None:
            print(f"   Best snapshot val MSE: {best_val_mse:.6f}")
            return best_synthetic

        return synthetic_data.detach()

    def _evaluate_snapshot(
        self,
        synthetic_sequence: torch.Tensor,
        val_data: torch.Tensor,
    ) -> float:
        """Train a temporary student on the current synthetic windows and score it on val_data."""
        M = synthetic_sequence.shape[0]
        windows = [
            synthetic_sequence[i : i + self.window_size]
            for i in range(M - self.window_size + 1)
        ]
        if not windows:
            return float('inf')

        syn_windows = torch.stack(windows).to(self.device)
        n_syn = syn_windows.shape[0]

        temp_model     = self.model_factory().to(self.device)
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=self.student_lr)
        temp_model.train()

        for _ in range(self.snapshot_student_steps):
            batch_size = min(self.syn_batch_size, n_syn)
            idx     = torch.randperm(n_syn, device=self.device)[:batch_size]
            batch   = syn_windows[idx]
            inputs  = batch[:, :self.seq_len, :]
            targets = batch[:, self.seq_len:, :]
            temp_optimizer.zero_grad()
            output = temp_model(inputs)
            loss   = self.criterion(output, targets)
            loss.backward()
            temp_optimizer.step()

        # Batched evaluation to avoid OOM on large val sets
        temp_model.eval()
        val_data_dev  = val_data.to(self.device)
        criterion_sum = nn.MSELoss(reduction='sum')
        total_loss    = 0.0
        total_elems   = 0

        with torch.no_grad():
            n_val = val_data_dev.shape[0]
            for i in range(0, n_val, self.syn_batch_size):
                batch   = val_data_dev[i : i + self.syn_batch_size]
                inputs  = batch[:, :self.seq_len, :]
                targets = batch[:, self.seq_len:, :]
                output  = temp_model(inputs)
                total_loss  += criterion_sum(output, targets).item()
                total_elems += targets.numel()

        return total_loss / total_elems

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

        # FIX: Dynamically create rolling windows from the continuous sequence
        M = synthetic_data.shape[0]
        windows = [synthetic_data[i : i + self.window_size] for i in range(M - self.window_size + 1)]
        
        # If M=384 and win_size=192, we only get 193 valid windows to train on.
        all_batch_data = torch.stack(windows) 
        num_syn_samples = all_batch_data.shape[0]

        for _ in range(self.student_steps):
            param_list = [params[name] for name in params.keys()]

            batch_size = min(self.syn_batch_size, num_syn_samples)
            indices = torch.randperm(num_syn_samples, device=synthetic_data.device)[:batch_size]
            batch_data = all_batch_data[indices]

            # Slice inputs and targets exactly
            inputs = batch_data[:, :self.seq_len, :]
            targets = batch_data[:, self.seq_len:, :]

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