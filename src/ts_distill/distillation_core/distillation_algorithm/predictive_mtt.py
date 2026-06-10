"""
Predictive MTT Distiller
------------------------
A variant of MTTDistiller that uses prediction-space matching instead of
parameter-space matching for the trajectory alignment loss.

Standard MTT (parameter matching):
    grand_loss = ||theta_student_final - theta_target||^2
                 / ||theta_start - theta_target||^2

Predictive MTT (prediction matching):
    grand_loss = ||f(X; theta_student_final) - f(X; theta_target)||^2
                 / ||f(X; theta_start)       - f(X; theta_target)||^2

where X is a batch sampled from the current synthetic sequence.

Motivation
----------
Parameter matching is architecture-specific: the loss directly compares
weight tensors, so the gradient signal encodes the expert's architectural
inductive bias.  Prediction matching is architecture-agnostic in its signal:
the loss compares model outputs, which carry only what the model predicts,
not how it is parameterised.

This file does NOT modify mtt.py.  PredictiveMTTDistiller is a standalone
subclass that overrides only the distill() method.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller


class PredictiveMTTDistiller(MTTDistiller):
    """
    MTT distiller that uses prediction-space matching as the outer-loop loss.

    The differentiable inner loop (_unroll_student) is inherited unchanged.
    Only the loss computation block in distill() is replaced.

    All constructor arguments are identical to MTTDistiller.
    """

    def distill(
        self,
        synthetic_init: torch.Tensor,
        n_steps: int,
        val_data: Optional[torch.Tensor] = None,
        val_snapshot_every: int = 50,
    ) -> torch.Tensor:
        """
        Run the predictive-matching MTT outer loop.

        Identical to MTTDistiller.distill() except Step 5 (loss computation):
        instead of comparing parameter vectors, we compare model predictions
        on a batch drawn from the synthetic sequence.

        Args / Returns: same as MTTDistiller.distill().
        """

        # ── Setup ─────────────────────────────────────────────────────────────
        synthetic_data = synthetic_init.detach().clone().to(self.device)
        synthetic_data.requires_grad_(True)

        n_synthetic = synthetic_data.shape[0]

        optimizer_img = torch.optim.SGD(
            [synthetic_data], lr=self.synthetic_lr, momentum=0.5
        )

        best_synthetic = synthetic_data.detach().clone()
        best_val_mse   = float("inf")

        print(
            f"[PredMTT] Distilling sequence of length {n_synthetic} "
            f"over {n_steps} steps..."
        )

        # ── Outer loop ────────────────────────────────────────────────────────
        for step in range(n_steps):

            optimizer_img.zero_grad()

            # ── Step 1: Sample expert trajectory segment ──────────────────────
            start_ckpt, end_ckpt = self.expert_recorder.sample_checkpoint_pair(
                step_gap=self.expert_epochs
            )
            expert_start_weights = start_ckpt["weights"]
            expert_end_weights   = end_ckpt["weights"]

            # ── Step 2: Fresh student ──────────────────────────────────────────
            student_model = self.model_factory().to(self.device)

            # ── Step 3: Differentiable inner loop ─────────────────────────────
            # Inherited from MTTDistiller — returns differentiable final params
            # and the (detached) starting params.
            final_student_params, start_student_params = self._unroll_student(
                student_model        = student_model,
                synthetic_data       = synthetic_data,
                expert_start_weights = expert_start_weights,
            )

            # ── Step 4: Sample a batch for prediction evaluation ───────────────
            # Build sliding windows from the current synthetic sequence (same
            # logic as the inner loop).  Detach the inputs so the gradient
            # path to synthetic_data comes only through final_student_params
            # (via the inner loop), not through a direct input-value path.
            M       = synthetic_data.shape[0]
            windows = [
                synthetic_data[i : i + self.window_size]
                for i in range(M - self.window_size + 1)
            ]
            all_windows = torch.stack(windows)      # (N, window_size, C)
            n_win       = all_windows.shape[0]

            batch_size = min(self.syn_batch_size, n_win)
            indices    = torch.randperm(n_win, device=self.device)[:batch_size]
            batch      = all_windows[indices]

            # Detach: gradient must not flow through the data values here
            inputs = batch[:, :self.seq_len, :].detach()

            # ── Step 5: Prediction-matching loss ──────────────────────────────
            #
            # pred_loss  = ||f(X; theta_student_final) - f(X; theta_target)||^2
            #   The student (after training on synthetic data) should predict
            #   what the expert predicts at the target checkpoint.  We want
            #   this to be small.
            #
            # start_dist = ||f(X; theta_start) - f(X; theta_target)||^2
            #   Baseline: how far the predictions diverge at the shared start.
            #   Normalises the loss so we measure the fraction of prediction
            #   gap that synthetic training closes.
            #
            # grand_loss = pred_loss / (start_dist + eps)
            #   If the student closes the prediction gap as well as the expert
            #   closes it on real data, grand_loss -> 1.

            # Expert (target) predictions — fixed, no gradient needed
            expert_end_on_device = {
                k: v.to(self.device) for k, v in expert_end_weights.items()
            }
            with torch.no_grad():
                expert_preds = torch.func.functional_call(
                    student_model, expert_end_on_device, (inputs,)
                )

            # Student predictions — in graph so backprop flows through
            # final_student_params into the inner loop and then synthetic_data
            student_preds = torch.func.functional_call(
                student_model, final_student_params, (inputs,)
            )

            # Starting predictions — only needed as a normalization denominator,
            # so no gradient is required
            start_detached = {k: v.detach() for k, v in start_student_params.items()}
            with torch.no_grad():
                start_preds = torch.func.functional_call(
                    student_model, start_detached, (inputs,)
                )

            pred_loss  = F.mse_loss(student_preds, expert_preds)
            start_dist = F.mse_loss(start_preds,   expert_preds)

            denom      = torch.clamp(start_dist.detach(), min=1e-4)
            grand_loss = pred_loss / denom

            # ── Step 6: Backpropagate through inner loop into synthetic_data ──
            grand_loss.backward()
            if synthetic_data.grad is not None:
                torch.nn.utils.clip_grad_norm_([synthetic_data], max_norm=1.0)

            # ── Step 7: Update synthetic sequence ─────────────────────────────
            optimizer_img.step()

            if (step + 1) % 5 == 0 or step == 0:
                print(f"[Step {step + 1:>3}/{n_steps}] Loss: {grand_loss.item():.4f}")

            # ── Step 8 (optional): Best-snapshot validation ───────────────────
            if val_data is not None and (step + 1) % val_snapshot_every == 0:
                val_mse = self._evaluate_snapshot(synthetic_data.detach(), val_data)
                if val_mse < best_val_mse:
                    best_val_mse   = val_mse
                    best_synthetic = synthetic_data.detach().clone()
                    print(
                        f"   [Snapshot @ step {step + 1}] "
                        f"New best val MSE: {val_mse:.6f} (saved)"
                    )
                else:
                    print(
                        f"   [Snapshot @ step {step + 1}] "
                        f"Val MSE: {val_mse:.6f} (best: {best_val_mse:.6f})"
                    )

        # ── Return ─────────────────────────────────────────────────────────────
        if val_data is not None:
            if best_val_mse == float("inf"):
                val_mse        = self._evaluate_snapshot(
                    synthetic_data.detach(), val_data
                )
                best_val_mse   = val_mse
                best_synthetic = synthetic_data.detach().clone()
                print(
                    f"   [Snapshot @ step {n_steps}] "
                    f"Val MSE: {val_mse:.6f} (end-of-run)"
                )
            print(f"   Best snapshot val MSE: {best_val_mse:.6f}")
            return best_synthetic

        return synthetic_data.detach()
