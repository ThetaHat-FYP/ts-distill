"""
Phase-Aware MTT Distiller
--------------------------
A hybrid MTT distiller that switches the outer-loop loss depending on which
phase of the expert trajectory the sampled checkpoint pair comes from.

Each sampled (start, end) checkpoint pair must fall entirely within ONE
phase — if start and end straddle phase_boundary (one is early, the other
late), the pair is discarded and a new one is sampled, until both endpoints
agree:

  Early phase (start step < phase_boundary AND end step < phase_boundary):
      parameter matching loss
      grand_loss = ||theta_student - theta_target||^2 / ||theta_start - theta_target||^2

  Late phase  (start step >= phase_boundary AND end step >= phase_boundary):
      prediction matching loss
      grand_loss = ||f(X; theta_student) - f(X; theta_target)||^2
                   / ||f(X; theta_start) - f(X; theta_target)||^2

Motivation
----------
In the early phase the expert is rapidly moving through parameter space —
parameter matching gives a strong, architecture-specific directional signal
that drives fast improvement in the synthetic data.

In the late phase the expert has largely converged and is fine-tuning.
Parameter matching at this stage encodes architecture-specific noise
(small weight adjustments that are specific to the expert's network topology).
Switching to prediction matching at this point anchors the loss to the
functional behaviour of the model — what it predicts — which is
architecture-agnostic and should distill a synthetic dataset that
generalises better across architectures.

Phase boundaries (from interim report validation-loss analysis):
  LSTM    — epoch 48
  MLP     — epoch 30
  CNN     — epoch 58
  DLinear — None (no clear boundary; pass None to keep param matching throughout)

When phase_boundary is None the distiller behaves identically to standard MTTDistiller.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder


class PhaseAwareMTTDistiller(MTTDistiller):
    """
    MTT distiller with a phase-switched outer-loop loss.

    Constructor args are identical to MTTDistiller plus one extra:

    Args:
        phase_boundary (int | None): Expert training step at which to switch
            from parameter matching to prediction matching.  If None, the
            distiller uses parameter matching for the entire trajectory
            (identical behaviour to standard MTTDistiller).
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
        snapshot_student_steps: int = 50,
        seq_len: int = 96,
        pred_len: int = 96,
        device: str = "cpu",
        eps: float = 1e-12,
        phase_boundary: Optional[int] = None,
    ) -> None:
        super().__init__(
            initializer=initializer,
            matcher=matcher,
            model_factory=model_factory,
            expert_recorder=expert_recorder,
            expert_epochs=expert_epochs,
            syn_batch_size=syn_batch_size,
            synthetic_lr=synthetic_lr,
            student_lr=student_lr,
            student_steps=student_steps,
            snapshot_student_steps=snapshot_student_steps,
            seq_len=seq_len,
            pred_len=pred_len,
            device=device,
            eps=eps,
        )
        self.phase_boundary = phase_boundary

    # =========================================================================
    # PUBLIC — Outer distillation loop
    # =========================================================================

    def distill(
        self,
        synthetic_init: torch.Tensor,
        n_steps: int,
        val_data: Optional[torch.Tensor] = None,
        val_snapshot_every: int = 50,
    ) -> torch.Tensor:
        """
        Run the phase-aware MTT outer loop.

        For each distillation step, a checkpoint pair (start, end) is sampled
        and re-drawn until both checkpoints fall in the SAME phase relative to
        phase_boundary:
          - start step < phase_boundary AND end step < phase_boundary
                => early phase => parameter matching loss (standard MTT)
          - start step >= phase_boundary AND end step >= phase_boundary
                => late phase  => prediction matching loss
          - a pair that straddles phase_boundary (start in early, end in late)
                is discarded and a new pair is sampled
          - phase_boundary is None => always parameter matching (no resampling)

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

        boundary_label = (
            f"boundary={self.phase_boundary}"
            if self.phase_boundary is not None
            else "no boundary (full param matching)"
        )
        print(
            f"[PhaseAwareMTT] Distilling {n_synthetic} steps over {n_steps} steps "
            f"({boundary_label})..."
        )

        # ── Outer loop ────────────────────────────────────────────────────────
        for step in range(n_steps):

            optimizer_img.zero_grad()

            # ── Step 1: Sample expert trajectory segment ──────────────────────
            # Re-sample until both checkpoints fall in the same phase relative
            # to phase_boundary — a pair straddling the boundary is ambiguous
            # (neither purely "early" nor purely "late") and is discarded.
            if self.phase_boundary is None:
                start_ckpt, end_ckpt = self.expert_recorder.sample_checkpoint_pair(
                    step_gap=self.expert_epochs
                )
                is_late_phase = False
            else:
                for _attempt in range(1000):
                    start_ckpt, end_ckpt = self.expert_recorder.sample_checkpoint_pair(
                        step_gap=self.expert_epochs
                    )
                    start_is_late = start_ckpt["step"] >= self.phase_boundary
                    end_is_late   = end_ckpt["step"]   >= self.phase_boundary
                    if start_is_late == end_is_late:
                        is_late_phase = start_is_late
                        break
                else:
                    # Could not find a same-phase pair after many attempts
                    # (e.g. trajectory too short) — fall back to the
                    # start checkpoint's phase rather than looping forever.
                    is_late_phase = start_is_late

            expert_start_weights = start_ckpt["weights"]
            expert_end_weights   = end_ckpt["weights"]

            # ── Step 2: Fresh student ──────────────────────────────────────────
            student_model = self.model_factory().to(self.device)

            # ── Step 3: Differentiable inner loop ─────────────────────────────
            final_student_params, start_student_params = self._unroll_student(
                student_model        = student_model,
                synthetic_data       = synthetic_data,
                expert_start_weights = expert_start_weights,
            )

            # ── Step 4: Decide which loss to apply ────────────────────────────
            # is_late_phase was determined in Step 1 from the (same-phase) pair.
            if is_late_phase:
                grand_loss = self._prediction_matching_loss(
                    student_model        = student_model,
                    final_student_params = final_student_params,
                    start_student_params = start_student_params,
                    expert_end_weights   = expert_end_weights,
                    synthetic_data       = synthetic_data,
                )
            else:
                grand_loss = self._parameter_matching_loss(
                    final_student_params = final_student_params,
                    start_student_params = start_student_params,
                    expert_end_weights   = expert_end_weights,
                )

            # ── Step 5: Backpropagate and update synthetic data ───────────────
            grand_loss.backward()
            # Clip gradients to prevent synthetic_data exploding when
            # start_dist is near zero in late-phase plateau regions.
            if synthetic_data.grad is not None:
                torch.nn.utils.clip_grad_norm_([synthetic_data], max_norm=1.0)
            optimizer_img.step()

            if (step + 1) % 5 == 0 or step == 0:
                phase_tag = "LATE/pred" if is_late_phase else "EARLY/param"
                print(
                    f"[Step {step + 1:>3}/{n_steps}] "
                    f"Loss: {grand_loss.item():.4f}  [{phase_tag}]"
                )

            # ── Step 6 (optional): Best-snapshot validation ───────────────────
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

    # =========================================================================
    # PRIVATE — Loss helpers
    # =========================================================================

    def _parameter_matching_loss(
        self,
        final_student_params,
        start_student_params,
        expert_end_weights,
    ) -> torch.Tensor:
        """
        Standard MTT parameter-space loss (identical to MTTDistiller Step 5).

        grand_loss = ||theta_student_final - theta_target||^2
                     / ||theta_start - theta_target||^2
        """
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

        return param_loss / (param_dist + self.eps)

    def _prediction_matching_loss(
        self,
        student_model,
        final_student_params,
        start_student_params,
        expert_end_weights,
        synthetic_data,
    ) -> torch.Tensor:
        """
        Prediction-space loss (identical to PredictiveMTTDistiller Step 5).

        grand_loss = ||f(X; theta_student_final) - f(X; theta_target)||^2
                     / ||f(X; theta_start)       - f(X; theta_target)||^2

        X is a batch sampled from synthetic_data windows.
        Inputs are detached so gradient flows only through final_student_params.
        """
        # Build synthetic windows and sample a batch
        M       = synthetic_data.shape[0]
        windows = [
            synthetic_data[i : i + self.window_size]
            for i in range(M - self.window_size + 1)
        ]
        all_windows = torch.stack(windows)
        n_win       = all_windows.shape[0]

        batch_size = min(self.syn_batch_size, n_win)
        indices    = torch.randperm(n_win, device=self.device)[:batch_size]
        batch      = all_windows[indices]

        # Detach inputs: gradient must flow through params only, not data values
        inputs = batch[:, :self.seq_len, :].detach()

        # Expert (target) predictions — no gradient needed
        expert_end_on_device = {
            k: v.to(self.device) for k, v in expert_end_weights.items()
        }
        with torch.no_grad():
            expert_preds = torch.func.functional_call(
                student_model, expert_end_on_device, (inputs,)
            )

        # Student predictions — in graph for backprop through final_student_params
        student_preds = torch.func.functional_call(
            student_model, final_student_params, (inputs,)
        )

        # Starting predictions — denominator only, no grad needed
        start_detached = {k: v.detach() for k, v in start_student_params.items()}
        with torch.no_grad():
            start_preds = torch.func.functional_call(
                student_model, start_detached, (inputs,)
            )

        pred_loss  = F.mse_loss(student_preds, expert_preds)
        start_dist = F.mse_loss(start_preds,   expert_preds)

        # In the late-phase plateau consecutive checkpoints make nearly identical
        # predictions, so start_dist can be near zero.  Clamp to 1e-4 to prevent
        # the loss from exploding and driving synthetic_data to NaN.
        denom = torch.clamp(start_dist.detach(), min=1e-4)
        return pred_loss / denom
