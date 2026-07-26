from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from ts_distill.distillation_core.initializer.base import BaseInitializer
from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder

from ts_distill._logging import get_logger

logger = get_logger(__name__)


class MTTDistiller(BaseDistiller):
    """
    Matching Training Trajectories (MTT) Distiller for time-series data.

    Core idea
    ---------
    Instead of matching model outputs or feature distributions, MTT matches
    the *training trajectory* — the path a model takes through parameter space
    when trained on real data.

    The goal: find a short synthetic sequence S such that a model trained on S
    for k gradient steps lands at the same parameters as a model trained on
    real data for k steps.  We achieve this by minimising the distance between
    the student's final parameters (trained on S) and the expert's target
    parameters (trained on real data), normalised by the baseline distance.

    Algorithm overview
    ------------------
    Outer loop (n_distill_steps):
      1. Sample an expert trajectory segment (θ_start → θ_target).
      2. Initialise a student at θ_start.
      3. Inner loop: train the student on synthetic data for student_steps
         steps using differentiable gradient descent (create_graph=True).
      4. Compute the normalised trajectory matching loss.
      5. Backpropagate through the entire inner loop into synthetic_data.
      6. Update synthetic_data with SGD.

    Parameters
    ----------
    initializer:            Responsible for creating the initial synthetic tensor.
    matcher:                Computes the differentiable parameter distance.
    model_factory:          Zero-argument callable that returns a fresh model.
    expert_recorder:        Holds the recorded expert weight trajectory.
    expert_epochs:          Step gap used when sampling checkpoint pairs.
    syn_batch_size:         Mini-batch size for synthetic window sampling.
    synthetic_lr:           Outer-loop learning rate for the synthetic tensor.
    student_lr:             Inner-loop learning rate for student gradient steps.
    student_steps:          Number of inner-loop gradient steps (k in the paper).
    snapshot_student_steps: Training steps for the validation snapshot student.
    seq_len / pred_len:     Input and forecast horizon lengths.
    eps:                    Small constant to prevent division by zero in the loss.
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
    ) -> None:
        super().__init__(initializer, matcher)
        self.model_factory          = model_factory
        self.expert_recorder        = expert_recorder
        self.expert_epochs          = expert_epochs
        self.syn_batch_size         = syn_batch_size
        self.synthetic_lr           = synthetic_lr
        self.student_lr             = student_lr
        self.student_steps          = student_steps
        self.snapshot_student_steps = snapshot_student_steps
        self.seq_len                = seq_len
        self.pred_len               = pred_len
        self.window_size            = seq_len + pred_len
        self.device                 = device
        self.eps                    = eps
        self.criterion              = nn.MSELoss()

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
        Run the MTT outer loop to optimise the synthetic sequence.

        The caller creates the initial tensor via any initializer's
        initialize_sequence() (e.g. RandomSampleInitializer) and passes it here.
        This keeps the initialisation strategy decoupled from the algorithm.

        Args:
            synthetic_init (Tensor):  Starting sequence, shape (M, C).
                                      Created by initialize_sequence().
            n_steps (int):            Number of outer-loop optimisation steps.
            val_data (Tensor | None): Windowed validation windows
                                      (N_val, window_size, C). When provided,
                                      a probe student is evaluated every
                                      val_snapshot_every steps and the
                                      best-scoring snapshot is returned.
            val_snapshot_every (int): Step interval for snapshot evaluation.

        Returns:
            Tensor (M, C): The optimised synthetic sequence — either the
            best validation snapshot (if val_data given) or the final step.
        """

        # ── Setup ─────────────────────────────────────────────────────────────
        # Move the synthetic sequence to the target device and ensure it is a
        # differentiable leaf tensor so SGD can accumulate gradients into it.
        synthetic_data = synthetic_init.detach().clone().to(self.device)
        synthetic_data.requires_grad_(True)

        n_synthetic = synthetic_data.shape[0]

        # Snapshot of initial synthetic for drift tracking (diagnostic).
        synthetic_data_init = synthetic_data.detach().clone()

        # The outer-loop optimiser updates the synthetic DATA tensor directly.
        # Momentum=0.5 damps oscillations that are common in meta-learning loops.
        optimizer_img = torch.optim.SGD([synthetic_data], lr=self.synthetic_lr, momentum=0.5)

        # Track the best synthetic sequence seen so far (used when val_data
        # is provided to guard against outer-loop overfitting).
        best_synthetic = synthetic_data.detach().clone()
        best_val_mse   = float('inf')

        logger.info(f"Distilling sequence of length {n_synthetic} over {n_steps} steps...")

        # ── Outer loop ────────────────────────────────────────────────────────
        for step in range(n_steps):

            # Zero the gradient accumulated on synthetic_data from the last step.
            optimizer_img.zero_grad()

            # ── Step 1: Sample an expert trajectory segment ───────────────────
            # Pick two checkpoints (θ_start, θ_target) that are expert_epochs
            # apart in the recorded real-data training trajectory.
            # θ_start is where the student will begin; θ_target is where we
            # want the student to end up after training on synthetic data.
            start_ckpt, end_ckpt = self.expert_recorder.sample_checkpoint_pair(
                step_gap=self.expert_epochs
            )
            expert_start_weights = start_ckpt["weights"]   # θ_start
            expert_end_weights   = end_ckpt["weights"]     # θ_target

            # ── Step 2: Initialise a fresh student at θ_start ─────────────────
            # A brand-new model instance is created so it shares architecture
            # with the expert but carries no training history of its own.
            # _unroll_student() copies expert_start_weights into its params.
            student_model = self.model_factory().to(self.device)

            # ── Step 3: Inner loop — train student on synthetic data ──────────
            # Runs student_steps differentiable gradient-descent steps on the
            # synthetic sequence.  "Differentiable" means the gradient graph is
            # kept alive through every inner step so that outer-loop .backward()
            # can propagate all the way back into synthetic_data.
            # Returns:
            #   final_student_params — student weights after inner training
            #   start_student_params — student weights before inner training (= θ_start)
            final_student_params, start_student_params = self._unroll_student(
                student_model        = student_model,
                synthetic_data       = synthetic_data,
                expert_start_weights = expert_start_weights,
            )

            # ── Step 4: Build aligned parameter lists for the loss ────────────
            # We align three lists by parameter name:
            #   final_param_list  — where the student ended up (trained on synthetic data)
            #   target_param_list — where the expert ended up (trained on real data)
            #   start_param_list  — where both started (θ_start, the shared origin)
            # Parameters that exist in the student but not in the expert checkpoint
            # are silently skipped (can happen with batch-norm running stats etc.).
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

            # ── Step 5: Compute the normalised trajectory matching loss ───────
            #
            # param_loss = ||θ_student_final − θ_target||²
            #   How far the student (trained on synthetic data) is from the
            #   expert's target checkpoint.  We want this to be small.
            #
            # param_dist = ||θ_start − θ_target||²
            #   Baseline distance from the shared starting point to the target.
            #   This tells us how much the expert moved in the same number of steps.
            #
            # grand_loss = param_loss / (param_dist + eps)
            #   Normalised ratio.  If the student closes the gap as effectively
            #   as the expert does on real data, grand_loss → 1.  Minimising it
            #   pushes the student to match the expert's trajectory as closely as
            #   possible, which forces synthetic_data to carry the same training
            #   signal as real data.
            #   eps prevents division by zero when θ_start ≈ θ_target.
            param_loss = self.matcher.calculate_loss(final_param_list, target_param_list)
            param_dist = self.matcher.calculate_loss(start_param_list, target_param_list)

            grand_loss = param_loss / (param_dist + self.eps)

            # ── Step 6: Backpropagate through the inner loop into synthetic_data
            # .backward() traces the full computation graph — from grand_loss
            # back through the inner-loop gradient steps (made possible by
            # create_graph=True in _unroll_student) and finally into
            # synthetic_data.  This is the meta-gradient that tells us: "how
            # should we change the pixel/timestep values so that training on
            # them produces parameter updates more like the expert's?"
            grand_loss.backward()

            # Clip meta-gradient norm before applying the outer update.
            # When grand_loss > 1 (student moved away from target), the raw
            # gradient can spike by orders of magnitude — especially for
            # recurrent models (LSTM) whose backward pass already amplifies
            # gradients through time.  Clipping at 1.0 caps catastrophic
            # updates without affecting steps where grad_norm is already small
            # (e.g. MLP typically has grad_norm ~5e-3, never triggers).
            torch.nn.utils.clip_grad_norm_([synthetic_data], max_norm=1.0)

            # ── Step 7: Update the synthetic sequence with the meta-gradient ──
            optimizer_img.step()

            if (step + 1) % 5 == 0 or step == 0:
                grad_norm = (synthetic_data.grad.norm().item()
                             if synthetic_data.grad is not None else 0.0)
                seq_delta = (synthetic_data.detach() - synthetic_data_init).abs().mean().item()
                logger.info(
                    f"[Step {step + 1:>3}/{n_steps}]"
                    f"  loss={grand_loss.item():.4f}"
                    f"  param_loss={param_loss.item():.3f}"
                    f"  param_dist={param_dist.item():.3f}"
                    f"  grad_norm={grad_norm:.2e}"
                    f"  seq_delta={seq_delta:.2e}"
                )

            # ── Step 8 (optional): Best-snapshot validation ───────────────────
            # Every val_snapshot_every steps, train a fresh probe student on the
            # current synthetic sequence and score it on the real validation set.
            # If the probe beats the current best, save the synthetic sequence.
            # This guards against outer-loop overfitting: the distillation loss
            # can keep decreasing while the actual generalisation quality peaks
            # somewhere in the middle of training.
            if val_data is not None and (step + 1) % val_snapshot_every == 0:
                val_mse = self._evaluate_snapshot(synthetic_data.detach(), val_data)
                if val_mse < best_val_mse:
                    best_val_mse   = val_mse
                    best_synthetic = synthetic_data.detach().clone()
                    logger.info(f"   [Snapshot @ step {step + 1}] New best val MSE: {val_mse:.6f} (saved)")
                else:
                    logger.info(f"   [Snapshot @ step {step + 1}] Val MSE: {val_mse:.6f} (best: {best_val_mse:.6f})")

        # ── Return ─────────────────────────────────────────────────────────────
        # If validation tracking was active, return the best checkpoint found
        # during training rather than the potentially overfit final sequence.
        # If no snapshot was triggered (n_steps < val_snapshot_every), evaluate
        # the final state now so the returned tensor reflects actual optimization.
        if val_data is not None:
            if best_val_mse == float('inf'):
                val_mse = self._evaluate_snapshot(synthetic_data.detach(), val_data)
                best_val_mse   = val_mse
                best_synthetic = synthetic_data.detach().clone()
                logger.info(f"   [Snapshot @ step {n_steps}] Val MSE: {val_mse:.6f} (end-of-run)")
            logger.info(f"   Best snapshot val MSE: {best_val_mse:.6f}")
            return best_synthetic

        return synthetic_data.detach()

    # =========================================================================
    # PRIVATE — Validation snapshot evaluation
    # =========================================================================

    def _evaluate_snapshot(
        self,
        synthetic_sequence: torch.Tensor,
        val_data: torch.Tensor,
    ) -> float:
        """
        Measure the real-world quality of the current synthetic sequence.

        Training signal (grand_loss) tells us how well the sequence drives
        trajectory matching, but it does not directly measure forecast accuracy.
        This method provides an independent quality signal by:
          1. Deriving sliding windows from the synthetic sequence.
          2. Training a fresh student on those windows (standard Adam, no
             meta-gradient — this is a normal training loop).
          3. Evaluating that student's forecast MSE on the real validation set.

        A lower val_mse means the synthetic sequence transfers better to
        unseen real data.

        Args:
            synthetic_sequence (Tensor): Current synthetic data, shape (M, C).
                                         Detached — no gradient needed here.
            val_data (Tensor):           Real validation windows,
                                         shape (N_val, window_size, C).

        Returns:
            float: Mean squared error per element on the validation set.
        """

        # ── Build sliding windows from the synthetic sequence ─────────────────
        # Same windowing logic used in the inner loop: step by 1 each time.
        # With M=384 and window_size=192 we get 193 overlapping windows.
        M = synthetic_sequence.shape[0]
        windows = [
            synthetic_sequence[i : i + self.window_size]
            for i in range(M - self.window_size + 1)
        ]
        if not windows:
            # Sequence is shorter than one window — cannot evaluate.
            return float('inf')

        syn_windows = torch.stack(windows).to(self.device)
        n_syn       = syn_windows.shape[0]

        # ── Train a temporary student on the synthetic windows ────────────────
        # This is a plain supervised training loop (not differentiable meta-
        # learning).  Adam is used here instead of SGD because we want fast,
        # stable convergence in the limited snapshot_student_steps budget.
        temp_model     = self.model_factory().to(self.device)
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=self.student_lr)
        temp_model.train()

        for _ in range(self.snapshot_student_steps):
            # Random mini-batch from synthetic windows
            batch_size = min(self.syn_batch_size, n_syn)
            idx        = torch.randperm(n_syn, device=self.device)[:batch_size]
            batch      = syn_windows[idx]

            # Split window into input look-back and forecast target
            inputs  = batch[:, :self.seq_len, :]    # (B, seq_len, C)
            targets = batch[:, self.seq_len:, :]    # (B, pred_len, C)

            temp_optimizer.zero_grad()
            output = temp_model(inputs)
            loss   = self.criterion(output, targets)
            loss.backward()
            temp_optimizer.step()

        # ── Evaluate student on the real validation set ───────────────────────
        # Use reduction='sum' and accumulate manually so the final average is
        # exact regardless of whether the last batch is smaller than the others.
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
                total_elems += targets.numel()   # B × pred_len × C

        # Mean squared error per individual predicted value
        return total_loss / total_elems

    # =========================================================================
    # PRIVATE — Differentiable inner loop (student unrolling)
    # =========================================================================

    def _unroll_student(
        self,
        student_model: nn.Module,
        synthetic_data: torch.Tensor,
        expert_start_weights: Dict[str, torch.Tensor],
    ):
        """
        Run the inner-loop gradient descent on synthetic data — differentiably.

        This is the most algorithmically critical part of MTT.  The key
        constraint is that every operation must remain in the PyTorch
        computation graph so that when grand_loss.backward() is called in the
        outer loop, gradients flow through ALL inner steps and back into
        synthetic_data.

        How differentiable gradient descent works here
        -----------------------------------------------
        Normal training: model.parameters() → loss → loss.backward() → optimizer.step()
            ✗ optimizer.step() breaks the graph — gradients do not flow back
              through the update into the data.

        Differentiable training (this function):
            ✓ Parameters are stored as plain tensors in a dict (params).
            ✓ torch.func.functional_call runs the forward pass using those
              tensors rather than model.parameters(), keeping them in the graph.
            ✓ torch.autograd.grad with create_graph=True computes gradients
              and records the gradient computation itself as graph nodes.
            ✓ The manual update  param ← param − lr × grad  is a regular
              tensor subtraction — fully differentiable.

        Result: grand_loss depends on final params, which depend on every
        gradient step, which depend on the losses computed on synthetic_data.
        So ∂grand_loss/∂synthetic_data is well-defined and non-zero.

        Args:
            student_model (nn.Module):          Architecture template (weights ignored).
            synthetic_data (Tensor):            Current synthetic sequence (M, C),
                                                with requires_grad=True.
            expert_start_weights (dict):        θ_start checkpoint from the expert.

        Returns:
            (final_params, start_params): Both are dicts {name: Tensor}.
        """

        # ── Initialise student parameters at θ_start ─────────────────────────
        # For each named parameter in the model:
        #   - If the expert checkpoint has a value for it, use that (warm start).
        #   - Otherwise fall back to the model's own random initialisation.
        # .clone().detach().requires_grad_(True) creates a NEW leaf tensor that
        # is connected to the outer computation graph via the inner gradient steps
        # but not to the model's own parameter buffers.
        params = {}
        for name, param in student_model.named_parameters():
            if name in expert_start_weights:
                base = expert_start_weights[name].to(self.device)
            else:
                base = param.detach().to(self.device)
            params[name] = base.clone().detach().requires_grad_(True)

        # Save a snapshot of the starting parameters (θ_start).
        # This is used later to compute param_dist (the baseline distance).
        start_params = {k: v for k, v in params.items()}

        # ── Derive sliding windows from the synthetic sequence ─────────────────
        # We slide a window of size (seq_len + pred_len) across the M-length
        # continuous sequence one step at a time.
        # With M=384 and window_size=192: 384 − 192 + 1 = 193 windows.
        #
        # Crucially, each window is a SLICE of synthetic_data — not a copy.
        # Slices share the underlying storage, so the computation graph stays
        # connected: loss on a window → gradient w.r.t. that slice → gradient
        # w.r.t. synthetic_data (by chain rule through torch.stack).
        M       = synthetic_data.shape[0]
        windows = [synthetic_data[i : i + self.window_size] for i in range(M - self.window_size + 1)]
        all_batch_data  = torch.stack(windows)   # (N_windows, window_size, C)
        num_syn_samples = all_batch_data.shape[0]

        # ── Inner gradient-descent loop ───────────────────────────────────────
        for _ in range(self.student_steps):

            # Collect current parameter tensors as a list for autograd.grad.
            param_list = [params[name] for name in params.keys()]

            # Random mini-batch from the synthetic windows.
            batch_size = min(self.syn_batch_size, num_syn_samples)
            indices    = torch.randperm(num_syn_samples, device=synthetic_data.device)[:batch_size]
            batch_data = all_batch_data[indices]

            # Split each window into look-back (input) and forecast (target).
            inputs  = batch_data[:, :self.seq_len, :]   # (B, seq_len, C)
            targets = batch_data[:, self.seq_len:, :]   # (B, pred_len, C)

            # Forward pass using functional_call.
            # functional_call injects `params` as the model weights for this
            # call only — the model's own .parameters() are untouched.
            # This is required because standard model(inputs) uses the model's
            # internal parameters, which are NOT in our differentiable params dict.
            predictions = torch.func.functional_call(student_model, params, (inputs,))
            loss        = self.criterion(predictions, targets)

            # Compute gradients of the loss w.r.t. each parameter tensor.
            # create_graph=True: record the gradient computation itself as graph
            #   nodes so that second-order derivatives exist.  Without this,
            #   grand_loss.backward() would fail to propagate through this step.
            # allow_unused=True: some parameters (e.g. biases in certain configs)
            #   may not contribute to the loss; treat their gradient as zero.
            grads = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                allow_unused=True,
            )

            # Manual SGD step — subtract lr × gradient from each parameter.
            # Replacing None gradients with zeros handles allow_unused cases.
            # This update is a plain tensor operation, so it stays in the graph.
            params = {
                name: param - self.student_lr * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(params.items(), grads)
            }

        # Return the student's final parameters (after inner training) and
        # starting parameters (before inner training) for loss computation.
        return params, start_params