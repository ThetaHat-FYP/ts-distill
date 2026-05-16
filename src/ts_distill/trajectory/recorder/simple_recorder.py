import random
from typing import Dict, List, Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np

from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder
from ts_distill.trainer.callback.base import BaseCallback


class SimpleRecorder(BaseTrajectoryRecorder, BaseCallback):
    """
    Extended Recorder for MTT

    Tracks:
    - Training loss
    - Validation loss
    - Gradient norm
    - Weight change
    - Full model checkpoints
    """

    def __init__(self, record_every: int = 1):
        self.trajectory: List[Dict] = []
        self.record_every = record_every

        # histories
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.grad_norm_history: List[float] = []
        self.weight_change_history: List[float] = []

        self.prev_weights = None

    # ============================================================
    # CALLBACKS
    # ============================================================

    def on_train_begin(self, model, **kwargs):
        self.trajectory = []

        self.loss_history = []
        self.val_loss_history = []
        self.grad_norm_history = []
        self.weight_change_history = []

        self.prev_weights = None

        self.record_checkpoint(
            model=model,
            step=0,
            loss=0.0,
            grad_norm=0.0,
            weight_change=0.0
        )

    def on_epoch_end(self, model, epoch: int, loss: float, val_loss: float = None, **kwargs):

        if (epoch + 1) % self.record_every == 0:

            grad_norm = self.compute_grad_norm(model)
            weight_change = self.compute_weight_change(model)

            self.record_checkpoint(
                model=model,
                step=epoch + 1,
                loss=loss,
                grad_norm=grad_norm,
                weight_change=weight_change
            )

            # ---- FIX 1: ALWAYS APPEND (keeps alignment) ----
            self.loss_history.append(loss)
            self.val_loss_history.append(val_loss if val_loss is not None else float("nan"))

            self.grad_norm_history.append(grad_norm)
            self.weight_change_history.append(weight_change)

            # update previous weights
            self.prev_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    def on_train_end(self, model, **kwargs):
        pass

    # ============================================================
    # CORE RECORDING
    # ============================================================

    def record_checkpoint(self, model, step, loss, grad_norm, weight_change):
        weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        self.trajectory.append({
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "weight_change": weight_change,
            "weights": weights
        })

    # ============================================================
    # METRICS
    # ============================================================

    def compute_grad_norm(self, model: torch.nn.Module) -> float:
        total_norm = 0.0

        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2

        return total_norm ** 0.5

    def compute_weight_change(self, model: torch.nn.Module) -> float:
        if self.prev_weights is None:
            return 0.0

        current = model.state_dict()
        change = 0.0

        for k in current:
            diff = current[k].cpu() - self.prev_weights[k]
            change += torch.norm(diff).item() ** 2

        return change ** 0.5

    # ============================================================
    # MTT SAMPLING
    # ============================================================

    def get_trajectory(self) -> List[Dict]:
        return self.trajectory

    def sample_checkpoint_pair(self, step_gap: int = 10) -> Tuple[Dict, Dict]:
        if len(self.trajectory) < 2:
            raise ValueError("Need at least 2 checkpoints")

        actual_gap = min(step_gap, len(self.trajectory) - 1)
        start_idx = random.randint(0, len(self.trajectory) - actual_gap - 1)

        return (
            self.trajectory[start_idx],
            self.trajectory[start_idx + actual_gap]
        )

    def sample_checkpoint(self) -> Dict:
        return random.choice(self.trajectory)

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def plot_loss_with_validation(self):
        plt.figure()

        epochs = np.arange(1, len(self.loss_history) + 1)

        plt.plot(epochs, self.loss_history, label="Train Loss")

        if len(self.val_loss_history) > 0:
            plt.plot(epochs, self.val_loss_history, label="Validation Loss")

        plt.title("Train vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    # ============================================================
    # FIXED OVERFITTING PLOT
    # ============================================================

    def plot_overfitting_curve(self):
        if len(self.val_loss_history) == 0:
            print("No validation loss recorded.")
            return

        epochs = np.arange(1, len(self.val_loss_history) + 1)

        train = self.loss_history[:len(epochs)]
        val = np.array(self.val_loss_history)

        # ---- FIX 2: correct overfitting definition ----
        overfit_epoch = np.argmin(val) + 1
        overfit_value = val[overfit_epoch - 1]

        plt.figure()

        plt.plot(epochs, train, label="Train Loss")
        plt.plot(epochs, val, label="Validation Loss")

        plt.scatter(
            overfit_epoch,
            overfit_value,
            color="red",
            s=80,
            label="Best Generalization Point"
        )

        plt.axvline(
            x=overfit_epoch,
            linestyle="--",
            color="red"
        )

        plt.title("Train vs Validation Loss (Overfitting Point)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def save_buffer(self, file_path: str):
        torch.save(self.trajectory, file_path)
        print(f"Saved trajectory → {file_path} ({len(self.trajectory)})")

    def load_buffer(self, file_path: str):
        self.trajectory = torch.load(file_path)
        print(f"Loaded trajectory ← {file_path} ({len(self.trajectory)})")

    def plot_loss_and_validation(self, optimal_epoch: int = None, model_type: str = "", smooth_window: int = 10):
        epochs     = np.arange(1, len(self.loss_history) + 1)
        train_loss = np.array(self.loss_history)
        val_loss   = np.array(self.val_loss_history)

        # Compute smoothed val loss
        if len(val_loss) >= smooth_window:
            smoothed_val    = np.convolve(val_loss, np.ones(smooth_window)/smooth_window, mode='valid')
            smoothed_epochs = np.arange(smooth_window, len(val_loss) + 1)
        else:
            smoothed_val    = None
            smoothed_epochs = None

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, alpha=0.4, linestyle="--", label="Val Loss (raw)")

        # ── Model-specific curve + annotation ────────────────────────────────────
        if model_type == 'LSTM':
            # Show smoothed val — minimum is the key signal
            if smoothed_val is not None:
                plt.plot(smoothed_epochs, smoothed_val, label=f"Val Loss (smoothed w={smooth_window})")
            marker_label   = f"Min smoothed val loss (epoch {optimal_epoch})"
            strategy_text  = "Strategy: min smoothed val loss"

        elif model_type == 'DLinear':
            # Show smoothed val — convergence (slope=0) is the key signal
            if smoothed_val is not None:
                plt.plot(smoothed_epochs, smoothed_val, label=f"Val Loss (smoothed w={smooth_window})")
            marker_label   = f"Val loss convergence (epoch {optimal_epoch})"
            strategy_text  = "Strategy: val loss convergence (slope almost 0)"

        elif model_type == 'MLP':
            # Show wider smoothed val — moderate noise needs wider window
            if len(val_loss) >= 15:
                s_wide        = np.convolve(val_loss, np.ones(15)/15, mode='valid')
                s_wide_epochs = np.arange(15, len(val_loss) + 1)
                plt.plot(s_wide_epochs, s_wide, label="Val Loss (smoothed w=15)")
            marker_label   = f"Min smoothed val loss (epoch {optimal_epoch})"
            strategy_text  = "Strategy: min smoothed val loss (window=15)"

        elif model_type == 'CNN':
            # Show smoothed val + gap curve — three conditions needed
            if smoothed_val is not None:
                plt.plot(smoothed_epochs, smoothed_val, label=f"Val Loss (smoothed w={smooth_window})")

            # Plot the gap as a secondary signal
            gap = val_loss - train_loss
            if len(gap) >= smooth_window:
                smoothed_gap    = np.convolve(gap, np.ones(smooth_window)/smooth_window, mode='valid')
                plt.plot(smoothed_epochs, smoothed_gap, linestyle=":", label="Gap (smoothed)", color="purple")

            marker_label   = f"Overfit point - train down + val up + gap widening (epoch {optimal_epoch})"
            strategy_text  = "Strategy: three-condition check"

        else:
            if smoothed_val is not None:
                plt.plot(smoothed_epochs, smoothed_val, label=f"Val Loss (smoothed w={smooth_window})")
            marker_label  = f"Optimal epoch ({optimal_epoch})"
            strategy_text = "Strategy: min smoothed val loss (fallback)"

        # ── Mark the optimal epoch on the plot ───────────────────────────────────
        if optimal_epoch is not None and optimal_epoch <= len(val_loss):
            plt.axvline(x=optimal_epoch, linestyle="--", color="red", label=marker_label)
            plt.scatter(optimal_epoch, val_loss[optimal_epoch - 1], color="red", s=80, zorder=5)

        # ── Strategy annotation box ───────────────────────────────────────────────
        plt.text(
            0.02, 0.97, strategy_text,
            transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

        plt.title(f"Train vs Validation Loss -- {model_type}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def find_overfitting_point(self):
        if len(self.val_loss_history) == 0:
            print("No validation loss recorded.")
            return None

        val = np.array(self.val_loss_history)
        
        # Method 1: epoch of minimum validation loss
        best_epoch = int(np.argmin(val)) + 1
        best_val   = val[best_epoch - 1]

        # Method 2: where the gap starts growing consistently (5-epoch window)
        train = np.array(self.loss_history)
        gap   = val - train
        gap_growth_epoch = None
        window = 5
        for i in range(window, len(gap)):
            if all(gap[i - j] < gap[i - j + 1] for j in range(window, 0, -1)):
                gap_growth_epoch = i - window + 2  # convert to 1-indexed epoch
                break

        print(f"Best generalization epoch (min val loss): {best_epoch}  |  Val Loss: {best_val:.6f}")
        if gap_growth_epoch:
            print(f"Gap started growing consistently at epoch: {gap_growth_epoch}")

        return best_epoch
    
    def find_optimal_epoch(self, model_type: str, smooth_window: int = 10) -> int:
        train = np.array(self.loss_history)
        val   = np.array(self.val_loss_history)
        gap   = val - train

        def smooth(x, w=smooth_window):
            if len(x) < w:
                return x
            return np.convolve(x, np.ones(w)/w, mode='valid')

        s_train = smooth(train)
        s_val   = smooth(val)
        s_gap   = smooth(gap)

        # ── LSTM: smooth min val loss ─────────────────────────────────────────
        if model_type == 'LSTM':
            best_epoch = int(np.argmin(s_val)) + smooth_window
            reason     = "min smoothed val loss"

        # ── DLinear: convergence (slope → 0) ─────────────────────────────────
        elif model_type == 'DLinear':
            slopes     = np.abs(np.diff(s_val))
            best_epoch = smooth_window  # fallback
            for i in range(5, len(slopes) - 5):
                if np.mean(slopes[i:i+5]) < 1e-4:
                    best_epoch = i + smooth_window
                    break
            reason = "val loss convergence (slope almost 0)"

        # ── MLP: smoothed min val loss ────────────────────────────────────────
        elif model_type == 'MLP':
            s_val_wide = smooth(val, w=15)
            best_epoch = int(np.argmin(s_val_wide)) + 15
            reason     = "min smoothed val loss (window=15)"

        # ── CNN: three-condition check ────────────────────────────────────────
        elif model_type == 'CNN':
            d_train      = np.diff(s_train)
            d_val        = np.diff(s_val)
            d_gap        = np.diff(s_gap)
            consecutive  = 5
            best_epoch   = int(np.argmin(s_val)) + smooth_window  # fallback

            for i in range(len(d_val) - consecutive):
                cond1 = np.all(d_train[i:i+consecutive] < 0)        # train dropping
                cond2 = np.all(d_val[i:i+consecutive]   > 0)        # val rising
                cond3 = np.mean(d_gap[i:i+consecutive]) > 1e-4      # gap widening

                if cond1 and cond2 and cond3:
                    best_epoch = i + smooth_window
                    break

            reason = "three-condition (train down + val up + gap widening)"

        else:
            best_epoch = int(np.argmin(s_val)) + smooth_window
            reason     = "fallback: min smoothed val loss"

        print("=" * 55)
        print(f"  Model         : {model_type}")
        print(f"  Strategy      : {reason}")
        print(f"  Optimal epoch : {best_epoch}")
        print(f"  Train loss    : {train[best_epoch-1]:.6f}")
        print(f"  Val loss      : {val[best_epoch-1]:.6f}")
        print(f"  Gap           : {gap[best_epoch-1]:.6f}")
        print("=" * 55)

        return best_epoch