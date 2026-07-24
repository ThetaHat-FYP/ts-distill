"""
Validation-Loss Plateau Phase Detector
=======================================
Detects the expert-trajectory phase boundary T+ from the expert's
per-epoch validation-loss curve: "When does the expert stop IMPROVING on
real validation data?"

T+ = the epoch of the best (lowest) smoothed val loss seen before a
sustained plateau begins (`patience` consecutive epochs each failing to
improve on the best-so-far by at least `min_delta_frac`, relative).

Moved out of example/experiments/h_detect_phase_boundary_valloss.py so the
same detection logic can be shared between the standalone boundary-sweep
script and the on-the-fly detection used inside experiment_matrix.py.
"""

from typing import Dict, List, Optional

import numpy as np

from ts_distill.trajectory.phase_detector.base import BasePhaseDetector


class ValLossPlateauDetector(BasePhaseDetector):
    """
    Args:
        smoothing_window (int): Rolling-average window applied to the raw
            val-loss sequence before plateau detection.
        patience (int): Number of consecutive non-improving epochs that
            defines a "plateau".
        min_delta_frac (float): An epoch only counts as "improvement" if it
            is at least this much better (relative) than the best-so-far
            smoothed val loss.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        patience: int = 5,
        min_delta_frac: float = 0.01,
    ) -> None:
        self.smoothing_window = smoothing_window
        self.patience = patience
        self.min_delta_frac = min_delta_frac

    @staticmethod
    def _smooth(values: List[float], window: int) -> np.ndarray:
        """Centred rolling average. Edge values use available data only."""
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        out = np.empty(n)
        half = window // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = arr[lo:hi].mean()
        return out

    def detect(self, val_losses: List[float], epochs: Optional[List[int]] = None) -> Dict:
        """
        Args:
            val_losses: Per-epoch validation loss (index i = epoch i+1).
            epochs: Optional 1-indexed epoch labels matching val_losses.
                Defaults to range(1, len(val_losses) + 1).

        Returns:
            dict with keys: boundary_epoch, boundary_idx, best_val_loss,
            final_val_loss, val_loss_smooth.
        """
        if epochs is None:
            epochs = list(range(1, len(val_losses) + 1))

        val_smooth = self._smooth(val_losses, self.smoothing_window)

        best_val = val_smooth[0]
        best_idx = 0
        no_improve = 0
        boundary_idx = None

        for i in range(1, len(val_smooth)):
            if val_smooth[i] < best_val * (1.0 - self.min_delta_frac):
                best_val = val_smooth[i]
                best_idx = i
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    boundary_idx = best_idx
                    break

        boundary_epoch = epochs[boundary_idx] if boundary_idx is not None else None

        return {
            "boundary_epoch": boundary_epoch,
            "boundary_idx": boundary_idx,
            "best_val_loss": float(best_val),
            "final_val_loss": float(val_smooth[-1]),
            "val_loss_smooth": val_smooth,
        }
