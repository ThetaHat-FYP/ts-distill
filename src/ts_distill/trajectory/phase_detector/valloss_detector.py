"""
Plateau detection — finds T+, the epoch where the expert stops making progress.

`PhaseAwareMTTDistiller` matches parameters before T+ and predictions after it,
so this boundary decides which objective applies to each trajectory segment.

How T+ is chosen
----------------
Smooth the validation curve, then walk forward from the burn-in point. Whenever
a value beats the running best by at least `min_delta_frac` (relative), that
becomes the new best. After `patience` consecutive epochs fail to clear that
bar, declare a plateau and return the BEST epoch — not the epoch where the
patience ran out.

The three knobs
---------------
  smoothing_window  noise suppression; raise it for jagged curves (CNN)
  min_delta_frac    what counts as improvement, relative to current best
  burn_in_epochs    epochs skipped entirely at the start

`burn_in_epochs` exists because training opens with a steep drop that then
naturally slows. Without a burn-in the detector reads that slowdown as a
plateau and puts T+ in the first few epochs, which is far too early.

Settings are tuned PER EXPERT ARCHITECTURE and live in `ts_distill.config`
(`phase_boundary_config`), because each architecture's curve has a different
shape: one setting that finds a good mid-trajectory boundary for DLinear pushes
CNN and MLP so late that phase-aware matching degenerates into plain parameter
matching.

Smoothing is CAUSAL (backward-looking) on purpose. A centred window would let
later epochs influence earlier smoothed values and shift the reported boundary
earlier than it truly occurred.

Never returns None: if no sustained plateau is found it falls back to the
overall minimum after burn-in, so callers always get a usable boundary.
"""

from typing import Dict, List, Optional
import numpy as np
from ts_distill.trajectory.phase_detector.base import BasePhaseDetector

class ValLossPlateauDetector(BasePhaseDetector):
    """
    Locate T+ as the best smoothed val-loss epoch before a sustained plateau.

    Args:
        smoothing_window (int): Causal moving-average width. Larger = more noise
                                suppression, at the cost of temporal precision.
        patience (int):         Consecutive non-improving epochs that confirm a
                                plateau. Should exceed smoothing_window.
        min_delta_frac (float): Relative improvement required to count, e.g.
                                0.01 = must beat the best by 1%.
        burn_in_epochs (int):   Epochs skipped before detection starts, so the
                                initial steep drop is not read as a plateau.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        patience: int = 10,          # Increased patience (should be > smoothing_window)
        min_delta_frac: float = 0.01,
        burn_in_epochs: int = 15,    # Grace period before checking for plateaus
    ) -> None:
        self.smoothing_window = smoothing_window
        self.patience = patience
        self.min_delta_frac = min_delta_frac
        self.burn_in_epochs = burn_in_epochs

    @staticmethod
    def _smooth(values: List[float], window: int) -> np.ndarray:
        """Causal (backward-looking) moving average to preserve true temporal boundaries."""
        arr = np.array(values, dtype=np.float64)
        if window <= 1:
            return arr

        # Simple moving average using causal window
        kernel = np.ones(window) / window
        # Pad left to avoid shifting indices
        padded = np.pad(arr, (window - 1, 0), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def detect(self, val_losses: List[float], epochs: Optional[List[int]] = None) -> Dict:
        """
        Find the phase boundary in a validation-loss curve.

        Args:
            val_losses (List[float]): Per-epoch validation loss, in order.
            epochs (List[int] | None): Epoch labels. Defaults to 1..len.

        Returns:
            dict: boundary_epoch, boundary_idx, best_val_loss, final_val_loss,
            and the smoothed curve. `boundary_epoch` is always populated — it
            falls back to the post-burn-in minimum when no plateau is found.
        """
        if epochs is None:
            epochs = list(range(1, len(val_losses) + 1))

        val_smooth = self._smooth(val_losses, self.smoothing_window)

        # 1. Start tracking ONLY after the burn-in period
        start_idx = min(self.burn_in_epochs, len(val_smooth) - 1)
        
        best_val = val_smooth[start_idx]
        best_idx = start_idx
        no_improve = 0
        boundary_idx = None

        # 2. Loop starts from the epoch after burn-in
        for i in range(start_idx + 1, len(val_smooth)):
            required_improvement = best_val * self.min_delta_frac
            
            if val_smooth[i] < (best_val - required_improvement):
                best_val = val_smooth[i]
                best_idx = i
                no_improve = 0
            else:
                no_improve += 1
                
                # We no longer need to check i >= burn_in_epochs here 
                # because the loop itself started after the burn-in.
                if no_improve >= self.patience:
                    boundary_idx = best_idx
                    break

        # Fallback: If no sustained plateau was hit, pick the overall minimum index
        # (also restricted to occur after the burn-in)
        if boundary_idx is None:
            boundary_idx = start_idx + int(np.argmin(val_smooth[start_idx:]))

        boundary_epoch = epochs[boundary_idx]

        return {
            "boundary_epoch": boundary_epoch,
            "boundary_idx": boundary_idx,
            "best_val_loss": float(val_smooth[boundary_idx]),
            "final_val_loss": float(val_smooth[-1]),
            "val_loss_smooth": val_smooth.tolist(), 
        }