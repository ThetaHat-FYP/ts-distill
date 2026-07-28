from typing import Dict, List, Optional
import numpy as np
from ts_distill.trajectory.phase_detector.base import BasePhaseDetector

class ValLossPlateauDetector(BasePhaseDetector):
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