"""
Simple Recorder
---------------
An in-memory expert trajectory recorder that doubles as a training callback.

How it fits into the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  Pass a SimpleRecorder instance to Trainer.fit() inside the `callbacks` list.
2.  After every `record_every` training epochs the recorder automatically
    saves a snapshot of the model weights (a "checkpoint").
3.  Pass the same recorder to MTTDistiller — it calls sample_checkpoint_pair()
    to fetch random (θ_t, θ_{t+k}) pairs for the trajectory matching loss.

The checkpoints are kept in a plain Python list in RAM.  For very long
expert runs you can persist them with save_buffer() / load_buffer().
"""

import random
from typing import Dict, List, Tuple

import torch

from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder
from ts_distill.trainer.callback.base import BaseCallback

from ts_distill._logging import get_logger

logger = get_logger(__name__)


class SimpleRecorder(BaseTrajectoryRecorder, BaseCallback):
    """
    In-memory trajectory recorder with training-callback hooks.

    Args:
        record_every (int): Save a weight checkpoint every this many epochs.
                            record_every=1 (default) captures every epoch,
                            which gives the distiller the most checkpoints to
                            sample from, at the cost of more memory.
    """

    def __init__(self, record_every: int = 1):
        self.trajectory:   List[Dict] = []
        self.record_every: int        = record_every

    # ── BaseCallback hooks (called automatically by Trainer.fit) ────────────

    def on_train_begin(self, model, **kwargs):
        """
        Reset the trajectory and capture the model's random-init weights.
        Recording the epoch-0 snapshot gives the distiller a valid θ_0 start.
        """
        self.trajectory = []
        self.record_checkpoint(model, step=0)

    def on_epoch_end(self, model, epoch: int, loss: float, **kwargs):
        """Save a checkpoint at the end of every eligible epoch."""
        if (epoch + 1) % self.record_every == 0:
            self.record_checkpoint(model, step=epoch + 1)

    def on_train_end(self, model, **kwargs):
        """Nothing to do — every checkpoint was saved as training progressed."""
        pass

    # ── Checkpoint operations ────────────────────────────────────────────────

    def record_checkpoint(self, model: torch.nn.Module, step: int):
        """
        Deep-copy the model's current state_dict and append it to the trajectory.

        Each checkpoint is a plain dict:
            {'step': int, 'weights': {param_name: cpu_tensor, ...}}
        """
        weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        self.trajectory.append({'step': step, 'weights': weights})

    def get_trajectory(self) -> List[Dict]:
        """Return the full ordered list of recorded checkpoints."""
        return self.trajectory

    def sample_checkpoint_pair(self, step_gap: int = 10) -> Tuple[Dict, Dict]:
        """
        Sample a consecutive checkpoint pair (θ_t, θ_{t+k}) for MTT.

        The MTT objective minimises the distance between where the student ends
        up after k gradient steps on synthetic data and where the expert
        trajectory ends up over the same k real-data steps.

        Args:
            step_gap (int): Number of recorded checkpoints between the pair.
                            If the trajectory is shorter than step_gap,
                            the actual gap is capped to len(trajectory) - 1.

        Returns:
            (start_checkpoint, end_checkpoint) — each a dict with 'step'
            and 'weights' keys.

        Raises:
            ValueError: If fewer than 2 checkpoints have been recorded.
        """
        if len(self.trajectory) < 2:
            raise ValueError(
                "Need at least 2 checkpoints for trajectory matching. "
                "Make sure expert training has run before distillation."
            )

        actual_gap = min(step_gap, len(self.trajectory) - 1)
        start_idx  = random.randint(0, len(self.trajectory) - actual_gap - 1)

        return self.trajectory[start_idx], self.trajectory[start_idx + actual_gap]

    def sample_checkpoint(self) -> Dict:
        """Return one randomly chosen checkpoint from the trajectory."""
        if not self.trajectory:
            raise ValueError("No checkpoints recorded yet.")
        return random.choice(self.trajectory)

    # ── Disk persistence ─────────────────────────────────────────────────────

    def save_buffer(self, file_path: str):
        """
        Persist the full trajectory list to disk using torch.save.
        Useful when expert training is expensive and you want to reuse
        the trajectory across multiple distillation runs.
        """
        torch.save(self.trajectory, file_path)
        logger.info(f"   Trajectory saved -> {file_path}  ({len(self.trajectory)} checkpoints)")

    def load_buffer(self, file_path: str):
        """Load a previously saved trajectory, replacing the current one."""
        self.trajectory = torch.load(file_path)
        logger.info(f"   Trajectory loaded <- {file_path}  ({len(self.trajectory)} checkpoints)")
