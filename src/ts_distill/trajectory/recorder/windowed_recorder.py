"""
Windowed Recorder
-----------------
A read-only BaseTrajectoryRecorder view that exposes only the subset of
checkpoints selected by a PhaseWindowSelector.

How it fits into the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  Train the expert with a SimpleRecorder → full trajectory.
2.  Build a PhaseWindowSelector for the desired epoch range.
3.  Construct a WindowedRecorder from the full recorder + selector.
4.  Pass the WindowedRecorder to MTTDistiller as expert_recorder.

MTTDistiller calls exactly one method on expert_recorder:
    start_ckpt, end_ckpt = expert_recorder.sample_checkpoint_pair(step_gap=k)

WindowedRecorder satisfies this contract while restricting sampling to the
phase window, so the outer distillation loop sees only checkpoints from that
window without any changes to MTTDistiller itself.

Because this is a read-only projection of an existing trajectory, the
recording methods (record_checkpoint, on_train_begin, on_epoch_end) are
intentional no-ops — the WindowedRecorder is never attached to a Trainer.
"""

import random
from typing import Dict, List, Optional, Tuple

import torch

from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder
from ts_distill.trajectory.selector.phase_window_selector import PhaseWindowSelector


class WindowedRecorder(BaseTrajectoryRecorder):
    """
    Read-only trajectory recorder scoped to a phase window.

    Wraps any BaseTrajectoryRecorder (typically a SimpleRecorder) and exposes
    only the subset of checkpoints that PhaseWindowSelector keeps.  Provides
    the same sample_checkpoint_pair() interface that MTTDistiller uses, so it
    can replace a full recorder without any changes to the distiller.

    Args:
        source_recorder (BaseTrajectoryRecorder): The fully-recorded expert
            trajectory to project.
        selector (PhaseWindowSelector): Determines which checkpoints to keep.

    Raises:
        ValueError: If the filtered trajectory contains fewer than 2
                    checkpoints (cannot form any pair).

    Example:
        recorder  = SimpleRecorder(record_every=1)
        # ... expert training ...
        selector  = PhaseWindowSelector(start_epoch=33, end_epoch=48)
        windowed  = WindowedRecorder(recorder, selector)
        distiller = MTTDistiller(..., expert_recorder=windowed, ...)
    """

    def __init__(
        self,
        source_recorder: BaseTrajectoryRecorder,
        selector: PhaseWindowSelector,
    ):
        self._selector   = selector
        self._trajectory: List[Dict] = selector.filter_trajectory(
            source_recorder.get_trajectory()
        )

    # ── Core property ─────────────────────────────────────────────────────────

    @property
    def n_checkpoints(self) -> int:
        """Number of checkpoints in the windowed trajectory."""
        return len(self._trajectory)

    # ── BaseTrajectoryRecorder — read interface ───────────────────────────────

    def get_trajectory(self) -> List[Dict]:
        """Return the filtered list of checkpoint dicts."""
        return self._trajectory

    def sample_checkpoint_pair(self, step_gap: int = 5) -> Tuple[Dict, Dict]:
        """
        Sample a consecutive pair (θ_start, θ_target) from the windowed trajectory.

        Mirrors SimpleRecorder.sample_checkpoint_pair() so MTTDistiller can
        use a WindowedRecorder as a drop-in replacement.

        Args:
            step_gap (int): Number of checkpoint positions between the pair.
                            Capped to len(trajectory) - 1 if the window is
                            shorter than step_gap checkpoints.

        Returns:
            (start_checkpoint, end_checkpoint) — each a dict with 'step' and
            'weights' keys.

        Raises:
            ValueError: If fewer than 2 checkpoints are available.
        """
        n = len(self._trajectory)
        if n < 2:
            raise ValueError(
                f"WindowedRecorder ({self._selector.label}) contains only "
                f"{n} checkpoint(s) — need at least 2 to form a pair. "
                f"Widen the epoch window or increase expert_epochs."
            )
        actual_gap = min(step_gap, n - 1)
        start_idx  = random.randint(0, n - actual_gap - 1)
        return self._trajectory[start_idx], self._trajectory[start_idx + actual_gap]

    # ── BaseTrajectoryRecorder — write interface (no-ops) ─────────────────────
    # WindowedRecorder is a read-only projection.  Attaching it to a Trainer
    # would silently do nothing, which is the safest default.

    def record_checkpoint(self, model, step: int):
        """No-op — WindowedRecorder does not record new checkpoints."""
        pass

    def on_train_begin(self, model, **kwargs):
        """No-op — WindowedRecorder is not attached to a training loop."""
        pass

    def on_epoch_end(self, model, epoch: int, loss: float = None, **kwargs):
        """No-op — WindowedRecorder is not attached to a training loop."""
        pass

    def on_train_end(self, model, **kwargs):
        """No-op."""
        pass

    # ── Disk persistence ──────────────────────────────────────────────────────

    def save_buffer(self, file_path: str):
        """Persist the windowed trajectory slice to disk."""
        torch.save(self._trajectory, file_path)
        print(f"   Windowed trajectory saved → {file_path} "
              f"({len(self._trajectory)} checkpoints, {self._selector.label})")

    def load_buffer(self, file_path: str):
        """Replace the in-memory trajectory with a previously saved slice."""
        self._trajectory = torch.load(file_path)
        print(f"   Windowed trajectory loaded ← {file_path} "
              f"({len(self._trajectory)} checkpoints)")

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"WindowedRecorder(selector={self._selector!r}, "
            f"n_checkpoints={self.n_checkpoints})"
        )
