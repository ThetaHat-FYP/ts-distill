"""
Phase Window Selector
---------------------
Restricts a recorded expert trajectory to checkpoints that fall within a
specific epoch window.  Used by the Phase-Aware Trajectory Matching experiment
(Thread 3) to test whether the phase of expert training from which checkpoint
pairs are sampled affects distillation quality (Hypothesis H1).

How it fits into the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  Train the expert for N epochs with a SimpleRecorder — this gives you a
    full trajectory of N+1 checkpoints (step 0 … step N).
2.  Construct a PhaseWindowSelector with the desired epoch range.
3.  Call selector.filter_trajectory(recorder.get_trajectory()) to get the
    subset of checkpoints for that window.
4.  Pass the filtered trajectory to a WindowedRecorder, which exposes the
    same sample_checkpoint_pair() interface that MTTDistiller expects.

Window labels used in Thread 3 (80-epoch expert, record_every=1):
  W1  epochs  1–16   Rapid:     large gradient steps, fast loss drop
  W2  epochs 17–32   Early-mid: slowing down, stabilising
  W3  epochs 33–48   Middle:    steady refinement, best generalisation
  W4  epochs 49–64   Late-mid:  plateau, small updates
  W5  epochs 65–80   Late:      near/post overfitting point
"""

from typing import Dict, List, Optional

from ts_distill.trajectory.selector.base import BaseTrajectorySelector, BaseTrajectoryMetric


class PhaseWindowSelector(BaseTrajectorySelector):
    """
    Filters a trajectory to checkpoints within a contiguous epoch window.

    This selector implements the intra-trajectory filtering pattern: given a
    single expert trajectory, it returns only the checkpoints whose step number
    (epoch) falls within [start_epoch, end_epoch] inclusive.

    It also satisfies the multi-trajectory select() contract from the base
    class by applying the window filter to the first trajectory in the pool
    and returning it as a single-element list.

    Args:
        start_epoch (int): First epoch to include (inclusive).
        end_epoch   (int): Last epoch to include (inclusive).
        metric      (BaseTrajectoryMetric | None): Unused for window selection;
                           kept for interface compatibility.

    Example:
        selector   = PhaseWindowSelector(start_epoch=33, end_epoch=48)  # W3 Middle
        w3_ckpts   = selector.filter_trajectory(recorder.get_trajectory())
        windowed   = WindowedRecorder(w3_ckpts, selector)
    """

    def __init__(
        self,
        start_epoch: int,
        end_epoch: int,
        metric: Optional[BaseTrajectoryMetric] = None,
    ):
        super().__init__(metric=metric)
        if start_epoch > end_epoch:
            raise ValueError(
                f"start_epoch ({start_epoch}) must be <= end_epoch ({end_epoch})."
            )
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch

    # ── Primary interface for intra-trajectory filtering ─────────────────────

    def filter_trajectory(self, trajectory: List[Dict]) -> List[Dict]:
        """
        Return only the checkpoints whose step falls within
        [self.start_epoch, self.end_epoch] inclusive.

        Args:
            trajectory: Ordered list of checkpoint dicts, each with 'step'
                        (int) and 'weights' (dict of param tensors) keys.

        Returns:
            Filtered list.  May be empty if no checkpoints fall in the window.
        """
        return [
            ckpt for ckpt in trajectory
            if self.start_epoch <= ckpt['step'] <= self.end_epoch
        ]

    # ── BaseTrajectorySelector contract ──────────────────────────────────────

    def select(self, trajectories: List[List[Dict]], k: int = 1) -> List[List[Dict]]:
        """
        Apply the window filter to each trajectory in the pool and return
        up to k non-empty filtered trajectories.

        Args:
            trajectories: Pool of expert paths to filter.
            k:            Maximum number of filtered trajectories to return.

        Returns:
            List of filtered trajectories (up to k), preserving pool order.
        """
        filtered = [
            self.filter_trajectory(traj)
            for traj in trajectories
        ]
        non_empty = [t for t in filtered if len(t) > 0]
        return non_empty[:k]

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def label(self) -> str:
        """Human-readable window label, e.g. 'epochs 33–48'."""
        return f"epochs {self.start_epoch}–{self.end_epoch}"

    def __repr__(self) -> str:
        return (
            f"PhaseWindowSelector(start_epoch={self.start_epoch}, "
            f"end_epoch={self.end_epoch})"
        )
