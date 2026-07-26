"""
Expert trajectories
===================
Trajectory matching needs the expert's weights over time, not just its final
state. This package records that path, measures distance between points on it,
and locates the phase boundary where the expert stops making structural
progress.

Sub-packages
------------
recorder       : Capture expert weights at each epoch (SimpleRecorder).
matcher        : Distance between student and expert parameters (MSEMatcher).
phase_detector : Locate T+, the epoch where the val-loss curve plateaus.
"""

from ts_distill.trajectory.recorder import BaseTrajectoryRecorder, SimpleRecorder
from ts_distill.trajectory.matcher import BaseTrajectoryMatcher, MSEMatcher
from ts_distill.trajectory.phase_detector import (
    BasePhaseDetector,
    ValLossPlateauDetector,
)

__all__ = [
    'BaseTrajectoryRecorder',
    'SimpleRecorder',
    'BaseTrajectoryMatcher',
    'MSEMatcher',
    'BasePhaseDetector',
    'ValLossPlateauDetector',
]
