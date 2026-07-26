"""
Trajectory matchers
===================
Distance between the student's parameters after training on synthetic data and
the expert's parameters at the corresponding point on its real-data trajectory.
This distance is the loss that distillation minimises.
"""

from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher

__all__ = ['BaseTrajectoryMatcher', 'MSEMatcher']
