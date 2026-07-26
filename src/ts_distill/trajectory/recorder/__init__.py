"""
Trajectory recorders
====================
Capture the expert's weights during training so distillation can match them.

SimpleRecorder stores a snapshot every `record_every` epochs and can persist the
buffer to disk (`save_buffer` / `load_buffer`) so an expensive expert run is
trained once and reused across distillation experiments.
"""

from ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder

__all__ = ['BaseTrajectoryRecorder', 'SimpleRecorder']
