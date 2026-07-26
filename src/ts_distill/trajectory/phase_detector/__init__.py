"""
Phase detectors
===============
An expert's training splits into an early phase, where it learns structure, and
a late phase, where it mostly refines. `PhaseAwareMTTDistiller` matches
parameters in the early phase and predictions in the late phase; the boundary
T+ between them comes from a detector here.

ValLossPlateauDetector picks T+ as the epoch of the best smoothed validation
loss before `patience` consecutive epochs fail to improve on it. When no plateau
is found it returns None, and phase-aware distillation degrades gracefully to
plain parameter matching.
"""

from ts_distill.trajectory.phase_detector.base import BasePhaseDetector
from ts_distill.trajectory.phase_detector.valloss_detector import ValLossPlateauDetector

__all__ = ['BasePhaseDetector', 'ValLossPlateauDetector']
