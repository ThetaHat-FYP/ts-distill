"""
MTT Walking Skeleton Package
Minimal end-to-end implementation for testing the ThetaHat pipeline.
"""

from .mock_components import (
    SimpleLSTM,
    SimpleRecorder,
    SimpleEvaluator,
    SimpleTrainer,
    SimpleCallback,
    MSEMatcher,
    RealSampleInitializer
)

from .mtt_distiller import MTTDistiller

__all__ = [
    'SimpleLSTM',
    'SimpleRecorder',
    'SimpleEvaluator',
    'SimpleTrainer',
    'SimpleCallback',
    'MSEMatcher',
    'RealSampleInitializer',
    'MTTDistiller'
]
