"""
MTT Walking Skeleton Package
Minimal end-to-end implementation for testing the ThetaHat pipeline.
"""

from .mock_components import (
    MockDataLoader,
    SimpleLSTM,
    SimpleRecorder,
    SimpleEvaluator,
    MSEMatcher,
    RandomInitializer,
    RealSampleInitializer
)

from .mtt_distiller import MTTDistiller, MTTDistillerSimplified

__all__ = [
    'MockDataLoader',
    'SimpleLSTM',
    'SimpleRecorder',
    'SimpleEvaluator',
    'MSEMatcher',
    'RandomInitializer',
    'RealSampleInitializer',
    'MTTDistiller',
    'MTTDistillerSimplified'
]
