"""
Default configuration for the ts_distill pipeline.

Exposes the same hyperparameters, dataset splits, dataset periods, and model
kwargs that the experiment scripts use, so that end-user code can import a
single ready-made config instead of redefining everything.
"""

from ts_distill.config.default_config import (
    DEFAULT_CONFIG,
    DATASET_CONFIGS,
    DATASET_PERIODS,
    MODEL_CONFIGS,
    resolve_csv_path,
    PHASE_BOUNDARY_CONFIGS,
    PHASE_BOUNDARY_CONFIG_DEFAULT,
    phase_boundary_config,
)

__all__ = [
    'DEFAULT_CONFIG',
    'DATASET_CONFIGS',
    'DATASET_PERIODS',
    'MODEL_CONFIGS',
    'PHASE_BOUNDARY_CONFIGS',
    'PHASE_BOUNDARY_CONFIG_DEFAULT',
    'phase_boundary_config',
    'resolve_csv_path',
]
