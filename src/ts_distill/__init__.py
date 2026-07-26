"""
ts_distill — modular time-series dataset distillation
=====================================================
Compress a large time-series training set into a small synthetic one that still
trains a forecaster to comparable accuracy, then repair the temporal properties
that distillation destroys.

Pipeline
--------
    1. Load + split      CSVDataLoader, get_data_splits, make_windows
    2. Train an expert   Trainer + SimpleRecorder  (records the trajectory)
    3. Initialise        RandomSample / Geometry / UncertaintySample
    4. Distil            MTTDistiller  (or PhaseAwareMTTDistiller)
    5. Post-fix          FFTAmplitudePostFix  (restores the amplitude spectrum)
    6. Mix + evaluate    BaseHybridEvaluator + find_optimal_ratio
    7. Measure fidelity  MetricAggregator  (spectrum, ACF, trend, variance, ...)

Quick start
-----------
    import torch
    from ts_distill import (
        MTTDistiller, RandomSampleInitializer, MSEMatcher, DEFAULT_CONFIG,
    )

    torch.manual_seed(42)
    initializer = RandomSampleInitializer()
    synthetic   = distiller.distill(
        synthetic_init = initializer.initialize_sequence(raw_train, 384),
        n_steps        = 300,
        val_data       = val_windows,
    )

See ``example/demo_lib.py`` for a complete runnable pipeline.

Determinism
-----------
Every random draw comes from the global torch RNG, so results are reproducible
under ``torch.manual_seed(...)`` — but only if the ORDER of random calls is
unchanged. Anything that iterates a DataLoader consumes RNG and shifts every
later draw. When inserting such a step, wrap it in
``torch.get_rng_state()`` / ``torch.set_rng_state()`` to leave the stream intact.
"""

__version__ = '0.1.0'

# ── Logging ──────────────────────────────────────────────────────────────────
# Importing ts_distill prints nothing. Call configure_logging() to see progress.
from ts_distill._logging import configure_logging, get_logger

# ── Configuration ────────────────────────────────────────────────────────────
from ts_distill.config import (
    DEFAULT_CONFIG,
    DATASET_CONFIGS,
    DATASET_PERIODS,
    MODEL_CONFIGS,
    resolve_csv_path,
    phase_boundary_config,
)

# ── Data ─────────────────────────────────────────────────────────────────────
from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader import BaseDataLoader, CSVDataLoader
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader

# ── Models ───────────────────────────────────────────────────────────────────
from ts_distill.models import BaseForecaster, create_model

# ── Training ─────────────────────────────────────────────────────────────────
from ts_distill.trainer import (
    Trainer,
    BaseCallback,
    SimpleCallback,
    ValLossRecorderCallback,
)

# ── Expert trajectory ────────────────────────────────────────────────────────
from ts_distill.trajectory import (
    SimpleRecorder,
    MSEMatcher,
    ValLossPlateauDetector,
)

# ── Distillation ─────────────────────────────────────────────────────────────
from ts_distill.distillation_core import (
    BaseDistiller,
    MTTDistiller,
    PhaseAwareMTTDistiller,
    BaseInitializer,
    RandomSampleInitializer,
    GeometrySequenceInitializer,
    UncertaintySampleInitializer,
)

# ── Post-processing ──────────────────────────────────────────────────────────
from ts_distill.post_processing import FFTAmplitudePostFix

# ── Evaluation ───────────────────────────────────────────────────────────────
from ts_distill.evaluation import BaseEvaluator, Evaluator
from ts_distill.evaluation.hybrid_evaluation import (
    BaseHybridEvaluator,
    RandomAnchorSelector,
    find_optimal_ratio,
)

# ── Temporal fidelity metrics ────────────────────────────────────────────────
from ts_distill.metrics import MetricAggregator

__all__ = [
    '__version__',
    # logging
    'configure_logging', 'get_logger',
    # config
    'DEFAULT_CONFIG', 'DATASET_CONFIGS', 'DATASET_PERIODS', 'MODEL_CONFIGS',
    'resolve_csv_path', 'phase_boundary_config',
    # data
    'get_data_splits', 'make_windows',
    'BaseDataLoader', 'CSVDataLoader', 'MiniBatchLoader',
    # models
    'BaseForecaster', 'create_model',
    # training
    'Trainer', 'BaseCallback', 'SimpleCallback', 'ValLossRecorderCallback',
    # trajectory
    'SimpleRecorder', 'MSEMatcher', 'ValLossPlateauDetector',
    # distillation
    'BaseDistiller', 'MTTDistiller', 'PhaseAwareMTTDistiller',
    'BaseInitializer', 'RandomSampleInitializer',
    'GeometrySequenceInitializer', 'UncertaintySampleInitializer',
    # post-processing
    'FFTAmplitudePostFix',
    # evaluation
    'BaseEvaluator', 'Evaluator', 'BaseHybridEvaluator',
    'RandomAnchorSelector', 'find_optimal_ratio',
    # metrics
    'MetricAggregator',
]
