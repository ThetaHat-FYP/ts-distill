"""
Default Config
==============
Central, importable configuration for the ts_distill pipeline.

These values are lifted directly from example/experiments/experiment_matrix.py
so that library users get the same, tested defaults without copying them.

Contents
--------
DEFAULT_CONFIG   : distillation + evaluation hyperparameters (one flat dict).
DATASET_CONFIGS  : per-dataset CSV path and train/val/test split definition.
DATASET_PERIODS  : dominant seasonal period per dataset (for ACF / STL / metrics).
MODEL_CONFIGS    : per-model constructor kwargs.

CSV paths are relative to the project root, so run scripts from the project
root (the folder that contains `example/` and `src/`), exactly like the
experiment scripts.
"""

# =============================================================================
# DISTILLATION + EVALUATION HYPERPARAMETERS
# =============================================================================

DEFAULT_CONFIG = {
    'in_features':            7,     # multivariate channels (ETT family)
    'seq_len':               96,     # look-back window
    'pred_len':              96,     # forecast horizon
    'expert_epochs':         80,
    'expert_lr':           0.01,
    'expert_momentum':      0.9,
    'n_distill_steps':      300,
    'n_synthetic':          384,     # synthetic sequence length (timesteps)
    'synthetic_lr':         0.01,
    'student_lr':           0.01,
    'student_steps':         20,
    'snapshot_student_steps': 50,
    'batch_size':            64,
    'trajectory_gap':         5,
    # Evaluation protocol (probe training on real/synthetic/hybrid data).
    'eval_max_epochs':      300,     # hard cap; early stopping fires well before
    'eval_lr':            0.001,
    'eval_batch_size':       32,
    'early_stop_patience':   10,
    # ACF metric setting
    'acf_n_lags':           100,
}


# =============================================================================
# DATASET CONFIGURATION  (benchmark splits, reproduced from experiment_matrix)
# =============================================================================

DATASET_CONFIGS = {
    'ETTh1': {
        'csv_path':    'example/ETTh1.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 24 - 96,       12 * 30 * 24 + 4 * 30 * 24 - 96],
        'border2s':    [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        'in_features': 7,
    },
    'ETTh2': {
        'csv_path':    'example/ETTh2.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 24 - 96,       12 * 30 * 24 + 4 * 30 * 24 - 96],
        'border2s':    [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        'in_features': 7,
    },
    'ETTm1': {
        'csv_path':    'example/ETTm1.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 96 - 96,       12 * 30 * 96 + 4 * 30 * 96 - 96],
        'border2s':    [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        'in_features': 7,
    },
    'ETTm2': {
        'csv_path':    'example/ETTm2.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 96 - 96,       12 * 30 * 96 + 4 * 30 * 96 - 96],
        'border2s':    [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        'in_features': 7,
    },
    'exchange_rate': {
        'csv_path':    'example/exchange_rate.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 8,
    },
    'weather': {
        'csv_path':    'example/weather.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 21,
    },
}


# =============================================================================
# DOMINANT SEASONAL PERIOD (timesteps) per dataset
# =============================================================================

DATASET_PERIODS = {
    'ETTh1': 24,   'ETTh2': 24,     # hourly  -> one day = 24
    'ETTm1': 96,   'ETTm2': 96,     # 15-min  -> one day = 96
    'exchange_rate': 5,             # daily   -> one trading week
    'weather': 144,                 # 10-min  -> one day = 144
}


# =============================================================================
# MODEL CONFIGURATION  (per-model constructor kwargs)
# =============================================================================

MODEL_CONFIGS = {
    'DLinear': {'individual': False},
    'LSTM':    {'hidden_dim': 64, 'num_layer': 1},
    'MLP':     {},
    'CNN':     {},
}


# =============================================================================
# PHASE-BOUNDARY DETECTION  (per expert architecture)
# =============================================================================
# Phase-aware distillation matches parameters early in the expert's trajectory
# and predictions late; T+ is the boundary between those phases, detected from
# the expert's validation-loss curve by ValLossPlateauDetector.
#
#   T+ = epoch of the best smoothed val loss before `patience` consecutive
#        epochs each fail to improve on the best-so-far by `min_delta_frac`
#        (relative). `burn_in_epochs` skips the chaotic start of training, so
#        the initial rapid drop cannot be mistaken for a plateau.
#
# These are PER-ARCHITECTURE on purpose. Each architecture's val-loss curve has
# a different shape, so a single setting that finds a good mid-trajectory
# boundary for one (DLinear) pushes others (CNN, MLP) so late that phase-aware
# matching degenerates into plain parameter matching. Each entry is tuned so T+
# lands in that architecture's genuine early->late transition.
#
# Use `phase_boundary_config(arch)` rather than indexing this dict directly, so
# unknown architectures fall back to PHASE_BOUNDARY_CONFIG_DEFAULT.

PHASE_BOUNDARY_CONFIGS = {
    'DLinear': {
        'smoothing_window': 5,      # Short window is fine, loss is smooth
        'patience':         15,     # Wide enough to confirm a flatline
        'min_delta_frac':   0.001,  # 0.1% - strict, DLinear stops improving completely
        'burn_in_epochs':   0,      # No early chaos to ignore
    },
    'CNN': {
        'smoothing_window': 10,     # Double smoothing to absorb gradient noise
        'patience':         20,     # Longer, to survive temporary plateaus
        'min_delta_frac':   0.01,   # 1.0% - fires when rapid learning stops, not all learning
        'burn_in_epochs':   15,     # Ignore the chaotic first 15 epochs entirely
    },
    'MLP': {
        'smoothing_window': 5,      # MLP curves are relatively smooth
        'patience':         15,     # Standard patience
        'min_delta_frac':   0.015,  # 1.5% - high threshold forces an earlier phase break
        'burn_in_epochs':   5,      # Short grace period for the initial drop
    },
}

# Fallback for any architecture without a dedicated entry above (e.g. LSTM).
PHASE_BOUNDARY_CONFIG_DEFAULT = {
    'smoothing_window': 5,
    'patience':         15,
    'min_delta_frac':   0.001,
    'burn_in_epochs':   0,
}


def phase_boundary_config(arch: str) -> dict:
    """
    Return the ValLossPlateauDetector settings tuned for an expert architecture.

    Args:
        arch (str): Expert architecture name, e.g. 'DLinear', 'CNN', 'MLP'.

    Returns:
        dict: kwargs for ValLossPlateauDetector. A copy, so callers can tweak a
        single field without mutating the shared defaults. Architectures with no
        dedicated entry get PHASE_BOUNDARY_CONFIG_DEFAULT.
    """
    return dict(PHASE_BOUNDARY_CONFIGS.get(arch, PHASE_BOUNDARY_CONFIG_DEFAULT))
