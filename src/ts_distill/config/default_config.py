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
