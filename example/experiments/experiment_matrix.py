"""
Experiment Matrix Runner
========================
Runs the full MTT distillation pipeline for every (dataset × model) combination
defined in ACTIVE_DATASETS and ACTIVE_MODELS, then records temporal statistical
metrics in two CSV files:

  results/metrics_main.csv         — one row per (dataset, model, method)
  results/metrics_feature_wise.csv — one row per (dataset, feature, model, method)

HOW TO CUSTOMISE
----------------
To run only a subset of datasets:
    Edit ACTIVE_DATASETS — remove the names you don't want.

To run only a subset of models:
    Edit ACTIVE_MODELS — remove the names you don't want.

To add a new dataset:
    1. Add its CSV path and split config to DATASET_CONFIGS.
    2. Add its name to ACTIVE_DATASETS.
    3. Add its dominant period to DATASET_PERIODS (used for ACF and STL).

To add a new model:
    1. Register the model in src/ts_distill/models/factory.py.
    2. Add its constructor kwargs to MODEL_CONFIGS.
    3. Add its name to ACTIVE_MODELS.

To change distillation hyperparameters:
    Edit DISTILL_CONFIG — these apply uniformly to every run in the matrix.

QUICK SMOKE TEST (before running the full matrix)
-------------------------------------------------
Set the overrides at the bottom of this file under __main__ to limit to one
dataset/model with very few steps, e.g.:
    ACTIVE_DATASETS = ['ETTh1']
    ACTIVE_MODELS   = ['DLinear']
    DISTILL_CONFIG['n_distill_steps'] = 10
    DISTILL_CONFIG['expert_epochs']   = 5

Run from the project root:
    python -m example.experiments.experiment_matrix
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader

# ── Make the project root importable ─────────────────────────────────────────
# This file lives at example/experiments/, so three levels up is the project
# root that contains src/.  This mirrors the approach used in run_cycle.py.
sys.path.append(str(Path(__file__).parent.parent.parent))

# ── Framework — data utilities ────────────────────────────────────────────────
from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader

# ── Framework — models ────────────────────────────────────────────────────────
from ts_distill.models.factory import create_model

# ── Framework — distillation ──────────────────────────────────────────────────
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.initializer.random_sample_initializer import RandomSampleInitializer

# ── Framework — training ──────────────────────────────────────────────────────
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback

# ── Framework — trajectory ────────────────────────────────────────────────────
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.recorder.windowed_recorder import WindowedRecorder
from ts_distill.trajectory.selector.phase_window_selector import PhaseWindowSelector
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher

# ── Framework — evaluation ────────────────────────────────────────────────────
from ts_distill.evaluation.evaluation import Evaluator

# ── Framework — temporal metrics (new module) ─────────────────────────────────
from ts_distill.metrics.aggregator import MetricAggregator

# ── Framework — hybrid mixing evaluation ──────────────────────────────────────
from ts_distill.evaluation.hybrid_evaluation.hybrid import (
    BaseHybridEvaluator,
    RandomAnchorSelector,
)


# =============================================================================
# EXPERIMENT MATRIX — edit these to control what runs
# =============================================================================

# Datasets to evaluate.
# Remove entries you don't have CSV files for.
ACTIVE_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'national_illness', 'weather']

# Models to evaluate against each dataset.
ACTIVE_MODELS = ['DLinear', 'LSTM', 'MLP', 'CNN']

# Distillation method label written to the result tables.
# Update this when you add temporal-aware losses in Stage 4.
DISTILLATION_METHOD = 'MTT'

# Set False to skip temporal metric computation and CSV saving.
# Useful for quick sanity checks that don't need statsmodels.
COMPUTE_METRICS = False

# Set True to run the hybrid mixing evaluation: trains a model on a mixture of
# synthetic + real data at each ratio in HYBRID_MIXING_RATIOS and records MSE.
# Results are saved to results/metrics_hybrid.csv regardless of COMPUTE_METRICS.
COMPUTE_HYBRID_MIXING = False
HYBRID_MIXING_RATIOS  = (0.1, 0.2, 0.5)  # fractions of real data to mix in

# Set True to run the Phase Window Experiment (Hypothesis H1, Thread 3).
# For every (dataset × model) cell the expert is trained ONCE, then MTT
# distillation is run six times — once per window condition below — and the
# student MSE for each condition is written to results/phase_window.csv.
# This is independent of the standard MTT run; both can be enabled together.
COMPUTE_PHASE_WINDOW = False

# Window definitions used when COMPUTE_PHASE_WINDOW = True.
# Format: (label, (start_epoch, end_epoch)) or (label, None) for uniform.
# With expert_epochs=80 and record_every=1 each window covers 16 checkpoints.
ACTIVE_WINDOWS = [
    ('uniform', None),         # Full trajectory — standard MTT baseline
    ('W1', (1,  16)),          # Rapid:     large gradient steps, fast loss drop
    ('W2', (17, 32)),          # Early-mid: slowing down, stabilising
    ('W3', (33, 48)),          # Middle:    steady refinement, best generalisation
    ('W4', (49, 64)),          # Late-mid:  plateau, small updates
    ('W5', (65, 80)),          # Late:      near/post overfitting point
]

# Seeds used for the distillation runs inside run_phase_window_cell().
# Multiple seeds let us measure within-window variance and compare it against
# cross-window variance — the key statistical test for H1.
# The expert is trained ONCE per cell (seed 42); only the distillation RNG varies.
PHASE_WINDOW_SEEDS = [42, 123, 456]

# =============================================================================
# DATASET CONFIGURATION
# Reproduced exactly from run_cycle.py so results are directly comparable.
# =============================================================================

# Dominant seasonal period (timesteps) used for ACF short/long split and STL.
#   ETTh:  hourly data  → period = 24  (one full day)
#   ETTm:  15-min data  → period = 96  (one full day = 96 × 15 min)
DATASET_PERIODS = {
    # ETT hourly → one day = 24 timesteps
    'ETTh1': 24,
    'ETTh2': 24,
    # ETT 15-min → one day = 96 timesteps
    'ETTm1': 96,
    'ETTm2': 96,
    # Exchange rate daily → one trading week = 5 timesteps
    'exchange_rate': 5,
    # National illness weekly → one year = 52 timesteps
    'national_illness': 52,
    # Weather 10-min → one day = 144 timesteps (6/hour × 24)
    'weather': 144,
}

# Absolute paths — avoids working-directory issues when running from any location.
# Split borders reproduce the published paper splits (Informer / DLinear benchmarks).
DATASET_CONFIGS = {
    # ── ETT family (benchmark borders from Informer / DLinear papers) ─────────
    'ETTh1': {
        'csv_path':    r'D:\Final year project\ts-distill\example\ETTh1.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 24 - 96,       12 * 30 * 24 + 4 * 30 * 24 - 96],
        'border2s':    [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        'in_features': 7,
    },
    'ETTh2': {
        'csv_path':    r'D:\Final year project\ts-distill\example\ETTh2.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 24 - 96,       12 * 30 * 24 + 4 * 30 * 24 - 96],
        'border2s':    [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        'in_features': 7,
    },
    'ETTm1': {
        'csv_path':    r'D:\Final year project\ts-distill\example\ETTm1.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 96 - 96,       12 * 30 * 96 + 4 * 30 * 96 - 96],
        'border2s':    [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        'in_features': 7,
    },
    'ETTm2': {
        'csv_path':    r'D:\Final year project\ts-distill\example\ETTm2.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12 * 30 * 96 - 96,       12 * 30 * 96 + 4 * 30 * 96 - 96],
        'border2s':    [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        'in_features': 7,
    },

    # ── Exchange Rate (daily, 8 currencies, ~7588 rows) ───────────────────────
    # 70/10/20 ratio split — no standard border indices in literature.
    # Period = 5 (one trading week); seq_len/pred_len inherit global 96/96.
    'exchange_rate': {
        'csv_path':    r'D:\Final year project\ts-distill\example\exchange_rate.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 8,
    },

    # ── National Illness / ILI (weekly CDC flu surveillance, ~966 rows) ───────
    # seq_len=36 overrides the global 96 — weekly data is too sparse for 96.
    # Period = 52 (annual flu season cycle).
    'national_illness': {
        'csv_path':    r'D:\Final year project\ts-distill\example\national_illness.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.6,
        'val_ratio':   0.2,
        'in_features': 7,
        'seq_len':     36,
        'pred_len':    36,
    },

    # ── Weather (10-min intervals, 21 meteorological features, ~52696 rows) ───
    # Period = 144 (one day = 6 readings/hour × 24 hours).
    # seq_len/pred_len inherit global 96/96.
    'weather': {
        'csv_path':    r'D:\Final year project\ts-distill\example\weather.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 21,
    },
}

# =============================================================================
# MODEL CONFIGURATION
# Reproduced exactly from run_cycle.py.
# =============================================================================

MODEL_CONFIGS = {
    'DLinear': {'individual': False},
    'LSTM':    {'hidden_dim': 16, 'num_layer': 1},
    'MLP':     {},
    'CNN':     {},
}

# =============================================================================
# DISTILLATION HYPERPARAMETERS
# Reproduced exactly from run_cycle.py so that results are reproducible and
# directly comparable to the single-run baseline.
# Change values here to run sensitivity experiments.
# =============================================================================

DISTILL_CONFIG = {
    'in_features':            7,    # multivariate channels in all ETT datasets
    'seq_len':               96,    # look-back window
    'pred_len':              96,    # forecast horizon
    'expert_epochs':         80,
    'expert_lr':           0.01,
    'expert_momentum':      0.9,
    'n_distill_steps':      300,
    'n_synthetic':          384,    # synthetic sequence length (timesteps)
    'synthetic_lr':         0.1,
    'student_lr':          0.01,
    'student_steps':         20,
    'snapshot_student_steps': 50,
    'batch_size':            64,
    'trajectory_gap':         5,
    # Both real and synthetic evaluation models use the same protocol:
    # train up to eval_max_epochs with early stopping on the real val set.
    'eval_max_epochs':    300,   # hard cap; early stopping fires well before this
    'eval_lr':          0.001,
    'eval_batch_size':     32,
    'early_stop_patience': 10,   # val epochs without improvement → restore best weights
    # ACF metric settings
    'acf_n_lags':           100,    # total ACF lags (must be < n_synthetic)
}


# =============================================================================
# PIPELINE — parameterised version of run_cycle.py::main()
# =============================================================================

def _print_run_cycle_summary(
    real_mse:     float,
    transfer_mse: float,
    n_synthetic:  int,
    n_real_win:   int,
) -> None:
    """Print the same per-run summary block that run_cycle.py produces."""
    perf_retention = (real_mse / transfer_mse) * 100
    print(f"   Real-data MSE:      {real_mse:.6f}")
    print(f"   Synthetic-data MSE: {transfer_mse:.6f}")
    print(f"   Error ratio:        {transfer_mse / real_mse:.2f}x")
    print(f"   Performance kept:   {perf_retention:.1f}%")
    print(f"   Compression:        {n_synthetic}/{n_real_win} windows "
          f"({n_synthetic / n_real_win * 100:.2f}%)")


def _make_model_factory(model_name: str, cfg: dict):
    """Return a zero-argument callable that creates a fresh model instance."""
    def factory():
        return create_model(
            model_type   = model_name,
            seq_len      = cfg['seq_len'],
            pred_len     = cfg['pred_len'],
            in_features  = cfg['in_features'],
            model_kwargs = MODEL_CONFIGS[model_name],
        )
    return factory


def run_single_experiment(
    dataset_name:    str,
    model_name:            str,
    cfg:                   dict,
    compute_metrics:       bool = True,
    compute_hybrid_mixing: bool = False,
) -> tuple:
    """
    Run the full MTT pipeline for one (dataset, model) combination.

    Args:
        dataset_name          (str):  Key in DATASET_CONFIGS, e.g. 'ETTh1'.
        model_name            (str):  Key in MODEL_CONFIGS,   e.g. 'DLinear'.
        cfg                   (dict): Merged distillation/eval hyperparameters.
        compute_metrics       (bool): When False, skip temporal metrics.
        compute_hybrid_mixing (bool): When True, run hybrid mixing evaluation across
                                      HYBRID_MIXING_RATIOS and return results.

    Returns:
        Tuple[List[Dict], Dict, Optional[Dict]]: (feature_rows, main_row, hybrid_row)
            feature_rows — per-channel metric dicts (empty when compute_metrics=False).
            main_row     — aggregate result dict.
            hybrid_row   — {dataset, model, hybrid_<r>_mse, ...} or None.
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Resolve per-dataset overrides ─────────────────────────────────────────
    # DATASET_CONFIGS entries may override seq_len, pred_len, and in_features.
    # Example: national_illness uses seq_len=36 (weekly data, 96 too long).
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])
    in_features = dataset_cfg.get('in_features', cfg['in_features'])
    window_size = seq_len + pred_len
    make_model  = _make_model_factory(
        model_name, {**cfg, 'seq_len': seq_len, 'pred_len': pred_len, 'in_features': in_features}
    )

    # ── Step 1: Load and normalise data ───────────────────────────────────────
    csv_path = dataset_cfg['csv_path']

    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV not found at '{csv_path}'. "
            f"Download the ETT datasets and place them in example/."
        )

    df_raw = pd.read_csv(csv_path)
    values = df_raw.iloc[:, 1:].values.astype(np.float32)   # drop date column

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values      = values,
        window_size = window_size,
        seq_len     = seq_len,
        dataset_cfg = dataset_cfg,
    )

    # Fit scaler only on train rows to prevent leakage.
    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data = scaler.transform(values)

    train_data = torch.tensor(make_windows(data[train_start:train_end], window_size), dtype=torch.float32)
    val_data   = torch.tensor(make_windows(data[val_start:val_end],     window_size), dtype=torch.float32)
    test_data  = torch.tensor(make_windows(data[test_start:test_end],   window_size), dtype=torch.float32)

    # Raw continuous training sequence — passed to the distiller and to metrics.
    raw_train_data = torch.tensor(data[train_start:train_end], dtype=torch.float32)

    # ── Step 2: Train expert and record trajectory ────────────────────────────
    recorder     = SimpleRecorder(record_every=1)
    expert_model = make_model()
    expert_trainer = Trainer(
        model     = expert_model,
        optimizer = torch.optim.SGD(
            expert_model.parameters(),
            lr       = cfg['expert_lr'],
            momentum = cfg['expert_momentum'],
        ),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    expert_loader = MiniBatchLoader(train_data, batch_size=cfg['batch_size'])
    expert_trainer.fit(
        dataloader = expert_loader,
        epochs     = cfg['expert_epochs'],
        callbacks  = [recorder, SimpleCallback()],
    )

    # ── Step 3: Distil synthetic sequence ─────────────────────────────────────
    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(raw_train_data, cfg['n_synthetic'])

    distiller = MTTDistiller(
        initializer            = initializer,
        matcher                = MSEMatcher(),
        model_factory          = make_model,
        expert_recorder        = recorder,
        expert_epochs          = cfg['trajectory_gap'],
        syn_batch_size         = cfg['batch_size'],
        synthetic_lr           = cfg['synthetic_lr'],
        student_lr             = cfg['student_lr'],
        student_steps          = cfg['student_steps'],
        snapshot_student_steps = cfg['snapshot_student_steps'],
        seq_len                = seq_len,
        pred_len               = pred_len,
    )

    synthetic_sequence = distiller.distill(
        synthetic_init = synthetic_init,
        n_steps        = cfg['n_distill_steps'],
        val_data       = val_data,
    )

    # Shared val loader — early-stopping signal for BOTH models so convergence
    # is measured against the same real validation distribution.
    eval_val_loader = TorchDataLoader(
        val_data, batch_size=cfg['eval_batch_size'], shuffle=False
    )

    # ── Step 4a: Evaluate model trained on real data ──────────────────────────
    # Re-seed before eval so model init is independent of distillation RNG state.
    torch.manual_seed(0)
    real_model   = make_model()
    real_trainer = Trainer(
        model     = real_model,
        optimizer = torch.optim.Adam(real_model.parameters(), lr=cfg['eval_lr']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    real_trainer.fit(
        dataloader = TorchDataLoader(train_data, batch_size=cfg['eval_batch_size'], shuffle=True),
        epochs     = cfg['eval_max_epochs'],
        val_loader = eval_val_loader,
        patience   = cfg['early_stop_patience'],
    )

    # ── Step 4b: Evaluate model trained on synthetic data ─────────────────────
    syn_windows = torch.tensor(
        make_windows(synthetic_sequence.cpu().numpy(), window_size),
        dtype=torch.float32,
    )
    torch.manual_seed(1)
    syn_model   = make_model()
    syn_trainer = Trainer(
        model     = syn_model,
        optimizer = torch.optim.Adam(syn_model.parameters(), lr=cfg['eval_lr']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    syn_trainer.fit(
        dataloader = TorchDataLoader(syn_windows, batch_size=cfg['eval_batch_size'], shuffle=True),
        epochs     = cfg['eval_max_epochs'],
        val_loader = eval_val_loader,
        patience   = cfg['early_stop_patience'],
    )

    # ── Step 4c: Score both models on the real held-out test set ─────────────
    evaluator         = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size'])
    real_metrics      = evaluator.test_on_real(real_model, test_data.to(device))
    synthetic_metrics = evaluator.test_on_real(syn_model,  test_data.to(device))

    real_mse     = real_metrics['MSE']
    transfer_mse = synthetic_metrics['MSE']

    # ── Hybrid / no-metrics summary ───────────────────────────────────────────
    if not compute_metrics:
        _print_run_cycle_summary(
            real_mse     = real_mse,
            transfer_mse = transfer_mse,
            n_synthetic  = cfg['n_synthetic'],
            n_real_win   = train_data.shape[0],
        )

    # ── Step 5: Hybrid mixing evaluation ─────────────────────────────────────
    # Trains a fresh model on synthetic + real mixtures at each ratio and
    # measures how MSE changes as the real data fraction increases.
    hybrid_row = None
    if compute_hybrid_mixing:
        hybrid_evaluator = BaseHybridEvaluator(
            anchor_selector = RandomAnchorSelector(),
            seq_len         = seq_len,
            batch_size      = cfg['batch_size'],
            device          = device,
        )
        mixing_results = hybrid_evaluator.evaluate_mixing(
            synthetic_data  = synthetic_sequence,
            real_train_data = raw_train_data,
            real_test_loader = TorchDataLoader(test_data, batch_size=cfg['eval_batch_size']),
            model_fn        = make_model,
            window_size     = window_size,
            mixing_ratios   = HYBRID_MIXING_RATIOS,
        )
        print("\n" + "-" * 60)
        print("Hybrid Mixing  (MSE vs real-data fraction)")
        print("-" * 60)
        for key, m in mixing_results.items():
            pct = key.replace("hybrid_", "")
            print(f"   Real {pct:>3}%  ->  MSE: {m['MSE']:.6f}")
        print("-" * 60)

        hybrid_row = {
            'dataset':             dataset_name,
            'model':               model_name,
            'distillation_method': DISTILLATION_METHOD,
            **{f"hybrid_{k.replace('hybrid_', '')}_mse": v['MSE']
               for k, v in mixing_results.items()},
        }

    if not compute_metrics:
        return [], {
            'dataset':             dataset_name,
            'model':               model_name,
            'distillation_method': DISTILLATION_METHOD,
            'real_mse':            real_mse,
            'transfer_mse':        transfer_mse,
        }, hybrid_row

    # ── Step 6: Compute temporal metrics ──────────────────────────────────────
    real_np = raw_train_data.detach().cpu().numpy()
    syn_np  = synthetic_sequence.detach().cpu().numpy()

    period     = DATASET_PERIODS[dataset_name]
    aggregator = MetricAggregator(period=period, n_lags=cfg['acf_n_lags'])

    feature_rows, main_row = aggregator.compute_all(
        real                = real_np,
        synthetic           = syn_np,
        real_mse            = real_mse,
        transfer_mse        = transfer_mse,
        dataset             = dataset_name,
        model               = model_name,
        distillation_method = DISTILLATION_METHOD,
    )

    return feature_rows, main_row, hybrid_row


# =============================================================================
# PHASE WINDOW EXPERIMENT — H1 (Thread 3)
# =============================================================================

def run_phase_window_cell(
    dataset_name: str,
    model_name:   str,
    cfg:          dict,
    results_dir:  Path = None,
) -> list:
    """
    Run the phase window experiment for one (dataset, model) cell.

    Trains the expert model ONCE and then runs MTT distillation once per
    window condition in ACTIVE_WINDOWS, using:
        SimpleRecorder       — records the full weight trajectory
        PhaseWindowSelector  — filters trajectory to the epoch window
        WindowedRecorder     — exposes the filtered trajectory to MTTDistiller

    Each (window, seed) row is flushed to CSV immediately after it completes
    when results_dir is provided, so no data is lost if the run crashes.

    Args:
        dataset_name (str):  Key in DATASET_CONFIGS.
        model_name   (str):  Key in MODEL_CONFIGS.
        cfg          (dict): Distillation / evaluation hyperparameters.
        results_dir  (Path): If provided, each row is saved to
                             results_dir/phase_window.csv as soon as it
                             completes (crash-safe).

    Returns:
        List of result dicts, one per (window, seed) pair.  Each dict
        contains the columns defined in PHASE_WINDOW_COLUMNS.
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Resolve per-dataset overrides ────────────────────────────────────────
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])
    in_features = dataset_cfg.get('in_features', cfg['in_features'])
    window_size = seq_len + pred_len
    make_model  = _make_model_factory(
        model_name, {**cfg, 'seq_len': seq_len, 'pred_len': pred_len, 'in_features': in_features}
    )

    # ── Step 1: Load and normalise data ──────────────────────────────────────
    csv_path = dataset_cfg['csv_path']
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: '{csv_path}'")

    df_raw = pd.read_csv(csv_path)
    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values=values, window_size=window_size, seq_len=seq_len, dataset_cfg=dataset_cfg,
    )
    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data = scaler.transform(values)

    train_data     = torch.tensor(make_windows(data[train_start:train_end], window_size), dtype=torch.float32)
    val_data       = torch.tensor(make_windows(data[val_start:val_end],     window_size), dtype=torch.float32)
    test_data      = torch.tensor(make_windows(data[test_start:test_end],   window_size), dtype=torch.float32)
    raw_train_data = torch.tensor(data[train_start:train_end], dtype=torch.float32)

    # ── Step 2: Train expert ONCE — shared across all window conditions ───────
    print(f"   [Expert] Training SimpleRecorder trajectory (record_every=1)...")
    recorder       = SimpleRecorder(record_every=1)
    expert_model   = make_model()
    expert_trainer = Trainer(
        model     = expert_model,
        optimizer = torch.optim.SGD(
            expert_model.parameters(),
            lr=cfg['expert_lr'], momentum=cfg['expert_momentum'],
        ),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    expert_loader = MiniBatchLoader(train_data, batch_size=cfg['batch_size'])
    expert_trainer.fit(
        dataloader = expert_loader,
        epochs     = cfg['expert_epochs'],
        callbacks  = [recorder, SimpleCallback()],
    )
    n_ckpts = len(recorder.get_trajectory())
    print(f"   [Expert] {n_ckpts} checkpoints recorded "
          f"(steps {recorder.get_trajectory()[0]['step']}"
          f"-{recorder.get_trajectory()[-1]['step']})")

    # ── Step 3: Real-data MSE baseline — computed once, reused for all windows
    eval_val_loader = TorchDataLoader(val_data, batch_size=cfg['eval_batch_size'], shuffle=False)

    torch.manual_seed(0)
    real_model   = make_model()
    real_trainer = Trainer(
        model     = real_model,
        optimizer = torch.optim.Adam(real_model.parameters(), lr=cfg['eval_lr']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    real_trainer.fit(
        dataloader = TorchDataLoader(train_data, batch_size=cfg['eval_batch_size'], shuffle=True),
        epochs     = cfg['eval_max_epochs'],
        val_loader = eval_val_loader,
        patience   = cfg['early_stop_patience'],
    )
    evaluator = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size'])
    real_mse  = evaluator.test_on_real(real_model, test_data.to(device))['MSE']
    print(f"   [Baseline] Real MSE = {real_mse:.6f}")

    # ── Step 4: Loop over window conditions × seeds ───────────────────────────
    # Multiple seeds per window give within-window variance.
    # The expert trajectory is fixed (trained once above); only the distillation
    # RNG varies across seeds.  This isolates the phase effect from init noise.
    rows = []
    for window_label, window_range in ACTIVE_WINDOWS:
        # Build the active recorder once per window (shared across seeds)
        if window_range is None:
            active_recorder = recorder
            n_win_ckpts     = n_ckpts
            recorder_desc   = f"full trajectory ({n_ckpts} checkpoints)"
        else:
            start_e, end_e  = window_range
            selector        = PhaseWindowSelector(start_epoch=start_e, end_epoch=end_e)
            active_recorder = WindowedRecorder(source_recorder=recorder, selector=selector)
            n_win_ckpts     = active_recorder.n_checkpoints
            recorder_desc   = (f"PhaseWindowSelector({start_e}-{end_e}) "
                               f"-> WindowedRecorder: {n_win_ckpts} checkpoints")

        print(f"\n   [Window {window_label}]  {recorder_desc}")

        if window_range is not None and n_win_ckpts < 2:
            note = f'insufficient_checkpoints ({n_win_ckpts})'
            print(f"   [SKIP] {note}")
            rows.append({
                'dataset': dataset_name, 'model': model_name,
                'distillation_method': DISTILLATION_METHOD,
                'window': window_label,
                'window_start': window_range[0], 'window_end': window_range[1],
                'sampling_strategy': f'window_{window_label}',
                'n_pairs_available': 0, 'seed': -1,
                'real_mse': real_mse, 'transfer_mse': float('nan'),
                'mse_ratio': float('nan'), 'notes': note,
            })
            continue

        n_pairs = max(0, n_win_ckpts - cfg['trajectory_gap'])

        for seed in PHASE_WINDOW_SEEDS:
            row = {
                'dataset':             dataset_name,
                'model':               model_name,
                'distillation_method': DISTILLATION_METHOD,
                'window':              window_label,
                'window_start':        0 if window_range is None else window_range[0],
                'window_end':          cfg['expert_epochs'] if window_range is None else window_range[1],
                'sampling_strategy':   'uniform' if window_range is None else f'window_{window_label}',
                'n_pairs_available':   n_pairs,
                'seed':                seed,
                'real_mse':            real_mse,
                'transfer_mse':        float('nan'),
                'mse_ratio':           float('nan'),
                'notes':               '',
            }

            try:
                torch.manual_seed(seed)
                initializer    = RandomSampleInitializer()
                synthetic_init = initializer.initialize_sequence(raw_train_data, cfg['n_synthetic'])

                distiller = MTTDistiller(
                    initializer            = initializer,
                    matcher                = MSEMatcher(),
                    model_factory          = make_model,
                    expert_recorder        = active_recorder,
                    expert_epochs          = cfg['trajectory_gap'],
                    syn_batch_size         = cfg['batch_size'],
                    synthetic_lr           = cfg['synthetic_lr'],
                    student_lr             = cfg['student_lr'],
                    student_steps          = cfg['student_steps'],
                    snapshot_student_steps = cfg['snapshot_student_steps'],
                    seq_len                = seq_len,
                    pred_len               = pred_len,
                )
                synthetic_sequence = distiller.distill(
                    synthetic_init = synthetic_init,
                    n_steps        = cfg['n_distill_steps'],
                    val_data       = val_data,
                )

                syn_windows = torch.tensor(
                    make_windows(synthetic_sequence.cpu().numpy(), window_size), dtype=torch.float32
                )
                torch.manual_seed(seed + 1)
                syn_model   = make_model()
                syn_trainer = Trainer(
                    model     = syn_model,
                    optimizer = torch.optim.Adam(syn_model.parameters(), lr=cfg['eval_lr']),
                    criterion = torch.nn.MSELoss(),
                    device    = device,
                    seq_len   = seq_len,
                )
                syn_trainer.fit(
                    dataloader = TorchDataLoader(syn_windows, batch_size=cfg['eval_batch_size'], shuffle=True),
                    epochs     = cfg['eval_max_epochs'],
                    val_loader = eval_val_loader,
                    patience   = cfg['early_stop_patience'],
                )
                transfer_mse = evaluator.test_on_real(syn_model, test_data.to(device))['MSE']
                mse_ratio    = transfer_mse / real_mse if real_mse > 0 else float('nan')
                row.update({'transfer_mse': transfer_mse, 'mse_ratio': mse_ratio})
                print(f"     seed={seed}  real={real_mse:.6f}  "
                      f"transfer={transfer_mse:.6f}  ratio={mse_ratio:.4f}")

            except Exception as exc:
                note = repr(exc)[:200]
                print(f"     seed={seed}  [ERROR] {note}")
                row['notes'] = note

            rows.append(row)
            if results_dir is not None:
                _flush_phase_row(row, results_dir)

    return rows


PHASE_WINDOW_COLUMNS = [
    'dataset', 'model', 'distillation_method',
    'window', 'window_start', 'window_end', 'sampling_strategy', 'n_pairs_available',
    'seed', 'real_mse', 'transfer_mse', 'mse_ratio', 'notes',
]


# =============================================================================
# CSV I/O
# =============================================================================

def save_results(
    main_rows:    list,
    feature_rows: list,
    results_dir:  Path,
) -> None:
    """
    Write (or overwrite) both result CSV files with the collected experiment rows.

    Column order is enforced via MetricAggregator.MAIN_COLUMNS and
    MetricAggregator.FEATURE_COLUMNS so the CSVs always match the table schema
    regardless of dict insertion order.

    Args:
        main_rows    (list): List of dicts for the aggregate main table.
        feature_rows (list): List of dicts for the feature-wise table.
        results_dir  (Path): Directory to write CSVs into (created if absent).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    main_path    = results_dir / 'metrics_main.csv'
    feature_path = results_dir / 'metrics_feature_wise.csv'

    if main_rows:
        pd.DataFrame(main_rows)[MetricAggregator.MAIN_COLUMNS].to_csv(
            main_path, index=False
        )
        print(f"\nMain table saved    → {main_path}")

    if feature_rows:
        pd.DataFrame(feature_rows)[MetricAggregator.FEATURE_COLUMNS].to_csv(
            feature_path, index=False
        )
        print(f"Feature table saved → {feature_path}")


def save_hybrid_results(hybrid_rows: list, results_dir: Path) -> None:
    """Write hybrid mixing results to metrics_hybrid.csv."""
    if not hybrid_rows:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / 'metrics_hybrid.csv'
    # Column order: id columns first, then ratio MSE columns in ratio order.
    id_cols    = ['dataset', 'model', 'distillation_method']
    ratio_cols = sorted(
        [c for c in hybrid_rows[0] if c not in id_cols],
        key=lambda c: int(c.split('_')[1]),
    )
    pd.DataFrame(hybrid_rows)[id_cols + ratio_cols].to_csv(path, index=False)
    print(f"Hybrid table saved  → {path}")


def save_phase_window_results(rows: list, results_dir: Path) -> None:
    """
    Append phase window experiment rows to results/phase_window.csv.

    Writes the header only on first call (when the file does not yet exist),
    so partial runs accumulate rows safely across restarts.

    Args:
        rows        (list): List of result dicts from run_phase_window_cell().
        results_dir (Path): Directory to write into (created if absent).
    """
    if not rows:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'phase_window.csv'
    write_header = not csv_path.exists()
    pd.DataFrame(rows)[PHASE_WINDOW_COLUMNS].to_csv(
        csv_path, mode='a', header=write_header, index=False
    )
    print(f"Phase window rows saved ({len(rows)}) -> {csv_path}")


def _flush_phase_row(row: dict, results_dir: Path) -> None:
    """Append a single phase-window result row to CSV immediately after it completes."""
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'phase_window.csv'
    write_header = not csv_path.exists()
    pd.DataFrame([row])[PHASE_WINDOW_COLUMNS].to_csv(
        csv_path, mode='a', header=write_header, index=False
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Loop over the full ACTIVE_DATASETS x ACTIVE_MODELS matrix, run each
    combination, and collect results into both CSV files.

    A try/except around each combination means a single failed run (e.g. a
    missing CSV file) does not abort the entire matrix — the error is logged
    and the loop continues.
    """
    results_dir  = Path(__file__).parent / 'results'
    all_main     = []
    all_features = []
    all_hybrid   = []

    total = len(ACTIVE_DATASETS) * len(ACTIVE_MODELS)
    done  = 0

    for dataset_name in ACTIVE_DATASETS:
        for model_name in ACTIVE_MODELS:
            done += 1
            header = f"[{done}/{total}]  {dataset_name}  x  {model_name}"
            print(f"\n{'=' * 70}")
            print(header)
            print(f"{'=' * 70}")

            # ── Standard MTT run ──────────────────────────────────────────────
            try:
                feature_rows, main_row, hybrid_row = run_single_experiment(
                    dataset_name          = dataset_name,
                    model_name            = model_name,
                    cfg                   = DISTILL_CONFIG,
                    compute_metrics       = COMPUTE_METRICS,
                    compute_hybrid_mixing = COMPUTE_HYBRID_MIXING,
                )
                all_main.append(main_row)
                all_features.extend(feature_rows)
                if hybrid_row is not None:
                    all_hybrid.append(hybrid_row)

                real_mse     = main_row['real_mse']
                transfer_mse = main_row['transfer_mse']
                mse_ratio    = transfer_mse / real_mse if real_mse > 0 else float('nan')
                print(
                    f"\n  [Standard MTT]  real_mse={real_mse:.6f}  "
                    f"transfer_mse={transfer_mse:.6f}  ratio={mse_ratio:.4f}"
                )
                if COMPUTE_METRICS:
                    print(
                        f"  acf_short={main_row['acf_short']:.4f}  "
                        f"acf_long={main_row['acf_long']:.4f}"
                    )

            except Exception as exc:
                print(f"\n  [SKIP] {dataset_name} x {model_name} standard run failed: {exc}")

            # ── Phase window experiment (H1, Thread 3) ───────────────────────
            if COMPUTE_PHASE_WINDOW:
                print(f"\n  [Phase Window] {dataset_name} x {model_name}")
                try:
                    run_phase_window_cell(
                        dataset_name = dataset_name,
                        model_name   = model_name,
                        cfg          = DISTILL_CONFIG,
                        results_dir  = results_dir,
                    )
                    # each row flushed immediately inside run_phase_window_cell
                except Exception as exc:
                    print(f"  [SKIP] Phase window failed: {exc}")

    if COMPUTE_METRICS:
        save_results(all_main, all_features, results_dir)
    if COMPUTE_HYBRID_MIXING:
        save_hybrid_results(all_hybrid, results_dir)

    print(f"\nMatrix complete. {len(all_main)}/{total} combinations succeeded.")


if __name__ == '__main__':
    # ── Optional overrides for quick smoke testing ────────────────────────────
    # Uncomment and edit the lines below to test a single fast combination
    # before committing to the full matrix run.
    #
    # H1 experiment: ETTh1 + ETTm1 x all 4 models, phase window enabled
    ACTIVE_DATASETS[:] = ['ETTh1', 'ETTm1']
    ACTIVE_MODELS[:]   = ['DLinear', 'LSTM', 'MLP', 'CNN']
    DISTILL_CONFIG['n_distill_steps'] = 300
    DISTILL_CONFIG['expert_epochs']   = 80
    DISTILL_CONFIG['eval_max_epochs'] = 50   # still uses early stopping

    # Enable the phase window experiment (H1).
    # Set False here to run only the standard MTT pipeline.
    COMPUTE_PHASE_WINDOW = True
    # ─────────────────────────────────────────────────────────────────────────

    main()
