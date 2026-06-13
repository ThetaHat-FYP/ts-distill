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
from ts_distill.distillation_core.initializer.geommetry_sequence_initializer import GeometrySequenceInitializer
from ts_distill.distillation_core.initializer.uncertainty_sequence_initializer import UncertaintySampleInitializer

# ── Framework — training ──────────────────────────────────────────────────────
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback

# ── Framework — trajectory ────────────────────────────────────────────────────
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
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
ACTIVE_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']

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

# CSV paths are relative to the project root (where you run the script from).
# Split borders reproduce the published paper splits (Informer / DLinear benchmarks).
DATASET_CONFIGS = {
    # ── ETT family (benchmark borders from Informer / DLinear papers) ─────────
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

    # ── Exchange Rate (daily, 8 currencies, ~7588 rows) ───────────────────────
    # 70/10/20 ratio split — no standard border indices in literature.
    # Period = 5 (one trading week); seq_len/pred_len inherit global 96/96.
    'exchange_rate': {
        'csv_path':    'example/exchange_rate.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 8,
    },

    # ── National Illness / ILI (weekly CDC flu surveillance, ~966 rows) ───────
    # seq_len=36 overrides the global 96 — weekly data is too sparse for 96.
    # Period = 52 (annual flu season cycle).
    'national_illness': {
        'csv_path':    'example/national_illness.csv',
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
        'csv_path':    'example/weather.csv',
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
    'synthetic_lr':         5.0,
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
    torch.manual_seed(123)
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
    # initializer    = RandomSampleInitializer()
    # synthetic_init = initializer.initialize_sequence(raw_train_data, cfg['n_synthetic'])
    # initializer    = GeometrySequenceInitializer()
    # synthetic_init = initializer.initialize_sequence(raw_train_data,cfg['n_synthetic'])
    initializer = UncertaintySampleInitializer()
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
            'n_distill_steps':     cfg['n_distill_steps'],
            'real_mse':            real_mse,
            'transfer_mse':        transfer_mse,
        }, hybrid_row

    # ── Step 6: Compute temporal metrics ──────────────────────────────────────
    real_np = raw_train_data.detach().cpu().numpy()
    syn_np  = synthetic_sequence.detach().cpu().numpy()

    period  = DATASET_PERIODS[dataset_name]
    # n_lags must exceed period (for acf_long range) and stay below n_synthetic.
    # Use at least 2×period so short/long ranges are equally wide.
    n_lags  = min(max(2 * period, cfg['acf_n_lags']), cfg['n_synthetic'] - 1)
    aggregator = MetricAggregator(period=period, n_lags=n_lags)

    feature_rows, main_row = aggregator.compute_all(
        real                = real_np,
        synthetic           = syn_np,
        real_mse            = real_mse,
        transfer_mse        = transfer_mse,
        dataset             = dataset_name,
        model               = model_name,
        distillation_method = DISTILLATION_METHOD,
        n_distill_steps     = cfg['n_distill_steps'],
    )

    return feature_rows, main_row, hybrid_row


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


def save_mse_results(main_rows: list, results_dir: Path) -> None:
    """Write real_mse and synthetic_mse for every completed (dataset, model) run."""
    if not main_rows:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / 'metrics_mse.csv'
    cols = ['dataset', 'model', 'distillation_method', 'real_mse', 'transfer_mse']
    pd.DataFrame(main_rows)[cols].rename(
        columns={'transfer_mse': 'synthetic_mse'}
    ).to_csv(path, index=False)
    print(f"MSE table saved     → {path}")


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


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_budget_sensitivity(
    dataset_name: str,
    model_name:   str,
    step_counts:  list,
    results_dir:  Path,
) -> None:
    """
    Run the same (dataset, model) at multiple distillation step counts and save
    results to metrics_budget_sensitivity.csv.

    Because torch.manual_seed(42) is called inside run_single_experiment, every
    step count gets an identical expert trajectory and identical initialization —
    only the distillation budget differs.  This makes the comparison clean.

    Args:
        dataset_name (str):  E.g. 'ETTh1'.
        model_name   (str):  E.g. 'DLinear'.
        step_counts  (list): List of int step counts, e.g. [0, 50, 100, 150, 200, 250, 300].
                             0 = pure initialization baseline (no MTT optimization).
        results_dir  (Path): Directory to write metrics_budget_sensitivity.csv into.
    """
    all_main     = []
    all_features = []
    total = len(step_counts)

    for i, n_steps in enumerate(step_counts, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{total}]  {dataset_name}  x  {model_name}  —  n_steps={n_steps}")
        print(f"{'=' * 70}")
        try:
            cfg = {**DISTILL_CONFIG, 'n_distill_steps': n_steps}
            feature_rows, main_row, _ = run_single_experiment(
                dataset_name          = dataset_name,
                model_name            = model_name,
                cfg                   = cfg,
                compute_metrics       = True,
                compute_hybrid_mixing = False,
            )
            all_main.append(main_row)
            all_features.extend(feature_rows)
            print(
                f"\n  steps={n_steps}  transfer_mse={main_row['transfer_mse']:.6f}  "
                f"acf_short={main_row['acf_short']:.4f}  acf_long={main_row['acf_long']:.4f}  "
                f"fft_distance={main_row['fft_distance']:.4f}  trend_error={main_row['trend_error']:.4f}"
            )
        except Exception as exc:
            print(f"\n  [SKIP] n_steps={n_steps} failed: {exc}")
            continue

    results_dir.mkdir(parents=True, exist_ok=True)
    if all_main:
        pd.DataFrame(all_main)[MetricAggregator.MAIN_COLUMNS].to_csv(
            results_dir / 'metrics_budget_sensitivity.csv', index=False
        )
        print(f"\nBudget sensitivity saved → {results_dir / 'metrics_budget_sensitivity.csv'}")
    if all_features:
        pd.DataFrame(all_features)[MetricAggregator.FEATURE_COLUMNS].to_csv(
            results_dir / 'metrics_budget_sensitivity_feature.csv', index=False
        )
        print(f"Budget feature table    → {results_dir / 'metrics_budget_sensitivity_feature.csv'}")


def main() -> None:
    """
    Loop over ACTIVE_DATASETS × ACTIVE_MODELS.  Saves metrics_mse.csv after
    every successful run; skips already-completed pairs so interrupted runs
    can be resumed by re-running the same command.
    """
    results_dir = Path(__file__).parent / 'results'
    mse_path    = results_dir / 'metrics_mse.csv'

    # Load any previous results so we can resume mid-matrix
    all_main: list = []
    completed: set = set()
    if mse_path.exists():
        existing = pd.read_csv(mse_path).rename(
            columns={'synthetic_mse': 'transfer_mse'}
        ).to_dict('records')
        all_main  = existing
        completed = {(r['dataset'], r['model']) for r in existing}
        print(f"Resuming: {len(completed)} combination(s) already done.")

    all_features: list = []
    all_hybrid:   list = []

    total = len(ACTIVE_DATASETS) * len(ACTIVE_MODELS)
    done  = 0

    for dataset_name in ACTIVE_DATASETS:
        for model_name in ACTIVE_MODELS:
            done += 1

            if (dataset_name, model_name) in completed:
                print(f"[{done}/{total}]  {dataset_name} x {model_name}  — already done, skipping")
                continue

            print(f"\n{'=' * 70}")
            print(f"[{done}/{total}]  {dataset_name}  x  {model_name}")
            print(f"{'=' * 70}")

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

                # Always save MSE results after each run
                save_mse_results(all_main, results_dir)
                print(
                    f"\n  real_mse={main_row['real_mse']:.6f}  "
                    f"synthetic_mse={main_row['transfer_mse']:.6f}"
                )

            except Exception as exc:
                print(f"\n  [SKIP] {dataset_name} x {model_name} failed: {exc}")

    if COMPUTE_METRICS:
        save_results(all_main, all_features, results_dir)
    if COMPUTE_HYBRID_MIXING:
        save_hybrid_results(all_hybrid, results_dir)

    print(f"\nMatrix complete. {len(all_main)}/{total} combinations succeeded.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run the experiment matrix.')
    parser.add_argument(
        '--dataset',
        choices=list(DATASET_CONFIGS.keys()) + ['all'],
        default='all',
        help='Single dataset to run, or "all" for the full matrix (default: all)',
    )
    parser.add_argument(
        '--model',
        choices=list(MODEL_CONFIGS.keys()) + ['all'],
        default='all',
        help='Single model to run, or "all" for every model (default: all)',
    )
    args = parser.parse_args()

    if args.dataset != 'all':
        ACTIVE_DATASETS[:] = [args.dataset]
    if args.model != 'all':
        ACTIVE_MODELS[:] = [args.model]

    main()
