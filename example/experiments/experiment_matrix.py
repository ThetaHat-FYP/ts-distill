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
from ts_distill.distillation_core.initializer.real_sample_initializer import RealSampleInitializer

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
    dataset_name: str,
    model_name:   str,
    cfg:          dict,
) -> tuple:
    """
    Run the full MTT pipeline for one (dataset, model) combination and return
    the metric rows ready to be appended to the result CSV files.

    Args:
        dataset_name (str): Key in DATASET_CONFIGS, e.g. 'ETTh1'.
        model_name   (str): Key in MODEL_CONFIGS,   e.g. 'DLinear'.
        cfg          (dict): Merged distillation/eval hyperparameters.

    Returns:
        Tuple[List[Dict], Dict]: (feature_rows, main_row)
            feature_rows — C dicts, one per channel, for the feature-wise table.
            main_row     — one dict for the aggregate main table.
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
    initializer    = RealSampleInitializer()
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

    # ── Step 5: Compute temporal metrics ──────────────────────────────────────
    # Convert to numpy — both are already normalised continuous sequences.
    real_np = raw_train_data.detach().cpu().numpy()         # shape (T, C)
    syn_np  = synthetic_sequence.detach().cpu().numpy()     # shape (n_synthetic, C)

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

    return feature_rows, main_row


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


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Loop over the full ACTIVE_DATASETS × ACTIVE_MODELS matrix, run each
    combination, and collect results into both CSV files.

    A try/except around each combination means a single failed run (e.g. a
    missing CSV file) does not abort the entire matrix — the error is logged
    and the loop continues.
    """
    results_dir  = Path(__file__).parent / 'results'
    all_main     = []
    all_features = []

    total = len(ACTIVE_DATASETS) * len(ACTIVE_MODELS)
    done  = 0

    for dataset_name in ACTIVE_DATASETS:
        for model_name in ACTIVE_MODELS:
            done += 1
            header = f"[{done}/{total}]  {dataset_name}  ×  {model_name}"
            print(f"\n{'=' * 70}")
            print(header)
            print(f"{'=' * 70}")

            try:
                feature_rows, main_row = run_single_experiment(
                    dataset_name = dataset_name,
                    model_name   = model_name,
                    cfg          = DISTILL_CONFIG,
                )
                all_main.append(main_row)
                all_features.extend(feature_rows)

                # Print a brief summary for this combination.
                print(
                    f"\n  real_mse={main_row['real_mse']:.6f}  "
                    f"transfer_mse={main_row['transfer_mse']:.6f}  "
                    f"acf_short={main_row['acf_short']:.4f}  "
                    f"acf_long={main_row['acf_long']:.4f}"
                )

            except Exception as exc:
                print(f"\n  [SKIP] {dataset_name} × {model_name} failed: {exc}")
                continue

    # Write everything collected so far (even if some combinations were skipped).
    save_results(all_main, all_features, results_dir)
    print(f"\nMatrix complete. {len(all_main)}/{total} combinations succeeded.")


if __name__ == '__main__':
    # ── Optional overrides for quick smoke testing ────────────────────────────
    # Uncomment and edit the lines below to test a single fast combination
    # before committing to the full matrix run.
    #
    ACTIVE_DATASETS[:] = ['national_illness']
    ACTIVE_MODELS[:]   = ['LSTM']
    DISTILL_CONFIG['n_distill_steps'] = 300
    DISTILL_CONFIG['expert_epochs']   = 80
    DISTILL_CONFIG['eval_max_epochs'] = 50   # still uses early stopping
    # ─────────────────────────────────────────────────────────────────────────

    main()
