"""
h2_strategy_sweep.py — H2: Selection Strategy Effect
======================================================
Tests whether the choice of real-window selection strategy (S1–S4) produces
statistically distinguishable MSE profiles at the same mixing ratio.

HYPOTHESIS H2
-------------
H0 (Null):
    All four selection strategies produce equivalent MSE at every ratio r
    for all datasets.

H1 (Alternative):
    At least one dataset shows statistically distinguishable MSE profiles
    between strategies at the same ratio r, with strategy_delta > 0.01.

STRATEGY DEFINITIONS
--------------------
S1  RandomAnchorSelector       — random windows (H1/H2 baseline)
S2  StartExtendSelector        — first r% chronologically
S3  UniformStrideSelector      — evenly spaced across series
S4  ImportanceWeightedSelector — highest expert-error windows first
S5  DiversityAnchorSelector     — diversity-based farthest-first coverage

SCOPE (per research guide Section 6, Step 3 & 4)
-------------------------------------------------
S1 + S2: ALL 24 cells (6 datasets × 4 models)
S3 + S4: 12 cells only:
    ETTh1, ETTm1, exchange_rate × all 4 models

Rationale for subset: S3/S4 are exploratory. The three chosen datasets
provide maximum contrast — periodic hourly (ETTh1), periodic 15-min (ETTm1),
non-periodic daily (exchange_rate).

HOW TO RUN
----------
Step 1 — S1 + S2 on all 24 cells:
    python -m example.experiments.h2_strategy_sweep --strategies S1 S2

Step 2 — S3 + S4 on 12 cells:
    python -m example.experiments.h2_strategy_sweep --strategies S3 S4

Step 3 — Analyse (reads all saved CSVs):
    python -m example.experiments.h2_strategy_sweep --analyse

Or run everything at once (long runtime):
    python -m example.experiments.h2_strategy_sweep --strategies S1 S2 S3 S4

OUTPUT FILES
------------
results/sweep_S1.csv     — 24 rows, MSE at each ratio, strategy=S1
results/sweep_S2.csv     — 24 rows, MSE at each ratio, strategy=S2
results/sweep_S3.csv     — 12 rows, MSE at each ratio, strategy=S3
results/sweep_S4.csv     — 12 rows, MSE at each ratio, strategy=S4
results/h2_analysis.csv  — H2 per-cell verdict (after --analyse)

KEY DESIGN DECISIONS
--------------------
1. Distillation is run ONCE per (dataset, model) cell and the synthetic
   sequence is SHARED across all strategies. Any MSE difference between
   strategies is purely due to window selection — not distillation randomness.

2. expert_losses are computed ONCE after expert training and passed to all
   strategy evaluators. S4 uses them; S1/S2/S3 ignore them.

3. The same synthetic sequence, val set, test set, and model factory are used
   for all strategies. The only variable is self.selector inside the evaluator.

4. Mixing ratios are the full 10-point profile: 0%, 1%, 2%, 5%, 10%, 15%,
   20%, 30%, 50%, 100%. Both 0% and 100% are always included as anchors.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader
from ts_distill.models.factory import create_model
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.initializer.random_sample_initializer import RandomSampleInitializer
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher
from ts_distill.evaluation.evaluation import Evaluator

from ts_distill.evaluation.hybrid_evaluation.hybrid import (
    BaseHybridEvaluator,
    RandomAnchorSelector,
    StartExtendSelector,
    UniformStrideSelector,
    ImportanceWeightedSelector,
    DiversityAnchorSelector,
    compute_expert_losses,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# S1 + S2 run on all datasets; S3 + S4 on subset only
ALL_DATASETS    = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate',
                   'national_illness', 'weather']
SUBSET_DATASETS = ['ETTh1', 'ETTm1', 'exchange_rate']   # S3 + S4 scope

ALL_MODELS = ['DLinear', 'LSTM', 'MLP', 'CNN']

MIXING_RATIOS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)

# H2 decision threshold: strategy_delta > this → H2 supported for cell
H2_EPSILON = 0.01

DATASET_CONFIGS = {
    'ETTh1': {
        'csv_path':    'example/ETTh1.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12*30*24-96,             12*30*24+4*30*24-96],
        'border2s':    [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24],
        'in_features': 7,
    },
    'ETTh2': {
        'csv_path':    'example/ETTh2.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12*30*24-96,             12*30*24+4*30*24-96],
        'border2s':    [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24],
        'in_features': 7,
    },
    'ETTm1': {
        'csv_path':    'example/ETTm1.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12*30*96-96,             12*30*96+4*30*96-96],
        'border2s':    [12*30*96, 12*30*96+4*30*96, 12*30*96+8*30*96],
        'in_features': 7,
    },
    'ETTm2': {
        'csv_path':    'example/ETTm2.csv',
        'split_mode':  'benchmark_borders',
        'border1s':    [0, 12*30*96-96,             12*30*96+4*30*96-96],
        'border2s':    [12*30*96, 12*30*96+4*30*96, 12*30*96+8*30*96],
        'in_features': 7,
    },
    'exchange_rate': {
        'csv_path':    'example/exchange_rate.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 8,
    },
    'national_illness': {
        'csv_path':    'example/national_illness.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.6,
        'val_ratio':   0.2,
        'in_features': 7,
        'seq_len':     36,
        'pred_len':    36,
    },
    'weather': {
        'csv_path':    'example/weather.csv',
        'split_mode':  'ratios',
        'train_ratio': 0.7,
        'val_ratio':   0.1,
        'in_features': 21,
    },
}

MODEL_CONFIGS = {
    'DLinear': {'individual': False},
    'LSTM':    {'hidden_dim': 16, 'num_layer': 1},
    'MLP':     {},
    'CNN':     {},
}

DISTILL_CONFIG = {
    'seq_len':                96,
    'pred_len':               96,
    'in_features':             7,
    'expert_epochs':          80,
    'expert_lr':            0.01,
    'expert_momentum':       0.9,
    'n_distill_steps':       300,
    'n_synthetic':           384,
    'synthetic_lr':          0.1,
    'student_lr':           0.01,
    'student_steps':          20,
    'snapshot_student_steps': 50,
    'batch_size':             64,
    'trajectory_gap':          5,
    'eval_max_epochs':       300,
    'eval_lr':             0.001,
    'eval_batch_size':        32,
    'early_stop_patience':    10,
}

# Strategy name → selector instance mapping
# Each run uses a FRESH selector instance so seeds are independent
def make_selector(strategy_name: str) -> BaseHybridEvaluator:
    """Return the selector for the given strategy name."""
    if strategy_name == 'S1':
        return RandomAnchorSelector(selector_seed=99)
    elif strategy_name == 'S2':
        return StartExtendSelector()
    elif strategy_name == 'S3':
        return UniformStrideSelector()
    elif strategy_name == 'S4':
        return ImportanceWeightedSelector(selector_seed=99)
    elif strategy_name == 'S5':
        return DiversityAnchorSelector()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Use S1/S2/S3/S4/S5.")


# =============================================================================
# SINGLE CELL — runs distillation once then sweeps all requested strategies
# =============================================================================

def run_cell_all_strategies(
    dataset_name: str,
    model_name:   str,
    strategies:   list,
    cfg:          dict,
) -> dict:
    """
    Run distillation ONCE and then sweep all requested strategies.

    The synthetic sequence is shared — any MSE difference between strategies
    is purely due to window selection, not distillation randomness.

    Returns
    -------
    dict mapping strategy_name → evaluate_mixing() results dict
    e.g. {'S1': {'hybrid_5': {'MSE': 0.41}, ...},
          'S2': {'hybrid_5': {'MSE': 0.43}, ...}}
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Resolve per-dataset overrides ─────────────────────────────────────────
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])
    in_features = dataset_cfg.get('in_features', cfg['in_features'])
    window_size = seq_len + pred_len

    def make_model():
        return create_model(
            model_type   = model_name,
            seq_len      = seq_len,
            pred_len     = pred_len,
            in_features  = in_features,
            model_kwargs = MODEL_CONFIGS[model_name],
        )

    # ── Load and normalise data ───────────────────────────────────────────────
    df_raw = pd.read_csv(dataset_cfg['csv_path'])
    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values=values, window_size=window_size,
        seq_len=seq_len, dataset_cfg=dataset_cfg,
    )

    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data = scaler.transform(values)

    train_data     = torch.tensor(make_windows(data[train_start:train_end], window_size), dtype=torch.float32)
    val_data       = torch.tensor(make_windows(data[val_start:val_end],     window_size), dtype=torch.float32)
    test_data      = torch.tensor(make_windows(data[test_start:test_end],   window_size), dtype=torch.float32)
    raw_train_data = torch.tensor(data[train_start:train_end], dtype=torch.float32)

    # ── Expert training (ONCE — shared across all strategies) ─────────────────
    print(f"  [Expert] Training expert model...")
    recorder     = SimpleRecorder(record_every=1)
    expert_model = make_model()
    Trainer(
        model     = expert_model,
        optimizer = torch.optim.SGD(
            expert_model.parameters(),
            lr=cfg['expert_lr'], momentum=cfg['expert_momentum'],
        ),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    ).fit(
        dataloader = MiniBatchLoader(train_data, batch_size=cfg['batch_size']),
        epochs     = cfg['expert_epochs'],
        callbacks  = [recorder, SimpleCallback()],
    )

    # ── Distillation (ONCE — shared across all strategies) ────────────────────
    print(f"  [Distil] Running MTT distillation...")
    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(raw_train_data, cfg['n_synthetic'])

    synthetic_sequence = MTTDistiller(
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
        pred_len                = pred_len,
    ).distill(
        synthetic_init = synthetic_init,
        n_steps        = cfg['n_distill_steps'],
        val_data       = val_data,
    )

    # ── Expert losses for S4 (ONCE — shared if S4 is in strategies) ──────────
    expert_losses = None
    if 'S4' in strategies:
        print(f"  [S4]    Computing expert per-window losses...")
        expert_losses = compute_expert_losses(
            expert_model   = expert_model,
            raw_train_data = raw_train_data,
            window_size    = window_size,
            seq_len        = seq_len,
            batch_size     = cfg['batch_size'],
            device         = device,
        )
        print(f"          Computed losses for {len(expert_losses)} windows. "
              f"Mean={expert_losses.mean():.4f}  Max={expert_losses.max():.4f}")

    # ── Build val/test loaders (shared) ───────────────────────────────────────
    eval_val_loader  = TorchDataLoader(val_data,  batch_size=cfg['eval_batch_size'], shuffle=False)
    test_loader      = TorchDataLoader(test_data, batch_size=cfg['eval_batch_size'])

    # ── Run each strategy ─────────────────────────────────────────────────────
    strategy_results = {}

    for strategy_name in strategies:
        print(f"  [{strategy_name}]   Sweeping {len(MIXING_RATIOS)} ratios...")

        evaluator = BaseHybridEvaluator(
            anchor_selector     = make_selector(strategy_name),
            seq_len             = seq_len,
            batch_size          = cfg['batch_size'],
            eval_max_epochs     = cfg['eval_max_epochs'],
            early_stop_patience = cfg['early_stop_patience'],
            eval_lr             = cfg['eval_lr'],
            device              = device,
        )

        results = evaluator.evaluate_mixing(
            synthetic_data   = synthetic_sequence,
            real_train_data  = raw_train_data,
            real_val_data    = val_data,
            real_test_loader = test_loader,
            model_fn         = make_model,
            window_size      = window_size,
            mixing_ratios    = MIXING_RATIOS,
            expert_losses    = expert_losses,   # None for S1/S2/S3
        )

        strategy_results[strategy_name] = results

        # Print per-strategy best ratio
        non_zero = {k: v['MSE'] for k, v in results.items() if k != 'hybrid_0'}
        best_k   = min(non_zero, key=non_zero.get)
        print(f"          Best ratio: {best_k}  MSE={non_zero[best_k]:.6f}")

    return strategy_results


# =============================================================================
# CONVERT RESULTS TO CSV ROWS
# =============================================================================

def results_to_rows(
    dataset_name:     str,
    model_name:       str,
    strategy_name:    str,
    mixing_results:   dict,
) -> dict:
    """
    Convert evaluate_mixing() output to a flat CSV row dict.

    Columns: dataset, model, strategy, hybrid_0_mse, hybrid_1_mse, ...
    """
    row = {
        'dataset':  dataset_name,
        'model':    model_name,
        'strategy': strategy_name,
    }
    for key, val in mixing_results.items():
        pct = key.replace('hybrid_', '')
        row[f"hybrid_{pct}_mse"] = val['MSE']
    return row


# =============================================================================
# H2 ANALYSIS — reads saved sweep CSVs and computes H2 verdict
# =============================================================================

def analyse_h2(results_dir: Path):
    """
    Read sweep CSVs, compute H2 analysis for each cell, and save results.

    H2 is SUPPORTED for a cell if strategy_delta > H2_EPSILON (0.01).
    H2 is SUPPORTED at the matrix level if majority of cells are supported.
    """
    # Load available strategy CSVs
    strategy_dfs = {}
    for s in ['S1', 'S2', 'S3', 'S4']:
        path = results_dir / f'sweep_{s}.csv'
        if path.exists():
            strategy_dfs[s] = pd.read_csv(path)
            print(f"Loaded sweep_{s}.csv: {len(strategy_dfs[s])} rows")

    if len(strategy_dfs) < 2:
        print("Need at least 2 strategy CSVs for H2 analysis.")
        return

    # Collect all (dataset, model) cells present in ALL loaded strategies
    first_df = next(iter(strategy_dfs.values()))
    cells    = [(r['dataset'], r['model']) for _, r in first_df.iterrows()]

    ratio_cols = sorted(
        [c for c in first_df.columns if c.startswith('hybrid_') and c.endswith('_mse')
         and c not in ('hybrid_0_mse', 'hybrid_100_mse')],
        key=lambda c: int(c.replace('hybrid_','').replace('_mse',''))
    )

    h2_rows = []
    print(f"\n{'='*70}")
    print("H2 ANALYSIS")
    print(f"{'='*70}")

    for ds, mdl in cells:
        # Build strategy_results dict for this cell
        cell_results = {}
        available    = []
        for s, df in strategy_dfs.items():
            mask = (df['dataset'] == ds) & (df['model'] == mdl)
            if not mask.any():
                continue
            row = df[mask].iloc[0]
            res = {}
            for col in ratio_cols:
                pct   = col.replace('hybrid_','').replace('_mse','')
                key   = f"hybrid_{pct}"
                val   = row.get(col)
                if val is not None and not pd.isna(val):
                    res[key] = {'MSE': float(val)}
            # Add 0% and 100% if present
            for anchor in ['hybrid_0_mse', 'hybrid_100_mse']:
                if anchor in row.index and not pd.isna(row[anchor]):
                    k = anchor.replace('_mse','')
                    res[k] = {'MSE': float(row[anchor])}
            cell_results[s] = res
            available.append(s)

        if len(available) < 2:
            continue

        # Run H2 per-cell analysis
        evaluator = BaseHybridEvaluator(
            anchor_selector=RandomAnchorSelector(),
            seq_len=DISTILL_CONFIG['seq_len'],
        )
        h2 = evaluator.analyse_h2_cell(
            strategy_results = {s: cell_results[s] for s in available},
            dataset          = ds,
            model            = mdl,
            epsilon          = H2_EPSILON,
        )

        # Print per-cell
        verdict = "✓ SUPPORTED" if h2['h2_supported'] else "✗ not supported"
        print(f"\n  {ds} × {mdl}")
        print(f"  Strategies compared: {available}")
        print(f"  Ranking (best→worst): {h2['strategy_ranking']}")
        print(f"  Best={h2['best_strategy']}  Worst={h2['worst_strategy']}")
        print(f"  strategy_delta={h2['strategy_delta']:.6f}  "
              f"at ratio={h2['best_ratio_key']}")
        print(f"  Consistent ranking: {h2['consistent_ranking']}")
        print(f"  H2 verdict: {verdict}")

        # Build CSV row
        row_out = {
            'dataset':            ds,
            'model':              mdl,
            'strategies_compared': '+'.join(available),
            'best_strategy':      h2['best_strategy'],
            'worst_strategy':     h2['worst_strategy'],
            'strategy_delta':     round(h2['strategy_delta'], 6),
            'best_ratio_key':     h2['best_ratio_key'],
            'consistent_ranking': h2['consistent_ranking'],
            'h2_supported':       h2['h2_supported'],
        }
        # Add mean MSE per strategy
        for s in available:
            row_out[f'mean_mse_{s}'] = round(h2['mean_mse'].get(s, float('nan')), 6)
        h2_rows.append(row_out)

    # Matrix-level verdict
    df_h2    = pd.DataFrame(h2_rows)
    total    = len(df_h2)
    supported = df_h2['h2_supported'].sum()
    majority  = supported > total / 2

    print(f"\n{'='*70}")
    print(f"H2 MATRIX-LEVEL VERDICT")
    print(f"{'='*70}")
    print(f"  Total cells:        {total}")
    print(f"  H2 supported cells: {supported} / {total}")
    if majority:
        print(f"  ✓ H2 SUPPORTED: strategy matters in {supported}/{total} cells")
    else:
        print(f"  ✗ H2 NOT SUPPORTED: strategy matters in only {supported}/{total} cells")
    print(f"{'='*70}")

    # Save
    path = results_dir / 'h2_analysis.csv'
    df_h2.to_csv(path, index=False)
    print(f"\nH2 analysis saved → {path}")

    return df_h2


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="H2 Strategy Sweep")
    parser.add_argument(
        '--strategies', nargs='+', choices=['S1','S2','S3','S4'],
        default=['S1','S2'],
        help="Which strategies to run. Default: S1 S2"
    )
    parser.add_argument(
        '--analyse', action='store_true',
        help="Run H2 analysis on existing sweep CSVs (no new training)"
    )
    args = parser.parse_args()

    # Output stays under example/experiments/results/ even though this
    # module now lives in src/. __file__ is src/ts_distill/evaluation/
    # hybrid_evaluation/strategy_selection_to_mix.py, so 4 parents up is
    # the project root.
    results_dir = Path(__file__).resolve().parents[4] / 'example' / 'experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Analysis-only mode ────────────────────────────────────────────────────
    if args.analyse:
        print("\nRunning H2 analysis on existing sweep CSVs...")
        analyse_h2(results_dir)
        return

    # ── Determine which datasets to run ──────────────────────────────────────
    # S1 + S2 run on ALL datasets
    # S3 + S4 run on SUBSET only
    s12 = [s for s in args.strategies if s in ('S1','S2')]
    s34 = [s for s in args.strategies if s in ('S3','S4')]

    # ── Collect all rows per strategy ─────────────────────────────────────────
    sweep_rows = {s: [] for s in args.strategies}

    # Determine cell list
    cells = []
    for ds in ALL_DATASETS:
        for mdl in ALL_MODELS:
            needed_strats = []
            if s12 and ds in ALL_DATASETS:
                needed_strats += s12
            if s34 and ds in SUBSET_DATASETS:
                needed_strats += s34
            if needed_strats:
                cells.append((ds, mdl, needed_strats))

    total = len(cells)
    done  = 0

    for ds, mdl, cell_strategies in cells:
        done += 1
        print(f"\n{'='*70}")
        print(f"[{done}/{total}]  {ds} × {mdl}  strategies={cell_strategies}")
        print(f"{'='*70}")

        try:
            strategy_results = run_cell_all_strategies(
                dataset_name = ds,
                model_name   = mdl,
                strategies   = cell_strategies,
                cfg          = DISTILL_CONFIG,
            )

            # Collect rows for each strategy
            for strat_name, mixing_results in strategy_results.items():
                row = results_to_rows(ds, mdl, strat_name, mixing_results)
                sweep_rows[strat_name].append(row)

            # Save after every cell in case of interruption
            for strat_name in cell_strategies:
                if sweep_rows[strat_name]:
                    path = results_dir / f'sweep_{strat_name}.csv'
                    # Append if file exists, create if not
                    df_new = pd.DataFrame(sweep_rows[strat_name])
                    if path.exists():
                        df_existing = pd.read_csv(path)
                        # Remove duplicate rows for this cell if re-running
                        df_existing = df_existing[
                            ~((df_existing['dataset']==ds) &
                              (df_existing['model']==mdl))
                        ]
                        df_combined = pd.concat([df_existing, df_new.tail(1)], ignore_index=True)
                    else:
                        df_combined = df_new.tail(1)
                    df_combined.to_csv(path, index=False)

        except Exception as exc:
            print(f"\n  [SKIP] {ds} × {mdl} failed: {exc}")
            import traceback
            traceback.print_exc()
            continue

    # ── Final save ────────────────────────────────────────────────────────────
    for strat_name in args.strategies:
        path = results_dir / f'sweep_{strat_name}.csv'
        if sweep_rows[strat_name]:
            pd.DataFrame(sweep_rows[strat_name]).to_csv(path, index=False)
            print(f"\nSaved {path}  ({len(sweep_rows[strat_name])} rows)")

    print(f"\nSweep complete.")
    print("To run H2 analysis: python -m example.experiments.h2_strategy_sweep --analyse")


if __name__ == '__main__':
    # import sys
    # sys.argv = ['h2_strategy_sweep.py', '--strategies', 'S1', 'S2']
    # ALL_DATASETS[:] = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather']
    # ALL_MODELS[:] = ['DLinear', 'LSTM', 'MLP', 'CNN']
    # DISTILL_CONFIG['n_distill_steps'] = 10
    # DISTILL_CONFIG['expert_epochs']   = 5
    # DISTILL_CONFIG['eval_max_epochs'] = 20
    main()
