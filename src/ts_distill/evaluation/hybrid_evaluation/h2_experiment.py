"""
h2_experiment.py — H2: Find Best Mixing Ratio per Architecture
===============================================================
Runs the full MTT distillation + hybrid mixing sweep for every
(dataset × model) cell and identifies the best mixing ratio r* for
each architecture — the ratio that minimises student test MSE.

UPDATED H2 DEFINITION
----------------------
For each dataset-architecture combination, there exists an interior
hybrid mixing ratio r* (0% < r* < 100%) that produces a lower final
test MSE than pure synthetic training at r = 0%, when evaluated
across the full ratio sweep.

NOTE: This version does NOT require r* to beat pure real data (100%).
It only finds the BEST ratio for each model. This answers the
practical question: given a distilled sequence, what fraction of real
data should I add for this architecture?

H2 DECISION RULE (updated)
----------------------------
H2 is SUPPORTED for a cell if:
    - best_mse < synth_mse (0% baseline)    → hybrid beats pure synthetic
    - r* is interior (0% < r* < 100%)       → a genuine mixing ratio is optimal
    - improvement > H2_EPSILON (0.005)      → improvement is above noise floor

H2 is SUPPORTED overall if majority (> 50%) of cells are supported.

Per-architecture best r* summary shows which ratio works best for
DLinear, LSTM, MLP, and CNN across all datasets.

HOW TO RUN
----------
Full matrix:
    python -m example.experiments.h2_experiment

Smoke test (fast, 1 cell):
    Uncomment the overrides at the bottom of this file.

OUTPUT
------
results/h2_best_ratios.csv      — per-cell: best r*, best MSE, improvement, verdict
results/h2_per_arch_summary.csv — per-architecture: mean r*, mean improvement
results/h2_full_sweep.csv       — full MSE at every ratio for every cell
Console prints full per-cell and per-architecture summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader
from ts_distill.models.factory import create_model
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.initializer.random_sample_initializer import RandomSampleInitializer
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher
from ts_distill.evaluation.hybrid_evaluation.hybrid import (
    BaseHybridEvaluator,
    RandomAnchorSelector,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ACTIVE_DATASETS = [
    'ETTh1'
]
ACTIVE_MODELS = ['DLinear', 'LSTM', 'MLP', 'CNN']

# Full 10-point sweep — 0% is the H2 baseline, 100% is included for reference
# but r* does NOT need to beat 100% under the new H2 definition
HYBRID_MIXING_RATIOS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)

# Minimum improvement over 0% baseline to count as H2 supported
H2_EPSILON = 0.005

# Output stays under example/experiments/ (where the CSV/dataset paths below
# are also anchored) even though this module now lives in src/. __file__ is
# src/ts_distill/evaluation/hybrid_evaluation/h2_experiment.py, so 4 parents
# up is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = _PROJECT_ROOT / 'example' / 'experiments' / 'h2_results'

DISTILL_CONFIG = {
    'seq_len':                96,
    'pred_len':               96,
    'in_features':             7,   # overridden per dataset
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


# =============================================================================
# PER-CELL RUNNER
# =============================================================================

def run_cell(dataset_name: str, model_name: str) -> dict:
    """
    Run full distillation + hybrid sweep for one (dataset, model) cell.

    Returns
    -------
    dict with keys:
        dataset, model,
        synth_mse        — MSE at ratio=0% (H2 baseline)
        real_mse         — MSE at ratio=100% (reference ceiling)
        best_ratio_pct   — percentage of the best interior r*
        best_ratio_key   — e.g. 'hybrid_15'
        best_mse         — MSE at best interior ratio
        improvement      — synth_mse - best_mse
        improvement_pct  — improvement / synth_mse * 100
        h2_supported     — True if improvement > H2_EPSILON and best is interior
        per_ratio        — {ratio_pct: mse} for all 10 ratios
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg        = DATASET_CONFIGS[dataset_name]
    seq_len    = cfg.get('seq_len',     DISTILL_CONFIG['seq_len'])
    pred_len   = cfg.get('pred_len',    DISTILL_CONFIG['pred_len'])
    in_features= cfg.get('in_features', DISTILL_CONFIG['in_features'])
    window_size= seq_len + pred_len

    # ── Load and normalise data ───────────────────────────────────────────────
    df_raw = pd.read_csv(cfg['csv_path'])
    values = df_raw.iloc[:, 1:].values.astype('float32')

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values=values, window_size=window_size,
        seq_len=seq_len, dataset_cfg=cfg,
    )

    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data = scaler.transform(values)

    train_data     = torch.tensor(make_windows(data[train_start:train_end], window_size), dtype=torch.float32)
    val_data       = torch.tensor(make_windows(data[val_start:val_end],     window_size), dtype=torch.float32)
    test_data      = torch.tensor(make_windows(data[test_start:test_end],   window_size), dtype=torch.float32)
    raw_train_data = torch.tensor(data[train_start:train_end], dtype=torch.float32)

    def make_model():
        return create_model(
            model_type   = model_name,
            seq_len      = seq_len,
            pred_len     = pred_len,
            in_features  = in_features,
            model_kwargs = MODEL_CONFIGS[model_name],
        )

    # ── Expert training ───────────────────────────────────────────────────────
    recorder     = SimpleRecorder(record_every=1)
    expert_model = make_model()
    Trainer(
        model     = expert_model,
        optimizer = torch.optim.SGD(
            expert_model.parameters(),
            lr=DISTILL_CONFIG['expert_lr'],
            momentum=DISTILL_CONFIG['expert_momentum'],
        ),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    ).fit(
        dataloader = MiniBatchLoader(train_data, batch_size=DISTILL_CONFIG['batch_size']),
        epochs     = DISTILL_CONFIG['expert_epochs'],
        callbacks  = [recorder, SimpleCallback()],
    )

    # ── MTT Distillation ──────────────────────────────────────────────────────
    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(raw_train_data, DISTILL_CONFIG['n_synthetic'])

    synthetic_sequence = MTTDistiller(
        initializer            = initializer,
        matcher                = MSEMatcher(),
        model_factory          = make_model,
        expert_recorder        = recorder,
        expert_epochs          = DISTILL_CONFIG['trajectory_gap'],
        syn_batch_size         = DISTILL_CONFIG['batch_size'],
        synthetic_lr           = DISTILL_CONFIG['synthetic_lr'],
        student_lr             = DISTILL_CONFIG['student_lr'],
        student_steps          = DISTILL_CONFIG['student_steps'],
        snapshot_student_steps = DISTILL_CONFIG['snapshot_student_steps'],
        seq_len                = seq_len,
        pred_len                = pred_len,
    ).distill(
        synthetic_init = synthetic_init,
        n_steps        = DISTILL_CONFIG['n_distill_steps'],
        val_data       = val_data,
    )

    # ── Hybrid sweep ──────────────────────────────────────────────────────────
    evaluator = BaseHybridEvaluator(
        anchor_selector     = RandomAnchorSelector(),
        seq_len             = seq_len,
        batch_size          = DISTILL_CONFIG['batch_size'],
        eval_max_epochs     = DISTILL_CONFIG['eval_max_epochs'],
        early_stop_patience = DISTILL_CONFIG['early_stop_patience'],
        eval_lr             = DISTILL_CONFIG['eval_lr'],
        device              = device,
    )

    test_loader = TorchDataLoader(
        test_data, batch_size=DISTILL_CONFIG['eval_batch_size']
    )

    results = evaluator.evaluate_mixing(
        synthetic_data   = synthetic_sequence,
        real_train_data  = raw_train_data,
        real_val_data    = val_data,
        real_test_loader = test_loader,
        model_fn         = make_model,
        window_size      = window_size,
        mixing_ratios    = HYBRID_MIXING_RATIOS,
    )

    # ── H2 Analysis (new definition) ──────────────────────────────────────────
    # Build per-ratio MSE dict
    per_ratio = {}
    for key, val in results.items():
        pct = int(key.replace('hybrid_', ''))
        per_ratio[pct] = val['MSE']

    synth_mse = per_ratio.get(0,   None)
    real_mse  = per_ratio.get(100, None)

    # Find best INTERIOR ratio (exclude 0% and 100%)
    interior = {pct: mse for pct, mse in per_ratio.items()
                if 0 < pct < 100}

    if not interior:
        return {
            'dataset': dataset_name, 'model': model_name,
            'synth_mse': synth_mse, 'real_mse': real_mse,
            'best_ratio_pct': None, 'best_ratio_key': None,
            'best_mse': None, 'improvement': None,
            'improvement_pct': None, 'h2_supported': False,
            'per_ratio': per_ratio,
        }

    best_ratio_pct = min(interior, key=interior.get)
    best_mse       = interior[best_ratio_pct]
    improvement    = (synth_mse - best_mse) if synth_mse is not None else None
    improvement_pct= (improvement / synth_mse * 100) if synth_mse else None

    # H2 supported under NEW DEFINITION:
    #   1. best interior ratio beats pure synthetic (improvement > epsilon)
    #   2. best ratio is genuinely interior (not 0% or 100%)
    h2_supported = (
        improvement is not None
        and improvement > H2_EPSILON
        and 0 < best_ratio_pct < 100
    )

    return {
        'dataset':        dataset_name,
        'model':          model_name,
        'synth_mse':      round(synth_mse, 6) if synth_mse else None,
        'real_mse':       round(real_mse,  6) if real_mse  else None,
        'best_ratio_pct': best_ratio_pct,
        'best_ratio_key': f"hybrid_{best_ratio_pct}",
        'best_mse':       round(best_mse, 6),
        'improvement':    round(improvement, 6) if improvement else None,
        'improvement_pct':round(improvement_pct, 2) if improvement_pct else None,
        'h2_supported':   h2_supported,
        'per_ratio':      per_ratio,
    }


# =============================================================================
# MATRIX RUNNER
# =============================================================================

def run_matrix() -> tuple:
    """Run all cells and return (cell_results, sweep_rows)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cell_results = []
    sweep_rows   = []
    total        = len(ACTIVE_DATASETS) * len(ACTIVE_MODELS)
    done         = 0

    for dataset_name in ACTIVE_DATASETS:
        for model_name in ACTIVE_MODELS:
            done += 1
            print(f"\n{'='*65}")
            print(f"[{done}/{total}]  H2 Sweep  {dataset_name} × {model_name}")
            print(f"{'='*65}")

            try:
                result = run_cell(dataset_name, model_name)
                cell_results.append(result)

                # Print per-cell result
                print(f"  Synth MSE (0%):   {result['synth_mse']:.6f}")
                print(f"  Best ratio:       {result['best_ratio_key']}  "
                      f"MSE={result['best_mse']:.6f}")
                print(f"  Improvement:      {result['improvement']:+.6f}  "
                      f"({result['improvement_pct']:.1f}%)")
                print(f"  Real MSE (100%):  {result['real_mse']:.6f}")
                verdict = "✓ SUPPORTED" if result['h2_supported'] else "✗ not supported"
                print(f"  H2 verdict:       {verdict}")

                # Per-ratio sweep row
                row = {
                    'dataset': dataset_name,
                    'model':   model_name,
                }
                for pct, mse in sorted(result['per_ratio'].items()):
                    row[f'ratio_{pct:03d}_pct'] = mse
                sweep_rows.append(row)

                # Save after every cell (safe against crashes)
                _save_intermediate(cell_results, sweep_rows)

            except Exception as exc:
                print(f"  [SKIP] Failed: {exc}")
                import traceback; traceback.print_exc()

    return cell_results, sweep_rows


# =============================================================================
# ANALYSIS — per-architecture best r* summary
# =============================================================================

def analyse_per_architecture(cell_results: list) -> pd.DataFrame:
    """
    For each architecture, compute:
        - mean best r* across datasets
        - std best r*
        - mean improvement %
        - how many cells have each ratio as best
        - H2 support rate
    """
    rows = []
    for model_name in ACTIVE_MODELS:
        cells = [r for r in cell_results if r['model'] == model_name
                 and r['best_ratio_pct'] is not None]
        if not cells:
            continue

        best_ratios   = [c['best_ratio_pct'] for c in cells]
        improvements  = [c['improvement_pct'] for c in cells if c['improvement_pct']]
        h2_supported  = sum(1 for c in cells if c['h2_supported'])

        # Most common best ratio for this architecture
        from collections import Counter
        most_common_ratio = Counter(best_ratios).most_common(1)[0][0]

        rows.append({
            'model':             model_name,
            'n_cells':           len(cells),
            'h2_supported':      h2_supported,
            'h2_support_rate':   round(h2_supported / len(cells) * 100, 1),
            'mean_best_r_pct':   round(float(np.mean(best_ratios)), 1),
            'std_best_r_pct':    round(float(np.std(best_ratios)),  1),
            'min_best_r_pct':    min(best_ratios),
            'max_best_r_pct':    max(best_ratios),
            'most_common_r_pct': most_common_ratio,
            'mean_improvement':  round(float(np.mean(improvements)), 2) if improvements else None,
        })

    return pd.DataFrame(rows)


def analyse_matrix(cell_results: list) -> pd.DataFrame:
    """H2 per-cell summary DataFrame."""
    rows = []
    for r in cell_results:
        rows.append({
            'dataset':        r['dataset'],
            'model':          r['model'],
            'synth_mse':      r['synth_mse'],
            'real_mse':       r['real_mse'],
            'best_ratio_pct': r['best_ratio_pct'],
            'best_ratio_key': r['best_ratio_key'],
            'best_mse':       r['best_mse'],
            'improvement':    r['improvement'],
            'improvement_pct':r['improvement_pct'],
            'h2_supported':   r['h2_supported'],
        })
    return pd.DataFrame(rows)


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(cell_results: list, df_arch: pd.DataFrame):
    total     = len(cell_results)
    supported = sum(1 for r in cell_results if r['h2_supported'])

    print(f"\n{'='*65}")
    print(f"H2 RESULTS SUMMARY  (new definition — beats pure synthetic only)")
    print(f"{'='*65}")
    print(f"\n  Decision rule: best interior r* must improve over 0% by > {H2_EPSILON}")
    print(f"  Total cells:      {total}")
    print(f"  H2 supported:     {supported} / {total}")
    print(f"  H2 verdict:       {'✓ SUPPORTED' if supported > total/2 else '✗ NOT SUPPORTED'}")

    print(f"\n  Per-cell best ratios:")
    print(f"  {'Dataset':<18} {'Model':<10} {'Best r*':>8} {'Improvement':>12} {'H2?':>12}")
    print(f"  {'-'*62}")
    for r in cell_results:
        v = "✓" if r['h2_supported'] else "✗"
        pct_str = f"{r['best_ratio_pct']}%" if r['best_ratio_pct'] else "N/A"
        imp_str = f"{r['improvement_pct']:.1f}%" if r['improvement_pct'] else "N/A"
        print(f"  {r['dataset']:<18} {r['model']:<10} {pct_str:>8} {imp_str:>12} {v:>12}")

    print(f"\n  {'='*65}")
    print(f"  PER-ARCHITECTURE BEST r* SUMMARY")
    print(f"  {'='*65}")
    print(f"  {'Model':<10} {'Mean r*':>8} {'±Std':>6} {'Range':>14} "
          f"{'Most Common':>12} {'Mean Imprv':>11} {'H2 Rate':>9}")
    print(f"  {'-'*74}")
    for _, row in df_arch.iterrows():
        print(f"  {row['model']:<10} "
              f"{row['mean_best_r_pct']:>7.1f}% "
              f"{row['std_best_r_pct']:>5.1f}% "
              f"{row['min_best_r_pct']:>5}%–{row['max_best_r_pct']}%"
              f"{' ':>4}"
              f"{row['most_common_r_pct']:>9}% "
              f"{row['mean_improvement']:>9.1f}% "
              f"{row['h2_supported']}/{row['n_cells']:>2}")
    print(f"  {'='*65}")


# =============================================================================
# SAVE
# =============================================================================

def _save_intermediate(cell_results, sweep_rows):
    """Save after every cell so crashes don't lose progress."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cell_results:
        analyse_matrix(cell_results).to_csv(
            RESULTS_DIR / 'h2_best_ratios.csv', index=False
        )
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(
            RESULTS_DIR / 'h2_full_sweep.csv', index=False
        )

def save_all(cell_results, sweep_rows, df_arch):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    analyse_matrix(cell_results).to_csv(
        RESULTS_DIR / 'h2_best_ratios.csv', index=False
    )
    print(f"  Saved: results/h2_best_ratios.csv")

    pd.DataFrame(sweep_rows).to_csv(
        RESULTS_DIR / 'h2_full_sweep.csv', index=False
    )
    print(f"  Saved: results/h2_full_sweep.csv")

    df_arch.to_csv(
        RESULTS_DIR / 'h2_per_arch_summary.csv', index=False
    )
    print(f"  Saved: results/h2_per_arch_summary.csv")


# =============================================================================
# ANALYSIS-ONLY MODE (reads existing h2_best_ratios.csv, no new training)
# =============================================================================

def run_analysis_only():
    """Read saved CSVs and reprint summary — no new training."""
    path = RESULTS_DIR / 'h2_best_ratios.csv'
    if not path.exists():
        print(f"ERROR: {path} not found. Run the full sweep first.")
        return

    df = pd.read_csv(path)
    cell_results = df.to_dict('records')
    for r in cell_results:
        r['h2_supported'] = bool(r['h2_supported'])

    df_arch = analyse_per_architecture(cell_results)
    print_summary(cell_results, df_arch)
    df_arch.to_csv(RESULTS_DIR / 'h2_per_arch_summary.csv', index=False)
    print(f"\n  Architecture summary saved → results/h2_per_arch_summary.csv")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="H2 Best Ratio Experiment")
    parser.add_argument(
        '--analyse', action='store_true',
        help='Re-run analysis only from saved h2_best_ratios.csv (no training)'
    )
    args = parser.parse_args()

    if args.analyse:
        print("\nAnalysis-only mode — reading from saved CSVs...")
        run_analysis_only()
        return

    print("\n" + "="*65)
    print("H2 EXPERIMENT — Find Best Mixing Ratio per Architecture")
    print("="*65)
    print(f"  Definition: best interior r* that beats pure synthetic (0%)")
    print(f"  Threshold:  improvement > {H2_EPSILON}")
    print(f"  Datasets:   {ACTIVE_DATASETS}")
    print(f"  Models:     {ACTIVE_MODELS}")
    print(f"  Ratios:     {[int(r*100) for r in HYBRID_MIXING_RATIOS]}%")
    print("="*65)

    cell_results, sweep_rows = run_matrix()

    if not cell_results:
        print("\nNo results — all cells failed.")
        return

    df_arch = analyse_per_architecture(cell_results)
    print_summary(cell_results, df_arch)
    save_all(cell_results, sweep_rows, df_arch)

    print("\nH2 experiment complete.")


if __name__ == '__main__':
    # ── Smoke test overrides — uncomment for quick test ──────────────────────
    import sys
    sys.argv = ['h2_experiment.py']
    ACTIVE_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather']
    ACTIVE_MODELS = ['DLinear', 'LSTM', 'MLP', 'CNN']
    DISTILL_CONFIG['n_distill_steps'] = 10
    DISTILL_CONFIG['expert_epochs']   = 5
    DISTILL_CONFIG['eval_max_epochs'] = 20
    # ─────────────────────────────────────────────────────────────────────────
    main()
