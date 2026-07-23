"""
ratio_predictor_verifier.py
=============================
Backtests ratio_predictor.py's predict_r_star() against the 20 cells that
already have a full H2 sweep (h2_best_ratios.csv / h2_full_sweep.csv), so
its heuristic can be judged against real measured MSE instead of trusted
blindly.

WHY LEAVE-ONE-OUT
------------------
predict_r_star()'s architecture-specific ratio range [r_min, r_max] comes
from h2_per_arch_summary.csv, which is itself DERIVED from these same 20
cells. Predicting a cell's r* using a range that was partly built FROM
that cell's own best ratio is leakage — it makes the prediction look
better than it would be on a genuinely new dataset. To avoid that, for
every cell under test we recompute the per-architecture range from the
OTHER cells only (excluding that cell's dataset), via
analyse_per_architecture() from h2_experiment.py, and inject it with
predict_r_star(..., arch_ratio_range_override=...).

WHAT THIS CHECKS
----------------
For each cell, two different notions of "ground truth" are compared
against the predicted r*:

  1. Discrete best ratio (best_ratio_pct / best_mse from h2_best_ratios.csv)
     — the best of only the 10 ratios that were actually tested (H2's own
     definition).
  2. Continuous optimum (from optimal_ratio_finder.py's compromise-
     programming method) — the true minimiser of the interpolated
     MSE(r) curve, not limited to the 10 tested points.

"Regret" = MSE at the predicted ratio (interpolated from the real sweep
curve) minus MSE at the continuous optimum. This matters more than raw
ratio distance: a curve that is flat near its optimum makes a 10-point
ratio miss nearly free, while a steep curve makes a 2-point miss costly.

WHAT THIS DOES NOT CHECK
-------------------------
This only backtests against the 20 ETTh1/ETTh2/ETTm1/ETTm2/weather cells
that already have full sweeps — it says nothing about how the predictor
will do on a truly new, never-swept dataset or architecture. Nor does it
retrain anything; it only re-derives predictions and compares against
sweep data that already exists on disk.

Usage
-----
    python -m example.experiments.ratio_predictor_verifier
    python -m example.experiments.ratio_predictor_verifier --regret-tol-pct 5

Output
------
    results/ratio_predictor_verification/backtest_report.csv
    results/ratio_predictor_verification/RATIO_PREDICTOR_VERIFICATION.md
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ts_distill.evaluation.hybrid_evaluation.ratio_predictor import predict_r_star, DEFAULT_WEIGHTS
from ts_distill.evaluation.hybrid_evaluation.optimal_ratio_finder import interpolate_curve, find_optimal_ratio, W_MSE, W_COMPRESSION
from ts_distill.evaluation.hybrid_evaluation.h2_experiment import analyse_per_architecture

# Output/input data stays under example/experiments/ even though this module
# now lives in src/. __file__ is src/ts_distill/evaluation/hybrid_evaluation/
# ratio_predictor_verifier.py, so 4 parents up is the project root.
RESULTS_DIR   = Path(__file__).resolve().parents[4] / 'example' / 'experiments' / 'results'
H2_DIR        = RESULTS_DIR / 'h2_results'
BEST_RATIOS_CSV = H2_DIR / 'h2_best_ratios.csv'
FULL_SWEEP_CSV  = H2_DIR / 'h2_full_sweep.csv'
INPUTS_DIR      = RESULTS_DIR / 'predictor_inputs'
OUTPUT_DIR      = RESULTS_DIR / 'ratio_predictor_verification'

CHANNEL_AWARE_MIN_CHANNELS = 3   # use channel-aware divergence/structure above this
DEFAULT_REGRET_TOL_PCT = 10.0    # max tolerable MSE regret (% of optimum) to "pass"

SWEEP_COL_RE = re.compile(r'^ratio_(\d+)_pct$')


def extract_sweep_curve(row: pd.Series) -> tuple:
    """Pull (ratios, mses) out of a row's ratio_XXX_pct columns, sorted by ratio."""
    points = []
    for col in row.index:
        m = SWEEP_COL_RE.match(col)
        if not m:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        points.append((int(m.group(1)), float(val)))
    points.sort(key=lambda p: p[0])
    return [p[0] for p in points], [p[1] for p in points]


def build_loo_arch_ranges(cell_results: list, exclude_dataset: str, exclude_model: str) -> dict:
    """{model: (min_r, max_r)} recomputed with this cell's own dataset excluded."""
    loo_cells = [
        c for c in cell_results
        if not (c['dataset'] == exclude_dataset and c['model'] == exclude_model)
    ]
    df_arch_loo = analyse_per_architecture(loo_cells)
    return {
        r['model']: (int(r['min_best_r_pct']), int(r['max_best_r_pct']))
        for _, r in df_arch_loo.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Backtest ratio_predictor.py against real H2 sweep data (leave-one-out)'
    )
    parser.add_argument('--regret-tol-pct', type=float, default=DEFAULT_REGRET_TOL_PCT,
                         help='Max tolerable MSE regret vs. the continuous optimum, as %% of optimum MSE')
    args = parser.parse_args()

    if not BEST_RATIOS_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {BEST_RATIOS_CSV}. Run h2_experiment.py first.")
    if not FULL_SWEEP_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {FULL_SWEEP_CSV}. Run h2_experiment.py first.")

    best_df  = pd.read_csv(BEST_RATIOS_CSV)
    sweep_df = pd.read_csv(FULL_SWEEP_CSV)
    df = best_df.merge(sweep_df, on=['dataset', 'model'], how='left')

    cell_results = best_df.to_dict('records')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'Dataset':<18} {'Model':<8} {'Pred':>6} {'Discrete*':>10} {'Continuous*':>12} "
          f"{'Regret%':>9} {'Pass':>6}")
    print('-' * 78)

    for _, row in df.iterrows():
        dataset, model = row['dataset'], row['model']

        inputs_path = INPUTS_DIR / f'{dataset}_{model}_inputs.npz'
        if not inputs_path.exists():
            print(f"  [SKIP] {dataset} x {model} — missing {inputs_path.name}")
            continue

        npz = np.load(inputs_path, allow_pickle=True)
        real_train, synthetic = npz['real_train'], npz['synthetic']

        arch_ranges_loo = build_loo_arch_ranges(cell_results, dataset, model)

        channel_aware = real_train.ndim > 1 and real_train.shape[-1] > CHANNEL_AWARE_MIN_CHANNELS

        predicted_r = predict_r_star(
            synth_mse   = float(row['synth_mse']),
            real_mse    = float(row['real_mse']),
            model_name  = model,
            real_train  = real_train,
            synthetic   = synthetic,
            weights     = DEFAULT_WEIGHTS,
            verbose     = False,
            channel_aware = channel_aware,
            arch_ratio_range_override = arch_ranges_loo,
        )

        ratios, mses = extract_sweep_curve(row)
        if len(ratios) < 2:
            print(f"  [SKIP] {dataset} x {model} — fewer than 2 ratio points in sweep")
            continue

        r_grid, mse_grid = interpolate_curve(ratios, mses, n_points=2000)
        mse_at_predicted = float(np.interp(predicted_r, r_grid, mse_grid))

        # Ground truth = optimal_ratio_finder.py's own compromise-programming
        # optimum (same W_MSE/W_COMPRESSION defaults), NOT a pure-MSE minimum —
        # r=100% (all real data) would trivially win on MSE alone, but that
        # defeats the whole point of predicting a compressed mixing ratio.
        continuous = find_optimal_ratio(ratios, mses, w_mse=W_MSE, w_compression=W_COMPRESSION, n_points=2000)
        mse_at_continuous_optimum = continuous['mse_at_r_star']
        r_at_continuous_optimum   = continuous['r_star']

        regret = max(0.0, mse_at_predicted - mse_at_continuous_optimum)
        regret_pct = (regret / mse_at_continuous_optimum * 100.0) if mse_at_continuous_optimum > 0 else 0.0

        discrete_best_r   = int(row['best_ratio_pct'])
        discrete_best_mse = float(row['best_mse'])
        ratio_gap_vs_discrete = abs(predicted_r - discrete_best_r)

        passed = regret_pct <= args.regret_tol_pct

        print(f"{dataset:<18} {model:<8} {predicted_r:>5}% {discrete_best_r:>9}% "
              f"{r_at_continuous_optimum:>11.1f}% {regret_pct:>8.2f}% {'PASS' if passed else 'FAIL':>6}")

        rows.append({
            'dataset': dataset,
            'model': model,
            'predicted_r_pct': predicted_r,
            'discrete_best_r_pct': discrete_best_r,
            'discrete_best_mse': round(discrete_best_mse, 6),
            'continuous_optimum_r_pct': round(r_at_continuous_optimum, 2),
            'continuous_optimum_mse': round(mse_at_continuous_optimum, 6),
            'mse_at_predicted': round(mse_at_predicted, 6),
            'regret': round(regret, 6),
            'regret_pct': round(regret_pct, 2),
            'ratio_gap_vs_discrete': ratio_gap_vs_discrete,
            'passed': passed,
        })

    report_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'backtest_report.csv'
    report_df.to_csv(csv_path, index=False)

    n = len(report_df)
    pass_rate = round(report_df['passed'].sum() / n * 100, 1) if n else 0.0
    mean_regret = round(report_df['regret_pct'].mean(), 2) if n else 0.0
    median_regret = round(report_df['regret_pct'].median(), 2) if n else 0.0
    worst = report_df.sort_values('regret_pct', ascending=False).head(5) if n else report_df

    print(f"\n{n} cells backtested — pass rate (regret <= {args.regret_tol_pct}%): {pass_rate}%")
    print(f"Mean regret: {mean_regret}%   Median regret: {median_regret}%")

    md_lines = [
        "# ratio_predictor.py — Leave-One-Out Backtest\n",
        f"Backtests `predict_r_star()` against {n} cells that already have a full H2 sweep. "
        f"Architecture ratio ranges are recomputed leave-one-out per cell (see module "
        f"docstring) so the architecture prior is genuinely out-of-sample for the dataset "
        f"being predicted.\n",
        "This does **not** test generalisation to unswept datasets/architectures — only "
        "whether the heuristic tracks real measured MSE on data it hasn't directly seen "
        "(within the existing 20-cell pool).\n",
        f"## Summary\n",
        f"- Pass rate (regret ≤ {args.regret_tol_pct}% of optimum MSE): **{pass_rate}%**",
        f"- Mean regret: {mean_regret}%",
        f"- Median regret: {median_regret}%\n",
        "## Per-cell results\n",
        "| Dataset | Model | Predicted r* | Discrete best r* | Continuous optimum r* | "
        "Regret % | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['model']} | {r['predicted_r_pct']}% | "
            f"{r['discrete_best_r_pct']}% | {r['continuous_optimum_r_pct']:.1f}% | "
            f"{r['regret_pct']:.2f}% | {'PASS' if r['passed'] else 'FAIL'} |"
        )

    if n:
        md_lines.append("\n## Worst 5 cells by regret\n")
        md_lines.append("| Dataset | Model | Predicted r* | Continuous optimum r* | Regret % |")
        md_lines.append("|---|---|---|---|---|")
        for _, r in worst.iterrows():
            md_lines.append(
                f"| {r['dataset']} | {r['model']} | {r['predicted_r_pct']}% | "
                f"{r['continuous_optimum_r_pct']:.1f}% | {r['regret_pct']:.2f}% |"
            )

    md_path = OUTPUT_DIR / 'RATIO_PREDICTOR_VERIFICATION.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f"\nReport saved  → {csv_path}")
    print(f"Summary saved → {md_path}")


if __name__ == '__main__':
    main()
