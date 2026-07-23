"""
optimal_ratio_finder.py
========================
Finds the mixing ratio r* that jointly maximises compression (as little
real data as possible) and minimises MSE, using the real mixing-curve
data already produced by prior sweeps (e.g. h2_strategy_selection/*/
metrics_hybrid.csv). Does not modify ratio_predictor.py.

METHOD — compromise programming (ideal-point method)
------------------------------------------------------
For each (dataset, model) cell we have real, measured points
(r, MSE(r)) at every ratio that was actually tested (r=0 -> pure
synthetic, r=100 -> pure real, plus whatever intermediate ratios exist).

We do NOT want to minimise MSE alone (that trivially picks r=100, i.e.
"just use all the real data" — zero compression). We want the point
that is simultaneously close to the best possible MSE AND close to
r=0% real data (maximum compression). This is a two-objective trade-off,
solved with the standard "distance to the ideal/utopia point" method
(Zeleny 1973 "compromise programming"; also known as TOPSIS's core
step):

    1. Interpolate the tested points into a smooth curve MSE(r)
       (monotone cubic / PCHIP — avoids the overshoot a plain cubic
       spline would introduce between sparse points).
    2. Normalise both axes to [0, 1] over that cell's own curve:
           mse_norm(r) = (MSE(r) - MSE_min) / (MSE_max - MSE_min)
           r_norm(r)   = r / 100                    (0 = full compression)
    3. The "ideal point" is (r_norm=0, mse_norm=0) — zero real data AND
       the best MSE on the curve, which is generally unreachable by any
       single r. Compute the Euclidean distance from every point on the
       curve to that ideal point:
           distance(r) = sqrt( (w_mse * mse_norm(r))**2
                              + (w_compression * r_norm(r))**2 )
    4. r* = argmin distance(r) over the interpolated curve.

Equal weights (w_mse = w_compression = 1.0) treat "1% better than the
worst MSE on this curve" and "1% more compression" as equally
valuable — adjust the weights below if you want to bias the trade-off
towards accuracy or towards compression.

Usage
-----
    python -m example.experiments.optimal_ratio_finder
    python -m example.experiments.optimal_ratio_finder --csv path/to/metrics_hybrid.csv

Output
------
    results/optimal_ratio/<dataset>_<model>_optimal_ratio.png   — one curve per cell
    results/optimal_ratio/optimal_ratios_summary.csv            — one row per cell
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# Output stays under example/experiments/ even though this module now lives
# in src/. __file__ is src/ts_distill/evaluation/hybrid_evaluation/
# optimal_ratio_finder.py, so 4 parents up is the project root.
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[4] / 'example' / 'experiments'
DEFAULT_CSV = (
    _EXPERIMENTS_DIR / 'results' / 'h2_strategy_selection'
    / 'S1_Random' / 'metrics_hybrid.csv'
)
OUTPUT_DIR = _EXPERIMENTS_DIR / 'results' / 'optimal_ratio'

# Trade-off weights: how much a unit of normalised MSE improvement
# matters relative to a unit of normalised compression improvement.
W_MSE = 1.0
W_COMPRESSION = 1.0

RATIO_COL_RE = re.compile(r'^hybrid_(\d+)_mse$')


def extract_ratio_curve(row: pd.Series) -> tuple:
    """Pull (ratios, mses) out of a row's hybrid_<r>_mse columns, sorted by ratio."""
    points = []
    for col in row.index:
        m = RATIO_COL_RE.match(col)
        if not m:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        points.append((int(m.group(1)), float(val)))

    points.sort(key=lambda p: p[0])
    ratios = [p[0] for p in points]
    mses = [p[1] for p in points]
    return ratios, mses


def interpolate_curve(ratios: list, mses: list, n_points: int = 1000) -> tuple:
    """Return a fine (r, MSE) grid spanning the tested ratio range."""
    r_grid = np.linspace(min(ratios), max(ratios), n_points)

    if _HAVE_SCIPY and len(ratios) >= 3:
        interpolator = PchipInterpolator(ratios, mses)
        mse_grid = interpolator(r_grid)
    else:
        mse_grid = np.interp(r_grid, ratios, mses)

    return r_grid, mse_grid


def find_optimal_ratio(
    ratios: list,
    mses: list,
    w_mse: float = W_MSE,
    w_compression: float = W_COMPRESSION,
    n_points: int = 1000,
) -> dict:
    """
    Find r* minimising distance to the ideal point (r=0, MSE=MSE_min)
    on the interpolated curve. See module docstring for the method.

    n_points controls the resolution of the search grid — exposed so
    callers (e.g. a grid-convergence check) can verify r* is stable as
    resolution increases, not a discretisation artifact.
    """
    r_grid, mse_grid = interpolate_curve(ratios, mses, n_points=n_points)

    mse_min, mse_max = mse_grid.min(), mse_grid.max()
    mse_norm = (mse_grid - mse_min) / (mse_max - mse_min) if mse_max > mse_min else np.zeros_like(mse_grid)
    r_norm = r_grid / 100.0

    distance = np.sqrt((w_mse * mse_norm) ** 2 + (w_compression * r_norm) ** 2)
    best_idx = int(np.argmin(distance))

    r_star = float(r_grid[best_idx])
    mse_at_r_star = float(mse_grid[best_idx])

    # Nearest ratio that was actually tested (for a realistic recommendation).
    nearest_tested_idx = int(np.argmin([abs(r - r_star) for r in ratios]))

    return {
        'r_star': r_star,
        'mse_at_r_star': mse_at_r_star,
        'compression_pct': 100.0 - r_star,
        'distance': float(distance[best_idx]),
        'nearest_tested_ratio': ratios[nearest_tested_idx],
        'nearest_tested_mse': mses[nearest_tested_idx],
        'r_grid': r_grid,
        'mse_grid': mse_grid,
        'distance_grid': distance,
        'best_idx': best_idx,
    }


def plot_cell(dataset: str, model: str, ratios: list, mses: list, result: dict, output_path: Path,
              real_mse: float = None, synth_mse: float = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(result['r_grid'], result['mse_grid'], '-', color='#9AA5B1', linewidth=1.5,
             label='Interpolated MSE(r)', zorder=2)
    ax.plot(ratios, mses, 'o', color='steelblue', markersize=6,
             label='Tested ratios', zorder=3)

    if synth_mse is not None:
        ax.axhline(synth_mse, color='#CC0000', linestyle='--', linewidth=1.3,
                   label=f'Pure synthetic (0%): {synth_mse:.4f}', zorder=1)
    if real_mse is not None:
        ax.axhline(real_mse, color='#217346', linestyle='--', linewidth=1.3,
                   label=f'Pure real (100%): {real_mse:.4f}', zorder=1)

    r_star = result['r_star']
    ax.axvline(r_star, color='#FF8C00', linestyle=':', linewidth=1.8, zorder=4)
    ax.scatter([r_star], [result['mse_at_r_star']], color='#FF8C00', s=260,
               marker='*', zorder=6,
               label=f"Optimal r*={r_star:.1f}%  "
                     f"(compression={result['compression_pct']:.1f}%, "
                     f"MSE={result['mse_at_r_star']:.4f})")

    ax.annotate(
        f"  r*={r_star:.1f}%\n  compression={result['compression_pct']:.1f}%\n  MSE={result['mse_at_r_star']:.4f}",
        xy=(r_star, result['mse_at_r_star']),
        xytext=(r_star + max(2, max(ratios) * 0.04), result['mse_at_r_star']),
        fontsize=8.5, color='#FF8C00', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.2),
    )

    ax.set_xlabel('Real Data Percentage (%)', fontsize=11)
    ax.set_ylabel('Test MSE', fontsize=11)
    ax.set_title(f'{dataset} × {model} — Optimal Mixing Ratio\n'
                 f'(compromise between compression and MSE)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8.5, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Find compression/MSE-optimal mixing ratios')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                         help='CSV with dataset, model, and hybrid_<r>_mse columns')
    parser.add_argument('--w-mse', type=float, default=W_MSE)
    parser.add_argument('--w-compression', type=float, default=W_COMPRESSION)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    print(f"{'Dataset':<18} {'Model':<10} {'r*':>7} {'Compression':>12} {'MSE@r*':>10}")
    print('-' * 62)

    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model']
        ratios, mses = extract_ratio_curve(row)

        if len(ratios) < 2:
            print(f"  [SKIP] {dataset} x {model} — fewer than 2 ratio points")
            continue

        result = find_optimal_ratio(ratios, mses, args.w_mse, args.w_compression)

        print(f"{dataset:<18} {model:<10} {result['r_star']:>6.1f}% "
              f"{result['compression_pct']:>11.1f}% {result['mse_at_r_star']:>10.4f}")

        mse_at_r100 = mses[ratios.index(100)] if 100 in ratios else None
        mse_at_r0 = mses[ratios.index(0)] if 0 in ratios else None

        out_path = OUTPUT_DIR / f"{dataset}_{model}_optimal_ratio.png"
        plot_cell(dataset, model, ratios, mses, result, out_path,
                  real_mse=mse_at_r100, synth_mse=mse_at_r0)

        summary_rows.append({
            'dataset': dataset,
            'model': model,
            'optimal_ratio_pct': round(result['r_star'], 2),
            'compression_pct': round(result['compression_pct'], 2),
            'mse_at_optimal': round(result['mse_at_r_star'], 6),
            'nearest_tested_ratio': result['nearest_tested_ratio'],
            'nearest_tested_mse': round(result['nearest_tested_mse'], 6),
            'mse_at_real_only_100pct': round(mse_at_r100, 6) if mse_at_r100 is not None else None,
            'mse_at_synthetic_only_0pct': round(mse_at_r0, 6) if mse_at_r0 is not None else None,
        })

    summary_path = OUTPUT_DIR / 'optimal_ratios_summary.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nSummary saved  → {summary_path}")
    print(f"Plots saved    → {OUTPUT_DIR}/*_optimal_ratio.png")


if __name__ == '__main__':
    main()
