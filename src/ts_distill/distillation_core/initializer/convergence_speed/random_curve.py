"""
Random Initializer — Convergence Curve
=======================================
Loads the steps-vs-MSE CSVs produced by experiment_matrix.py's convergence
sweep (--mode convergence, ACTIVE_INITIALIZER='random'), computes the Area
Under the Curve (AUC) per (dataset, model) pair, and plots the convergence
curve(s). A smaller AUC means the model spent less time at higher error,
i.e. faster convergence.

Run from the project root, after generating the CSVs via:
    python -m example.experiments.experiment_matrix --mode convergence
(with ACTIVE_INITIALIZER = 'random' in experiment_matrix.py)

    python -m ts_distill.distillation_core.initializer.convergence_speed.random_curve
"""

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ts_distill.distillation_core.initializer.convergence_speed.base import calculate_convergence_auc

INITIALIZER_NAME = 'random'
RESULTS_DIR      = Path(__file__).resolve().parents[5] / 'example' / 'experiments' / 'results'


def load_curves(results_dir: Path = RESULTS_DIR) -> dict:
    """Returns {'<dataset> x <model>': DataFrame} for every CSV matching this initializer."""
    pattern = str(results_dir / f'{INITIALIZER_NAME}_convergence_*.csv')
    curves = {}
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path).sort_values('n_distill_steps')
        label = f"{df['dataset'].iloc[0]} x {df['model'].iloc[0]}"
        curves[label] = df
    return curves


def main() -> None:
    curves = load_curves()
    if not curves:
        print(f"No CSVs found matching '{INITIALIZER_NAME}_convergence_*.csv' in {RESULTS_DIR}")
        return

    plt.figure(figsize=(10, 6))
    print(f"--- {INITIALIZER_NAME.title()} Initializer — Convergence AUC ---")
    for label, df in curves.items():
        auc = calculate_convergence_auc(df['n_distill_steps'], df['transfer_mse'])
        print(f"{label:<20s} AUC = {auc:.4f}")
        plt.plot(df['n_distill_steps'], df['transfer_mse'], marker='o', linewidth=2, label=label)
        plt.fill_between(df['n_distill_steps'], df['transfer_mse'], alpha=0.1)

    plt.title(f'Convergence Speed ({INITIALIZER_NAME.title()} Initializer)')
    plt.xlabel('Distillation Steps')
    plt.ylabel('Downstream Test MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    out_path = RESULTS_DIR / f'{INITIALIZER_NAME}_convergence_curve.png'
    plt.savefig(out_path)
    print(f"\nCurve plot saved -> {out_path}")


if __name__ == '__main__':
    main()
