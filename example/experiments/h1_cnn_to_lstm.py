"""
H1 Cross-Architecture — CNN expert -> LSTM student (single pair)
==================================================================

Runs only the CNN -> LSTM cell of the cross-architecture matrix, using the
exact same functions, config, and evaluation protocol as
h1_cross_arch_baseline.py (early stopping on real val, patience=10, max 300
epochs, for both real and synthetic training).

Appends to the same results CSV (results/cross_arch_baseline_clean.csv) with
experiment="cross_arch_baseline_clean", method="standard_mtt_clean", so it is
directly comparable to the rest of the matrix.

Run
---
  python -m example.experiments.h1_cnn_to_lstm
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from example.experiments import h1_cross_arch_baseline as m

EXPERT_NAME  = "CNN"
#"DLinear"
STUDENT_NAME = "MLP"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Running {EXPERT_NAME} -> {STUDENT_NAME} only")

    cfg = m.DISTILL_CONFIG
    cfg["n_distill_steps"] = 1000

    for dataset_name in ["ETTh1"]:

        seeds = (
            m.SEEDS_MULTI
            if dataset_name in m.MULTI_SEED_DATASETS
            else m.SEEDS_SINGLE
        )

        data = m.load_dataset(dataset_name, cfg)

        for seed in seeds:

            print("\n" + "=" * 60)
            print(f"Dataset={dataset_name} | {EXPERT_NAME} -> {STUDENT_NAME} | seed={seed}")
            print("=" * 60)

            if m.row_exists(dataset_name, EXPERT_NAME, STUDENT_NAME, seed):
                print("  [RESUME] Already in CSV — skipping.")
                continue

            # ---------------------------------------------------------
            # Distill synthetic data from CNN expert
            # ---------------------------------------------------------
            try:
                synthetic_seq, n_pairs = m.distill_synthetic(
                    EXPERT_NAME, data, cfg, device, seed,
                )
            except Exception as exc:
                print(f"  [FAIL] Distillation: {exc}")
                m.append_csv({
                    "experiment":    "cross_arch_baseline_clean",
                    "dataset":       dataset_name,
                    "expert_model":  EXPERT_NAME,
                    "student_model": STUDENT_NAME,
                    "seed":          seed,
                    "method":        "standard_mtt_clean",
                    "notes":         f"distillation_failed: {str(exc)[:120]}",
                })
                continue

            # ---------------------------------------------------------
            # Real baseline for LSTM student
            # ---------------------------------------------------------
            print(f"\n  Computing real baseline for {STUDENT_NAME}...")
            try:
                real_mse = m.eval_on_real(STUDENT_NAME, data, cfg, device, seed)
            except Exception as exc:
                print(f"  [FAIL] real eval: {exc}")
                real_mse = float("nan")

            # ---------------------------------------------------------
            # Synthetic evaluation for LSTM student
            # ---------------------------------------------------------
            try:
                transfer_mse = m.eval_on_synthetic(
                    STUDENT_NAME, synthetic_seq, data, cfg, device, seed,
                )
            except Exception as exc:
                print(f"  [FAIL] synthetic eval: {exc}")
                m.append_csv({
                    "experiment":        "cross_arch_baseline_clean",
                    "dataset":           dataset_name,
                    "expert_model":      EXPERT_NAME,
                    "student_model":     STUDENT_NAME,
                    "seed":              seed,
                    "method":            "standard_mtt_clean",
                    "real_mse":          real_mse,
                    "n_pairs_available": n_pairs,
                    "notes":             f"synthetic_eval_failed: {str(exc)[:120]}",
                })
                continue

            mse_ratio = (
                transfer_mse / real_mse
                if (real_mse > 0 and not np.isnan(real_mse))
                else float("nan")
            )

            print(
                f"  real={real_mse:.6f} | "
                f"transfer={transfer_mse:.6f} | "
                f"ratio={mse_ratio:.3f}"
            )

            m.append_csv({
                "experiment":        "cross_arch_baseline_clean",
                "dataset":           dataset_name,
                "expert_model":      EXPERT_NAME,
                "student_model":     STUDENT_NAME,
                "seed":              seed,
                "method":            "standard_mtt_clean",
                "real_mse":          real_mse,
                "transfer_mse":      transfer_mse,
                "mse_ratio":         mse_ratio,
                "n_pairs_available": n_pairs,
                "notes":             "",
            })

    print("\nDone.")
    print(f"Results: {m.CSV_PATH}")


if __name__ == "__main__":
    main()
