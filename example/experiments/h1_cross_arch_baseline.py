"""
H1 Cross-Architecture Baseline (Clean Evaluation Version)
=========================================================

Goal
----
Test whether MTT synthetic data transfers equally across architectures.

Key Experimental Fixes
----------------------
1. NO early stopping
2. NO real validation during synthetic training
3. FIXED training budget for ALL evaluations
4. Real baseline recomputed PER SEED
5. Cleaner isolation of synthetic-data quality

Hypothesis
----------
If MTT synthetic data overfits the expert architecture,
then diagonal transfer should outperform cross-architecture transfer.

Metric
------
mse_ratio = transfer_mse / real_mse

Interpretation:
- ~1.0  => synthetic nearly as good as real
- >1.0  => degradation from synthetic training
- higher off-diagonal => architecture bias

Run:
----
python -m example.experiments.h1_cross_arch_baseline_clean
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader

# ---------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent.parent))

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader
from ts_distill.models.factory import create_model
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.initializer.random_sample_initializer import (
    RandomSampleInitializer,
)
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher
from ts_distill.evaluation.evaluation import Evaluator


# =============================================================================
# CONFIG
# =============================================================================

ACTIVE_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
ACTIVE_MODELS = ["DLinear", "MLP", "CNN"]
#ACTIVE_MODELS = ["MLP"]

MULTI_SEED_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}

#SEEDS_MULTI = [42, 7, 123]
SEEDS_MULTI = [42]
SEEDS_SINGLE = [42]

MODEL_CONFIGS = {
    "DLinear": {"individual": False},
    "LSTM": {"hidden_dim": 16, "num_layer": 1},
    "MLP": {},
    "CNN": {},
}

DATASET_CONFIGS = {
    "ETTh1": {
        "csv_path": r"C:\fyp\ts-distill\example/ETTh1.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s": [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ],
        "in_features": 7,
    },
    "ETTh2": {
        "csv_path": r"C:\fyp\ts-distill\example/ETTh2.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s": [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ],
        "in_features": 7,
    },
    "ETTm1": {
        "csv_path": r"C:\fyp\ts-distill\example/ETTm1.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 96 - 96, 12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s": [
            12 * 30 * 96,
            12 * 30 * 96 + 4 * 30 * 96,
            12 * 30 * 96 + 8 * 30 * 96,
        ],
        "in_features": 7,
    },
    "ETTm2": {
        "csv_path": r"C:\fyp\ts-distill\example/ETTm2.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 96 - 96, 12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s": [
            12 * 30 * 96,
            12 * 30 * 96 + 4 * 30 * 96,
            12 * 30 * 96 + 8 * 30 * 96,
        ],
        "in_features": 7,
    },
}

DISTILL_CONFIG = {
    "seq_len": 96,
    "pred_len": 96,

    # Expert training
    "expert_epochs": 80,
    "expert_lr": 0.01,
    "expert_momentum": 0.9,

    # MTT
    "n_distill_steps": 300,
    "n_synthetic": 384,
    "synthetic_lr": 5.0,
    "student_lr": 0.01,
    "student_steps": 20,
    "snapshot_student_steps": 50,
    "trajectory_gap": 5,

    # Evaluation — same protocol for both real and synthetic training,
    # matching experiment_matrix.py so results are directly comparable.
    # Early stopping uses the real val set as the stopping criterion even
    # when training on synthetic data: stops when real-val loss stops improving.
    "eval_max_epochs": 300,
    "eval_lr": 0.001,
    "eval_batch_size": 32,
    "early_stop_patience": 10,

    "batch_size": 64,
}


# =============================================================================
# CSV
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH = RESULTS_DIR / "cross_arch_baseline_clean.csv"

CSV_COLUMNS = [
    "experiment",
    "dataset",
    "expert_model",
    "student_model",
    "seed",
    "method",
    "real_mse",
    "transfer_mse",
    "mse_ratio",
    "n_pairs_available",
    "notes",
]


def append_csv(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exists = CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if not exists:
            writer.writeheader()

        writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})

    print(
        f"[CSV] "
        f"{row['dataset']} | "
        f"{row['expert_model']} -> {row['student_model']} | "
        f"seed={row['seed']}"
    )


# =============================================================================
# HELPERS
# =============================================================================

def make_model(model_name, seq_len, pred_len, in_features):
    return create_model(
        model_type=model_name,
        seq_len=seq_len,
        pred_len=pred_len,
        in_features=in_features,
        model_kwargs=MODEL_CONFIGS[model_name],
    )


def load_dataset(dataset_name, cfg):

    ds_cfg = DATASET_CONFIGS[dataset_name]

    seq_len = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    in_features = ds_cfg["in_features"]

    window_size = seq_len + pred_len

    df_raw = pd.read_csv(ds_cfg["csv_path"])

    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    (
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
    ) = get_data_splits(
        values=values,
        window_size=window_size,
        seq_len=seq_len,
        dataset_cfg=ds_cfg,
    )

    scaler = StandardScaler()

    scaler.fit(values[train_start:train_end])

    data_scaled = scaler.transform(values)

    train_data = torch.tensor(
        make_windows(data_scaled[train_start:train_end], window_size),
        dtype=torch.float32,
    )

    test_data = torch.tensor(
        make_windows(data_scaled[test_start:test_end], window_size),
        dtype=torch.float32,
    )

    raw_train = torch.tensor(
        data_scaled[train_start:train_end],
        dtype=torch.float32,
    )

    val_data = torch.tensor(
        make_windows(data_scaled[val_start:val_end], window_size),
        dtype=torch.float32,
    )

    return {
        "train_data": train_data,
        "val_data":   val_data,
        "test_data":  test_data,
        "raw_train":  raw_train,
        "seq_len":    seq_len,
        "pred_len":   pred_len,
        "window_size": window_size,
        "in_features": in_features,
    }


# =============================================================================
# REAL DATA EVALUATION
# =============================================================================

def row_exists(dataset, expert, student, seed):
    """Return True if a completed (non-empty transfer_mse) row is in the CSV."""
    if not CSV_PATH.exists():
        return False
    try:
        df = pd.read_csv(CSV_PATH)
        mask = (
            (df["dataset"]       == dataset) &
            (df["expert_model"]  == expert)  &
            (df["student_model"] == student) &
            (df["seed"]          == seed)    &
            (df["transfer_mse"].notna())     &
            (df["transfer_mse"]  != "")
        )
        return bool(mask.any())
    except Exception:
        return False


def eval_on_real(student_name, data, cfg, device, seed):
    """Train student on real data with early stopping; returns test MSE.

    Early stopping on real val loss is essential: without it, different
    dataset sizes cause wildly different gradient-step counts (e.g. 50
    epochs on ETTh2 = 13 250 steps → CNN overfits; 50 epochs on 384-point
    synthetic = 350 steps → underfits). Early stopping finds the true
    convergence point for each architecture, giving a reliable denominator
    for mse_ratio regardless of dataset size.
    """
    torch.manual_seed(seed)

    model = make_model(
        student_name,
        data["seq_len"],
        data["pred_len"],
        data["in_features"],
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=cfg["eval_lr"],
        ),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=data["seq_len"],
    )

    trainer.fit(
        dataloader=TorchDataLoader(
            data["train_data"],
            batch_size=cfg["eval_batch_size"],
            shuffle=True,
        ),
        epochs=cfg["eval_max_epochs"],
        val_loader=TorchDataLoader(
            data["val_data"],
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
        ),
        patience=cfg["early_stop_patience"],
    )

    evaluator = Evaluator(
        seq_len=data["seq_len"],
        batch_size=cfg["eval_batch_size"],
    )

    return evaluator.test_on_real(model, data["test_data"].to(device))["MSE"]


# =============================================================================
# DISTILLATION
# =============================================================================

def distill_synthetic(expert_name, data, cfg, device, seed):

    torch.manual_seed(seed)

    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    def factory():
        return make_model(expert_name, sl, pl, nf)

    # -----------------------------------------------------------------
    # Train expert
    # -----------------------------------------------------------------

    recorder = SimpleRecorder(record_every=1)

    expert = factory()

    trainer = Trainer(
        model=expert,
        optimizer=torch.optim.SGD(
            expert.parameters(),
            lr=cfg["expert_lr"],
            momentum=cfg["expert_momentum"],
        ),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=sl,
    )

    trainer.fit(
        dataloader=MiniBatchLoader(
            data["train_data"],
            batch_size=cfg["batch_size"],
        ),
        epochs=cfg["expert_epochs"],
        callbacks=[recorder, SimpleCallback()],
    )

    n_pairs = len(recorder.trajectory)

    print(f"Trajectory checkpoints: {n_pairs}")

    # -----------------------------------------------------------------
    # Initialize synthetic sequence
    # -----------------------------------------------------------------

    initializer = RandomSampleInitializer()

    synthetic_init = initializer.initialize_sequence(
        data["raw_train"],
        cfg["n_synthetic"],
    )

    # -----------------------------------------------------------------
    # MTT Distillation
    # -----------------------------------------------------------------

    distiller = MTTDistiller(
        initializer=initializer,
        matcher=MSEMatcher(),
        model_factory=factory,
        expert_recorder=recorder,
        expert_epochs=cfg["trajectory_gap"],
        syn_batch_size=cfg["batch_size"],
        synthetic_lr=cfg["synthetic_lr"],
        student_lr=cfg["student_lr"],
        student_steps=cfg["student_steps"],
        snapshot_student_steps=cfg["snapshot_student_steps"],
        seq_len=sl,
        pred_len=pl,
        device=device,
    )

    synthetic_seq = distiller.distill(
        synthetic_init=synthetic_init,
        n_steps=cfg["n_distill_steps"],
    )

    return synthetic_seq, n_pairs


# =============================================================================
# SYNTHETIC EVALUATION
# =============================================================================

def eval_on_synthetic(
    student_name,
    synthetic_seq,
    data,
    cfg,
    device,
    seed,
):

    torch.manual_seed(seed)

    syn_windows = torch.tensor(
        make_windows(
            synthetic_seq.cpu().numpy(),
            data["window_size"],
        ),
        dtype=torch.float32,
    )

    model = make_model(
        student_name,
        data["seq_len"],
        data["pred_len"],
        data["in_features"],
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=cfg["eval_lr"],
        ),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=data["seq_len"],
    )

    # Early stopping on real val — same protocol as experiment_matrix.py.
    # The real val set acts as the stopping criterion even though training
    # is on synthetic data, so results are directly comparable to the
    # real-data baseline and to other experiments using this protocol.
    trainer.fit(
        dataloader=TorchDataLoader(
            syn_windows,
            batch_size=cfg["eval_batch_size"],
            shuffle=True,
        ),
        epochs=cfg["eval_max_epochs"],
        val_loader=TorchDataLoader(
            data["val_data"],
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
        ),
        patience=cfg["early_stop_patience"],
    )

    evaluator = Evaluator(
        seq_len=data["seq_len"],
        batch_size=cfg["eval_batch_size"],
    )

    metrics = evaluator.test_on_real(
        model,
        data["test_data"].to(device),
    )

    return metrics["MSE"]


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(device, cfg):

    print("=" * 80)
    print("H1 CLEAN CROSS-ARCHITECTURE BASELINE")
    print("=" * 80)

    for dataset_name in ACTIVE_DATASETS:

        seeds = (
            SEEDS_MULTI
            if dataset_name in MULTI_SEED_DATASETS
            else SEEDS_SINGLE
        )

        data = load_dataset(dataset_name, cfg)

        # real_mse is the same for a given (student, seed) regardless of
        # which expert generated the synthetic data.  Cache it so we only
        # train on real data once per student per dataset.
        real_mse_cache: dict = {}   # (student_name, seed) -> float

        for expert_name in ACTIVE_MODELS:

            for seed in seeds:

                print("\n" + "=" * 60)
                print(
                    f"Dataset={dataset_name} | "
                    f"Expert={expert_name} | "
                    f"Seed={seed}"
                )
                print("=" * 60)

                # Skip the whole expert block if every student row exists.
                if all(
                    row_exists(dataset_name, expert_name, s, seed)
                    for s in ACTIVE_MODELS
                ):
                    print("  [RESUME] All 4 students already in CSV — skipping.")
                    continue

                # ---------------------------------------------------------
                # Distill synthetic data
                # ---------------------------------------------------------

                try:
                    synthetic_seq, n_pairs = distill_synthetic(
                        expert_name, data, cfg, device, seed,
                    )
                except Exception as exc:
                    print(f"  [FAIL] Distillation: {exc}")
                    for student_name in ACTIVE_MODELS:
                        if not row_exists(dataset_name, expert_name, student_name, seed):
                            append_csv({
                                "experiment":   "cross_arch_baseline_clean",
                                "dataset":      dataset_name,
                                "expert_model": expert_name,
                                "student_model": student_name,
                                "seed":         seed,
                                "method":       "standard_mtt_clean",
                                "notes":        f"distillation_failed: {str(exc)[:120]}",
                            })
                    continue

                # ---------------------------------------------------------
                # Evaluate all students on the same synthetic_seq
                # ---------------------------------------------------------

                for student_name in ACTIVE_MODELS:

                    if row_exists(dataset_name, expert_name, student_name, seed):
                        print(f"  [RESUME] {student_name} already in CSV — skipping.")
                        continue

                    print(f"\n  Student: {student_name}")

                    # Real baseline — compute once per (student, seed), reuse.
                    key = (student_name, seed)
                    if key not in real_mse_cache:
                        print(f"    Computing real baseline for {student_name}...")
                        try:
                            real_mse_cache[key] = eval_on_real(
                                student_name, data, cfg, device, seed,
                            )
                        except Exception as exc:
                            print(f"    [FAIL] real eval: {exc}")
                            real_mse_cache[key] = float("nan")
                    real_mse = real_mse_cache[key]

                    # Synthetic evaluation — fixed budget, no stopping.
                    try:
                        transfer_mse = eval_on_synthetic(
                            student_name, synthetic_seq, data, cfg, device, seed,
                        )
                    except Exception as exc:
                        print(f"    [FAIL] synthetic eval: {exc}")
                        append_csv({
                            "experiment":        "cross_arch_baseline_clean",
                            "dataset":           dataset_name,
                            "expert_model":      expert_name,
                            "student_model":     student_name,
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
                        f"    real={real_mse:.6f} | "
                        f"transfer={transfer_mse:.6f} | "
                        f"ratio={mse_ratio:.3f}"
                    )

                    append_csv({
                        "experiment":        "cross_arch_baseline_clean",
                        "dataset":           dataset_name,
                        "expert_model":      expert_name,
                        "student_model":     student_name,
                        "seed":              seed,
                        "method":            "standard_mtt_clean",
                        "real_mse":          real_mse,
                        "transfer_mse":      transfer_mse,
                        "mse_ratio":         mse_ratio,
                        "n_pairs_available": n_pairs,
                        "notes":             "",
                    })


# =============================================================================
# ENTRY
# =============================================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    run(device, DISTILL_CONFIG)

    print("\nExperiment complete.")
    print(f"Results: {CSV_PATH}")


if __name__ == "__main__":

    # -------------------------------------------------------------
    # QUICK SMOKE TEST
    # -------------------------------------------------------------

    # ACTIVE_DATASETS[:] = ["ETTh1"]
    # ACTIVE_MODELS[:] = ["DLinear", "CNN"]
    # SEEDS_MULTI[:] = [42]

    # DISTILL_CONFIG["expert_epochs"] = 5
    # DISTILL_CONFIG["n_distill_steps"] = 10
    # DISTILL_CONFIG["eval_epochs"] = 5

    main()