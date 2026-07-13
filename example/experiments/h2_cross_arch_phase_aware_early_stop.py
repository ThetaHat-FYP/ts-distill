"""
H2 Cross-Architecture Baseline — Phase-Aware vs Standard MTT (Early-Stopping Evaluation)
==========================================================================================

Variant of h2_cross_arch_phase_aware_fixed_budget.py with early stopping
during evaluation: both the real-data model and the synthetic-data model are
trained with a val_loader and patience-based stopping (up to eval_max_epochs).

Everything else (expert training, phase-boundary loading, param_mtt vs
phase_aware_mtt distillation, cross-architecture student matrix, RNG-state
sharing between methods, all hyperparameters) is identical to
h2_cross_arch_phase_aware_fixed_budget.py.

  param_mtt       — Standard MTT (parameter matching throughout)
  phase_aware_mtt — Hybrid: parameter matching in the early phase of the
                    EXPERT's trajectory, prediction matching in the late
                    phase (boundary T+ taken from
                    results/phase_boundaries.csv, validation-loss-velocity
                    detector — see BOUNDARY_METHOD below)

Both methods share the same expert trajectory (trained once per
(dataset, expert, seed)) and the same evaluation protocol — the only
variable is the outer-loop matching loss used during distillation.

Evaluation protocol (early stopping):
  - real and synthetic training both use val_loader + patience on the real
    validation set, stopping early when val MSE stops improving.

Metric
------
  mse_ratio = transfer_mse / real_mse
  Lower is better. ~1 means synthetic nearly as good as real data.

Output
------
  results/cross_arch_phase_aware_early_stop.csv

Run
---
  python -m example.experiments.h2_cross_arch_phase_aware_early_stop
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent.parent))

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader
from ts_distill.models.factory import create_model
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.distillation_algorithm.phase_aware_mtt import (
    PhaseAwareMTTDistiller,
)
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
ACTIVE_MODELS   = ["DLinear","MLP", "CNN"]

MULTI_SEED_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}
#MULTI_SEED_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}
SEEDS_MULTI = [7, 42, 123]
SEEDS_SINGLE = [7, 42, 123]

MODEL_CONFIGS = {
    "DLinear": {"individual": False},
    "LSTM":    {"hidden_dim": 16, "num_layer": 1},
    "MLP":     {},
    "CNN":     {},
}

DATASET_CONFIGS = {
    "ETTh1": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTh1.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12 * 30 * 24 - 96,  12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s":   [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        "in_features": 7,
    },
    "ETTh2": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTh2.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12 * 30 * 24 - 96,  12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s":   [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        "in_features": 7,
    },
    "ETTm1": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTm1.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12 * 30 * 96 - 96,  12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s":   [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        "in_features": 7,
    },
    "ETTm2": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTm2.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12 * 30 * 96 - 96,  12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s":   [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        "in_features": 7,
    },
}

# Phase-boundary detection method:
#   "velocity" — results/phase_boundaries.csv         (h_detect_phase_boundary.py,
#                 validation-loss-velocity threshold detector)
#   "valloss"  — results/phase_boundaries_valloss.csv (h_detect_phase_boundary_valloss.py,
#                 validation-loss plateau detector)
BOUNDARY_METHOD = "velocity"

DISTILL_CONFIG = {
    "seq_len":  96,
    "pred_len": 96,

    # Expert training
    "expert_epochs":   80,
    "expert_lr":       0.01,
    "expert_momentum": 0.9,

    # MTT (shared by both methods)
    "n_distill_steps":        1000,
    "n_synthetic":            384,
    "synthetic_lr":           5.0,
    "student_lr":             0.1,
    "student_steps":          20,
    "snapshot_student_steps": 50,
    "trajectory_gap":         5,
    "batch_size":             64,

    # Evaluation — early stopping on real val set. Both real and synthetic
    # training run for up to eval_max_epochs epochs, stopping early when
    # val MSE does not improve for early_stop_patience consecutive epochs.
    "eval_max_epochs":     300,
    "eval_lr":             0.001,
    "eval_batch_size":     32,
    "early_stop_patience": 10,
}


# =============================================================================
# CSV
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "cross_arch_phase_aware_early_stop_stu_Lr_0.1.csv"

CSV_COLUMNS = [
    "experiment",
    "dataset",
    "expert_model",
    "student_model",
    "seed",
    "method",          # param_mtt | phase_aware_mtt
    "phase_boundary",  # boundary used for the expert (empty for param_mtt)
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
        f"[CSV] {row['dataset']} | "
        f"{row['expert_model']} -> {row['student_model']} | "
        f"seed={row['seed']} | method={row['method']} | "
        f"ratio={row.get('mse_ratio', '?')}"
    )


def row_exists(dataset, expert, student, seed, method):
    if not CSV_PATH.exists():
        return False
    try:
        df = pd.read_csv(CSV_PATH)
        mask = (
            (df["dataset"]       == dataset) &
            (df["expert_model"]  == expert)  &
            (df["student_model"] == student) &
            (df["seed"]          == seed)    &
            (df["method"]        == method)  &
            (df["transfer_mse"].notna())     &
            (df["transfer_mse"]  != "")
        )
        return bool(mask.any())
    except Exception:
        return False


# =============================================================================
# PHASE BOUNDARY LOADING
# =============================================================================

if BOUNDARY_METHOD == "valloss":
    BOUNDARY_CSV    = RESULTS_DIR / "phase_boundaries_valloss.csv"
    BOUNDARY_SCRIPT = "h_detect_phase_boundary_valloss.py"
else:
    BOUNDARY_CSV    = RESULTS_DIR / "phase_boundaries.csv"
    BOUNDARY_SCRIPT = "h_detect_phase_boundary.py"
    # NOTE: this CSV is produced by the validation-loss-velocity detector,
    # not parameter distance — see h_detect_phase_boundary.py.


def load_phase_boundaries() -> dict:
    """Load detected T+ values: {(dataset, model, seed): boundary_epoch_or_None}."""
    boundaries = {}

    if not BOUNDARY_CSV.exists():
        print(
            f"[WARN] {BOUNDARY_CSV} not found. "
            f"Run {BOUNDARY_SCRIPT} first to auto-detect T+.\n"
            f"       phase_aware_mtt will be skipped for all experts."
        )
        return boundaries

    try:
        df = pd.read_csv(BOUNDARY_CSV)
        for _, row in df.iterrows():
            val = row.get("boundary_epoch", None)
            if pd.isna(val) or val == "":
                val = None
            else:
                val = int(val)
            boundaries[(row["dataset"], row["model"], int(row["seed"]))] = val
    except Exception as exc:
        print(f"[WARN] Could not read {BOUNDARY_CSV}: {exc}")

    return boundaries


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
    ds_cfg      = DATASET_CONFIGS[dataset_name]
    seq_len     = cfg["seq_len"]
    pred_len    = cfg["pred_len"]
    in_features = ds_cfg["in_features"]
    window_size = seq_len + pred_len

    df_raw = pd.read_csv(ds_cfg["csv_path"])
    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values=values,
        window_size=window_size,
        seq_len=seq_len,
        dataset_cfg=ds_cfg,
    )

    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data_scaled = scaler.transform(values)

    train_data = torch.tensor(
        make_windows(data_scaled[train_start:train_end], window_size), dtype=torch.float32
    )
    val_data = torch.tensor(
        make_windows(data_scaled[val_start:val_end], window_size), dtype=torch.float32
    )
    test_data = torch.tensor(
        make_windows(data_scaled[test_start:test_end], window_size), dtype=torch.float32
    )
    raw_train = torch.tensor(data_scaled[train_start:train_end], dtype=torch.float32)

    return {
        "train_data":  train_data,
        "val_data":    val_data,
        "test_data":   test_data,
        "raw_train":   raw_train,
        "seq_len":     seq_len,
        "pred_len":    pred_len,
        "window_size": window_size,
        "in_features": in_features,
    }


# =============================================================================
# EXPERT TRAINING
# =============================================================================

def train_expert(model_name, data, cfg, device, seed):
    """Train expert on real data and record the full trajectory."""
    torch.manual_seed(seed)

    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    expert   = make_model(model_name, sl, pl, nf).to(device)
    recorder = SimpleRecorder(record_every=1)

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
        dataloader=MiniBatchLoader(data["train_data"], batch_size=cfg["batch_size"]),
        epochs=cfg["expert_epochs"],
        callbacks=[recorder, SimpleCallback()],
    )

    n_pairs = len(recorder.trajectory)
    print(f"  Expert trained. Trajectory checkpoints: {n_pairs}")

    # RNG state right after expert training — callers restore this state
    # before each distillation call so synthetic_init is identical between
    # param_mtt and phase_aware_mtt (isolating the matching loss as the only
    # difference between the two methods).
    rng_state = torch.get_rng_state()
    return recorder, n_pairs, rng_state


# =============================================================================
# DISTILLATION
# =============================================================================

def distill_param_mtt(model_name, recorder, data, cfg, device, seed):
    """Standard MTT — parameter matching throughout."""
    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    def factory():
        return make_model(model_name, sl, pl, nf)

    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(data["raw_train"], cfg["n_synthetic"])

    # No `device` kwarg — defaults to "cpu", matching experiment_matrix.py's
    # MTTDistiller construction exactly (CPU and CUDA RNG streams diverge even
    # under the same torch.manual_seed, which would otherwise change the
    # distilled synthetic sequence).
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
    )

    # val_data enables best-snapshot selection, matching experiment_matrix.py
    # Step 3 (distiller.distill(..., val_data=val_data)). This is part of the
    # distillation algorithm itself, not the evaluation early stopping below.
    return distiller.distill(
        synthetic_init=synthetic_init,
        n_steps=cfg["n_distill_steps"],
        val_data=data["val_data"],
    )


def distill_phase_aware_mtt(model_name, recorder, data, cfg, device, seed, phase_boundary):
    """Hybrid MTT — parameter matching early, prediction matching late."""
    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    def factory():
        return make_model(model_name, sl, pl, nf)

    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(data["raw_train"], cfg["n_synthetic"])

    # No `device` kwarg — defaults to "cpu", matching the param-matching
    # distiller above and experiment_matrix.py's MTTDistiller construction.
    distiller = PhaseAwareMTTDistiller(
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
        phase_boundary=phase_boundary,
    )

    return distiller.distill(
        synthetic_init=synthetic_init,
        n_steps=cfg["n_distill_steps"],
        val_data=data["val_data"],
    )


# =============================================================================
# EVALUATION (early stopping on real val set)
# =============================================================================

def eval_on_real(student_name, data, cfg, device, seed):
    """Train student on real data with early stopping; return test MSE.

    Uses a fixed seed (0) rather than the experiment seed, matching
    experiment_matrix.py Step 4a.
    """
    torch.manual_seed(0)

    model = make_model(
        student_name, data["seq_len"], data["pred_len"], data["in_features"]
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=data["seq_len"],
    )

    trainer.fit(
        dataloader=TorchDataLoader(
            data["train_data"], batch_size=cfg["eval_batch_size"], shuffle=True
        ),
        epochs=cfg["eval_max_epochs"],
        val_loader=TorchDataLoader(
            data["val_data"], batch_size=cfg["eval_batch_size"], shuffle=False
        ),
        patience=cfg["early_stop_patience"],
    )

    evaluator = Evaluator(seq_len=data["seq_len"], batch_size=cfg["eval_batch_size"])
    return evaluator.test_on_real(model, data["test_data"].to(device))["MSE"]


def eval_on_synthetic(student_name, synthetic_seq, data, cfg, device, seed):
    """Train student on synthetic windows with early stopping on real val;
    return test MSE.

    Uses a fixed seed (1) rather than the experiment seed, matching
    experiment_matrix.py Step 4b.
    """
    torch.manual_seed(1)

    syn_windows = torch.tensor(
        make_windows(synthetic_seq.cpu().numpy(), data["window_size"]),
        dtype=torch.float32,
    )

    model = make_model(
        student_name, data["seq_len"], data["pred_len"], data["in_features"]
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=data["seq_len"],
    )

    trainer.fit(
        dataloader=TorchDataLoader(
            syn_windows, batch_size=cfg["eval_batch_size"], shuffle=True
        ),
        epochs=cfg["eval_max_epochs"],
        val_loader=TorchDataLoader(
            data["val_data"], batch_size=cfg["eval_batch_size"], shuffle=False
        ),
        patience=cfg["early_stop_patience"],
    )

    evaluator = Evaluator(seq_len=data["seq_len"], batch_size=cfg["eval_batch_size"])
    return evaluator.test_on_real(model, data["test_data"].to(device))["MSE"]


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(device, cfg):

    print("=" * 80)
    print("H2 CROSS-ARCHITECTURE BASELINE — PHASE-AWARE vs STANDARD MTT (EARLY STOPPING)")
    print("=" * 80)

    # {(dataset, model, seed): boundary_epoch_or_None}, keyed by EXPERT model
    phase_boundaries = load_phase_boundaries()

    for dataset_name in ACTIVE_DATASETS:

        seeds = (
            SEEDS_MULTI
            if dataset_name in MULTI_SEED_DATASETS
            else SEEDS_SINGLE
        )

        data = load_dataset(dataset_name, cfg)

        # real_mse is the same for a given (student, seed) regardless of
        # which expert/method generated the synthetic data.
        real_mse_cache: dict = {}   # (student_name, seed) -> float

        for expert_name in ACTIVE_MODELS:

            for seed in seeds:

                boundary = phase_boundaries.get((dataset_name, expert_name, seed))

                print("\n" + "=" * 60)
                print(
                    f"Dataset={dataset_name} | Expert={expert_name} | "
                    f"Seed={seed} | boundary={boundary}"
                )
                print("=" * 60)

                # Skip the whole expert block only if every (student, method)
                # combo is already done.
                methods_needed = ["param_mtt"]
                if boundary is not None:
                    methods_needed.append("phase_aware_mtt")

                if all(
                    row_exists(dataset_name, expert_name, s, seed, method)
                    for s in ACTIVE_MODELS
                    for method in methods_needed
                ):
                    print("  [RESUME] All students/methods already in CSV — skipping.")
                    continue

                # ---------------------------------------------------------
                # Train expert (shared trajectory for both methods)
                # ---------------------------------------------------------
                try:
                    recorder, n_pairs, post_expert_rng_state = train_expert(
                        expert_name, data, cfg, device, seed
                    )
                except Exception as exc:
                    print(f"  [FAIL] Expert training: {exc}")
                    for student_name in ACTIVE_MODELS:
                        for method in methods_needed:
                            if not row_exists(dataset_name, expert_name, student_name, seed, method):
                                append_csv({
                                    "experiment":     "cross_arch_phase_aware_early_stop",
                                    "dataset":        dataset_name,
                                    "expert_model":   expert_name,
                                    "student_model":  student_name,
                                    "seed":           seed,
                                    "method":         method,
                                    "phase_boundary": boundary if method == "phase_aware_mtt" else "",
                                    "notes":          f"expert_training_failed: {str(exc)[:120]}",
                                })
                    continue

                # ---------------------------------------------------------
                # Distill synthetic data (param_mtt always; phase_aware_mtt
                # only if a boundary was detected for this expert)
                # ---------------------------------------------------------
                synthetic = {}

                # Restore RNG state to right after expert training so
                # synthetic_init matches between methods for the same
                # expert/seed.
                torch.set_rng_state(post_expert_rng_state)

                try:
                    synthetic["param_mtt"] = distill_param_mtt(
                        expert_name, recorder, data, cfg, device, seed
                    )
                except Exception as exc:
                    print(f"  [FAIL] param_mtt distillation: {exc}")
                    for student_name in ACTIVE_MODELS:
                        if not row_exists(dataset_name, expert_name, student_name, seed, "param_mtt"):
                            append_csv({
                                "experiment":     "cross_arch_phase_aware_early_stop",
                                "dataset":        dataset_name,
                                "expert_model":   expert_name,
                                "student_model":  student_name,
                                "seed":           seed,
                                "method":         "param_mtt",
                                "phase_boundary": "",
                                "n_pairs_available": n_pairs,
                                "notes":          f"distillation_failed: {str(exc)[:120]}",
                            })

                if boundary is None:
                    print(
                        f"  [phase_aware_mtt] No boundary for {expert_name} on "
                        f"{dataset_name} — skipping (identical to param_mtt)."
                    )
                else:
                    # Restore the same starting RNG state as param_mtt so both
                    # methods distill from an identical synthetic_init — the
                    # matching loss is the only difference.
                    torch.set_rng_state(post_expert_rng_state)
                    try:
                        synthetic["phase_aware_mtt"] = distill_phase_aware_mtt(
                            expert_name, recorder, data, cfg, device, seed, boundary
                        )
                    except Exception as exc:
                        print(f"  [FAIL] phase_aware_mtt distillation: {exc}")
                        for student_name in ACTIVE_MODELS:
                            if not row_exists(dataset_name, expert_name, student_name, seed, "phase_aware_mtt"):
                                append_csv({
                                    "experiment":     "cross_arch_phase_aware_early_stop",
                                    "dataset":        dataset_name,
                                    "expert_model":   expert_name,
                                    "student_model":  student_name,
                                    "seed":           seed,
                                    "method":         "phase_aware_mtt",
                                    "phase_boundary": boundary,
                                    "n_pairs_available": n_pairs,
                                    "notes":          f"distillation_failed: {str(exc)[:120]}",
                                })

                # ---------------------------------------------------------
                # Evaluate every student on each available synthetic set
                # ---------------------------------------------------------
                for student_name in ACTIVE_MODELS:

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

                    for method, synthetic_seq in synthetic.items():

                        if row_exists(dataset_name, expert_name, student_name, seed, method):
                            print(f"    [RESUME] {method} already in CSV — skipping.")
                            continue

                        try:
                            transfer_mse = eval_on_synthetic(
                                student_name, synthetic_seq, data, cfg, device, seed,
                            )
                        except Exception as exc:
                            print(f"    [FAIL] {method} synthetic eval: {exc}")
                            append_csv({
                                "experiment":        "cross_arch_phase_aware_early_stop",
                                "dataset":           dataset_name,
                                "expert_model":      expert_name,
                                "student_model":     student_name,
                                "seed":              seed,
                                "method":            method,
                                "phase_boundary":    boundary if method == "phase_aware_mtt" else "",
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
                            f"    [{method}] real={real_mse:.6f} | "
                            f"transfer={transfer_mse:.6f} | ratio={mse_ratio:.3f}"
                        )

                        append_csv({
                            "experiment":        "cross_arch_phase_aware_early_stop",
                            "dataset":           dataset_name,
                            "expert_model":      expert_name,
                            "student_model":     student_name,
                            "seed":              seed,
                            "method":            method,
                            "phase_boundary":    boundary if method == "phase_aware_mtt" else "",
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

    # ----------------------------------------------------------
    # QUICK SMOKE TEST (uncomment to verify the pipeline runs)
    # ----------------------------------------------------------
    # ACTIVE_DATASETS[:] = ["ETTh1"]
    # ACTIVE_MODELS[:] = ["DLinear", "CNN"]
    # SEEDS_MULTI[:] = [42]
    # DISTILL_CONFIG["expert_epochs"]   = 5
    # DISTILL_CONFIG["n_distill_steps"] = 10
    # DISTILL_CONFIG["eval_max_epochs"] = 5

    main()
