"""
Phase-Aware MTT vs Standard MTT — Matching Objective Comparison
===============================================================

Context
-------
H1 (cross-architecture bias) was NOT confirmed.  The H1 redirect is:

  "Test whether matching objective type affects distillation quality
   independently of cross-architecture use."

This experiment compares two MTT variants on same-architecture evaluation:

  param_mtt       — Standard MTT (all parameter matching throughout)
  phase_aware_mtt — Hybrid: parameter matching in early phase,
                    prediction matching in late phase

The phase boundary T+ is loaded from the output of h_detect_phase_boundary.py
(results/phase_boundaries.csv).  Run that script first to auto-detect T+
per (dataset, model) using the gradient magnitude / parameter distance method.
If no boundary CSV is found, the experiment falls back to None (param matching
only) and logs a warning.

Why this split makes sense
--------------------------
Early phase: expert is rapidly traversing parameter space.  Parameter
  matching gives a strong, direction-specific signal that drives the
  synthetic data toward capturing the real-data training dynamics.

Late phase: expert weights are fine-tuning with small, architecture-
  specific adjustments.  Matching those micro-adjustments in parameter
  space encodes architecture-specific noise.  Switching to prediction
  matching anchors the loss to what the model *predicts*, which is
  architecture-agnostic and should produce synthetic data that
  generalises better.

Both variants share the same expert trajectory, same inner loop, and the
same evaluation protocol — the only variable is the outer-loop loss function.

Metric
------
  mse_ratio = transfer_mse / real_mse
  Lower is better.  ~1 means synthetic nearly as good as real data.

Output
------
  results/pred_vs_param_matching.csv

Run
---
  python -m example.experiments.h_pred_vs_param_matching
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
#ACTIVE_MODELS   = ["DLinear", "LSTM", "MLP", "CNN"]
ACTIVE_MODELS   = ["DLinear", "MLP", "CNN"]

SEEDS = [42]

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

# Which phase-boundary detection method to use:
#   "paramdist" — Method A (parameter-distance plateau, h_detect_phase_boundary.py)
#   "valloss"   — Method B (validation-loss plateau, h_detect_phase_boundary_valloss.py)
BOUNDARY_METHOD = "valloss"

DISTILL_CONFIG = {
    "seq_len":  96,
    "pred_len": 96,

    # Expert training
    "expert_epochs":   80,
    "expert_lr":       0.01,
    "expert_momentum": 0.9,

    # MTT (shared by both variants)
    "n_distill_steps":        300,
    "n_synthetic":            384,
    "synthetic_lr":           5.0,
    "student_lr":             0.01,
    "student_steps":          20,
    "snapshot_student_steps": 50,
    "trajectory_gap":         5,
    "batch_size":             64,

    # Evaluation — same protocol for both real and synthetic training,
    # matching experiment_matrix.py so results are directly comparable.
    # Early stopping uses the real val set as the stopping criterion even
    # when training on synthetic data: stops when real-val loss stops improving.
    "eval_max_epochs":     300,
    "eval_lr":             0.001,
    "eval_batch_size":     32,
    "early_stop_patience": 10,
}


# =============================================================================
# CSV
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"

if BOUNDARY_METHOD == "valloss":
    CSV_PATH     = RESULTS_DIR / "pred_vs_param_matching_valloss.csv"
    EXPERIMENT_NAME = "phase_aware_vs_param_valloss"
else:
    CSV_PATH     = RESULTS_DIR / "pred_vs_param_matching.csv"
    EXPERIMENT_NAME = "phase_aware_vs_param"

CSV_COLUMNS = [
    "experiment",
    "dataset",
    "model",
    "seed",
    "method",            # param_mtt | phase_aware_mtt
    "phase_boundary",    # boundary used (empty for param_mtt and DLinear)
    "real_mse",
    "transfer_mse",
    "mse_ratio",
    "n_pairs",
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
        f"[CSV] {row['dataset']} | {row['model']} | "
        f"seed={row['seed']} | method={row['method']} | "
        f"ratio={row.get('mse_ratio', '?')}"
    )


def row_exists(dataset, model, seed, method):
    if not CSV_PATH.exists():
        return False
    try:
        df   = pd.read_csv(CSV_PATH)
        mask = (
            (df["dataset"] == dataset) &
            (df["model"]   == model)   &
            (df["seed"]    == seed)    &
            (df["method"]  == method)  &
            (df["transfer_mse"].notna()) &
            (df["transfer_mse"] != "")
        )
        return bool(mask.any())
    except Exception:
        return False


# =============================================================================
# PHASE BOUNDARY LOADING
# =============================================================================

if BOUNDARY_METHOD == "valloss":
    BOUNDARY_CSV = RESULTS_DIR / "phase_boundaries_valloss.csv"
    BOUNDARY_SCRIPT = "h_detect_phase_boundary_valloss.py"
else:
    BOUNDARY_CSV = RESULTS_DIR / "phase_boundaries.csv"
    BOUNDARY_SCRIPT = "h_detect_phase_boundary.py"


def load_phase_boundaries(seed: int = 42) -> dict:
    """
    Load detected T+ values from the phase-boundary detector output
    (h_detect_phase_boundary.py for "paramdist", or
    h_detect_phase_boundary_valloss.py for "valloss" — see BOUNDARY_METHOD).

    Returns a dict: {(dataset, model): boundary_epoch_or_None}

    Falls back to None for any missing entry (model treated as having no
    boundary, so phase_aware_mtt behaves identically to param_mtt).
    """
    boundaries = {}

    if not BOUNDARY_CSV.exists():
        print(
            f"[WARN] {BOUNDARY_CSV} not found. "
            f"Run {BOUNDARY_SCRIPT} first to auto-detect T+.\n"
            f"       phase_aware_mtt will be skipped for all models."
        )
        return boundaries

    try:
        df = pd.read_csv(BOUNDARY_CSV)
        df = df[df["seed"] == seed]
        for _, row in df.iterrows():
            val = row.get("boundary_epoch", None)
            if pd.isna(val) or val == "":
                val = None
            else:
                val = int(val)
            boundaries[(row["dataset"], row["model"])] = val
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
    return recorder, n_pairs


# =============================================================================
# DISTILLATION
# =============================================================================

def distill_param_mtt(model_name, recorder, data, cfg, device, seed):
    """Standard MTT — parameter matching throughout."""
    torch.manual_seed(seed)

    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    def factory():
        return make_model(model_name, sl, pl, nf)

    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(data["raw_train"], cfg["n_synthetic"])

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

    return distiller.distill(synthetic_init=synthetic_init, n_steps=cfg["n_distill_steps"])


def distill_phase_aware_mtt(model_name, recorder, data, cfg, device, seed, phase_boundary):
    """Hybrid MTT — parameter matching early, prediction matching late."""
    torch.manual_seed(seed)

    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    def factory():
        return make_model(model_name, sl, pl, nf)

    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(data["raw_train"], cfg["n_synthetic"])

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
        device=device,
        phase_boundary=phase_boundary,
    )

    return distiller.distill(synthetic_init=synthetic_init, n_steps=cfg["n_distill_steps"])


# =============================================================================
# EVALUATION  (same protocol as H1 cross-arch baseline)
# =============================================================================

def eval_on_real(model_name, data, cfg, device, seed):
    """Train student on real data with early stopping; return test MSE."""
    torch.manual_seed(seed)

    model = make_model(
        model_name, data["seq_len"], data["pred_len"], data["in_features"]
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


def eval_on_synthetic(model_name, synthetic_seq, data, cfg, device, seed):
    """Train student on synthetic windows with early stopping on real val; return test MSE."""
    torch.manual_seed(seed)

    syn_windows = torch.tensor(
        make_windows(synthetic_seq.cpu().numpy(), data["window_size"]),
        dtype=torch.float32,
    )

    model = make_model(
        model_name, data["seq_len"], data["pred_len"], data["in_features"]
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=data["seq_len"],
    )

    # Early stopping on real val — same protocol as experiment_matrix.py.
    # The real val set acts as the stopping criterion even though training
    # is on synthetic data, so results are directly comparable to the baseline.
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
    print("PHASE-AWARE MTT vs STANDARD MTT — MATCHING OBJECTIVE COMPARISON")
    print("=" * 80)
    print("Same-arch evaluation (expert arch == student arch)")
    print("param_mtt       : parameter matching throughout (baseline)")
    print("phase_aware_mtt : parameter matching early, prediction matching late")
    print()

    # Load auto-detected boundaries from h_detect_phase_boundary.py output.
    # Key: (dataset, model) -> int or None
    phase_boundaries = load_phase_boundaries(seed=SEEDS[0])

    for dataset_name in ACTIVE_DATASETS:

        print("\n" + "=" * 60)
        print(f"  Dataset: {dataset_name}")
        print("=" * 60)

        data = load_dataset(dataset_name, cfg)

        real_mse_cache: dict = {}   # (model_name, seed) -> float

        for model_name in ACTIVE_MODELS:

            for seed in SEEDS:

                boundary = phase_boundaries.get((dataset_name, model_name))

                print(f"\n--- Model={model_name} | Seed={seed} | "
                      f"boundary={boundary} ---")

                param_done = row_exists(dataset_name, model_name, seed, "param_mtt")
                # phase_aware is same as param_mtt when boundary is None
                phase_done = (
                    boundary is None
                    or row_exists(dataset_name, model_name, seed, "phase_aware_mtt")
                )

                if param_done and phase_done:
                    print("  [RESUME] All methods already in CSV — skipping.")
                    continue

                # ---------------------------------------------------------
                # Expert training (shared trajectory for both methods)
                # ---------------------------------------------------------
                try:
                    recorder, n_pairs = train_expert(
                        model_name, data, cfg, device, seed
                    )
                except Exception as exc:
                    print(f"  [FAIL] Expert training: {exc}")
                    for method in ["param_mtt", "phase_aware_mtt"]:
                        if method == "phase_aware_mtt" and boundary is None:
                            continue
                        append_csv({
                            "experiment":    EXPERIMENT_NAME,
                            "dataset":       dataset_name,
                            "model":         model_name,
                            "seed":          seed,
                            "method":        method,
                            "phase_boundary": boundary or "",
                            "notes":         f"expert_training_failed: {str(exc)[:120]}",
                        })
                    continue

                # ---------------------------------------------------------
                # Real-data baseline (cached per model+seed)
                # ---------------------------------------------------------
                key = (model_name, seed)
                if key not in real_mse_cache:
                    print(f"  Computing real baseline for {model_name}...")
                    try:
                        real_mse_cache[key] = eval_on_real(
                            model_name, data, cfg, device, seed
                        )
                        print(f"  real_mse = {real_mse_cache[key]:.6f}")
                    except Exception as exc:
                        print(f"  [FAIL] real eval: {exc}")
                        real_mse_cache[key] = float("nan")
                real_mse = real_mse_cache[key]

                # ---------------------------------------------------------
                # param_mtt  (standard MTT — baseline)
                # ---------------------------------------------------------
                if not param_done:
                    print(f"\n  [param_mtt] Distilling...")
                    try:
                        syn_param    = distill_param_mtt(
                            model_name, recorder, data, cfg, device, seed
                        )
                        transfer_param = eval_on_synthetic(
                            model_name, syn_param, data, cfg, device, seed
                        )
                        ratio_param = (
                            transfer_param / real_mse
                            if (real_mse > 0 and not np.isnan(real_mse))
                            else float("nan")
                        )
                        print(
                            f"  [param_mtt] real={real_mse:.6f} | "
                            f"transfer={transfer_param:.6f} | ratio={ratio_param:.3f}"
                        )
                        append_csv({
                            "experiment":    EXPERIMENT_NAME,
                            "dataset":       dataset_name,
                            "model":         model_name,
                            "seed":          seed,
                            "method":        "param_mtt",
                            "phase_boundary": "",
                            "real_mse":      real_mse,
                            "transfer_mse":  transfer_param,
                            "mse_ratio":     ratio_param,
                            "n_pairs":       n_pairs,
                            "notes":         "",
                        })
                    except Exception as exc:
                        print(f"  [FAIL] param_mtt: {exc}")
                        append_csv({
                            "experiment":    EXPERIMENT_NAME,
                            "dataset":       dataset_name,
                            "model":         model_name,
                            "seed":          seed,
                            "method":        "param_mtt",
                            "phase_boundary": "",
                            "real_mse":      real_mse,
                            "n_pairs":       n_pairs,
                            "notes":         f"failed: {str(exc)[:120]}",
                        })

                # ---------------------------------------------------------
                # phase_aware_mtt  (param early + pred late)
                # ---------------------------------------------------------
                if boundary is None:
                    print(
                        f"  [phase_aware_mtt] No boundary for {model_name} — "
                        f"skipping (identical to param_mtt)."
                    )
                elif not phase_done:
                    print(
                        f"\n  [phase_aware_mtt] Distilling "
                        f"(boundary={boundary})..."
                    )
                    try:
                        syn_phase    = distill_phase_aware_mtt(
                            model_name, recorder, data, cfg, device, seed, boundary
                        )
                        transfer_phase = eval_on_synthetic(
                            model_name, syn_phase, data, cfg, device, seed
                        )
                        ratio_phase = (
                            transfer_phase / real_mse
                            if (real_mse > 0 and not np.isnan(real_mse))
                            else float("nan")
                        )
                        print(
                            f"  [phase_aware_mtt] real={real_mse:.6f} | "
                            f"transfer={transfer_phase:.6f} | "
                            f"ratio={ratio_phase:.3f}"
                        )
                        append_csv({
                            "experiment":    EXPERIMENT_NAME,
                            "dataset":       dataset_name,
                            "model":         model_name,
                            "seed":          seed,
                            "method":        "phase_aware_mtt",
                            "phase_boundary": boundary,
                            "real_mse":      real_mse,
                            "transfer_mse":  transfer_phase,
                            "mse_ratio":     ratio_phase,
                            "n_pairs":       n_pairs,
                            "notes":         "",
                        })
                    except Exception as exc:
                        print(f"  [FAIL] phase_aware_mtt: {exc}")
                        append_csv({
                            "experiment":    EXPERIMENT_NAME,
                            "dataset":       dataset_name,
                            "model":         model_name,
                            "seed":          seed,
                            "method":        "phase_aware_mtt",
                            "phase_boundary": boundary,
                            "real_mse":      real_mse,
                            "n_pairs":       n_pairs,
                            "notes":         f"failed: {str(exc)[:120]}",
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
    # ACTIVE_MODELS[:] = ["LSTM"]
    # SEEDS[:] = [42]
    # DISTILL_CONFIG["expert_epochs"]    = 5
    # DISTILL_CONFIG["n_distill_steps"]  = 10
    # DISTILL_CONFIG["eval_syn_epochs"]  = 5
    # DISTILL_CONFIG["eval_max_epochs"]  = 10
    # To test with a small boundary: edit phase_boundaries.csv and set boundary_epoch=3

    main()
