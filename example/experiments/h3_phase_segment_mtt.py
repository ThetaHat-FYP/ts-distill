"""
H3 Phase-Segment MTT — Early-Phase vs Late-Phase Synthetic Data
=================================================================

For each (dataset, expert, seed) with a detected phase boundary T+
(results/phase_boundaries.csv, validation-loss-velocity detector,
threshold=0.10 — see h_detect_phase_boundary.py):

  1. Train the expert once and record its full trajectory.
  2. Split the trajectory into two segments:
       early_phase — checkpoints with step in [0, T+]
       late_phase  — checkpoints with step in [T+, expert_epochs]
  3. Run standard MTT (parameter matching, MTTDistiller) separately on
     each segment to produce two synthetic datasets.
  4. For each synthetic dataset, run the same cross-architecture
     evaluation protocol as experiment_matrix.py / h1_cross_arch_baseline.py /
     h2_cross_arch_phase_aware.py:
       - real and synthetic student training both use early stopping on the
         real validation set (eval_max_epochs=50, early_stop_patience=10,
         eval_batch_size=32, eval_lr=0.001)
       - real student: torch.manual_seed(0)
       - synthetic student: torch.manual_seed(1)
       - test MSE via Evaluator.test_on_real

Experts without a detected boundary are skipped (cannot segment).

Metric
------
  mse_ratio = transfer_mse / real_mse
  Lower is better. ~1 means synthetic nearly as good as real data.

Output
------
  results/phase_segment_mtt.csv

Run
---
  python -m example.experiments.h3_phase_segment_mtt
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

# Experts to segment — must match h_detect_phase_boundary.py's ACTIVE_MODELS,
# since that script is what populates results/phase_boundaries.csv.
EXPERT_MODELS = ["DLinear", "MLP", "CNN"]

# Students for the cross-architecture evaluation.
ACTIVE_MODELS = ["DLinear", "LSTM", "MLP", "CNN"]

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

DISTILL_CONFIG = {
    "seq_len":  96,
    "pred_len": 96,

    # Expert training — must match h_detect_phase_boundary.py so that the
    # recorded trajectory (and its 'step' values) line up with the
    # boundary_epoch values in results/phase_boundaries.csv.
    "expert_epochs":   200,
    "expert_lr":       0.01,
    "expert_momentum": 0.9,

    # MTT distillation — same as experiment_matrix.py / h2_cross_arch_phase_aware.py
    "n_distill_steps":        300,
    "n_synthetic":            384,
    "synthetic_lr":           5.0,
    "student_lr":             0.01,
    "student_steps":          20,
    "snapshot_student_steps": 50,
    "trajectory_gap":         5,
    "batch_size":             64,

    # Evaluation — same protocol as experiment_matrix.py / h1_cross_arch_baseline.py /
    # h2_cross_arch_phase_aware.py. NOTE: experiment_matrix.py's DISTILL_CONFIG
    # declares eval_max_epochs=300, but its __main__ block overrides this to 50
    # before running — 50 is what actually executes, so that's the value used here.
    "eval_max_epochs":     50,
    "eval_lr":             0.001,
    "eval_batch_size":     32,
    "early_stop_patience": 10,
}


# =============================================================================
# CSV
# =============================================================================

RESULTS_DIR  = Path(__file__).parent / "results"
CSV_PATH     = RESULTS_DIR / "phase_segment_mtt.csv"
BOUNDARY_CSV = RESULTS_DIR / "phase_boundaries.csv"

CSV_COLUMNS = [
    "experiment",
    "dataset",
    "expert_model",
    "student_model",
    "seed",
    "phase",            # early_phase | late_phase
    "boundary_epoch",
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
        f"seed={row['seed']} | phase={row['phase']} | "
        f"ratio={row.get('mse_ratio', '?')}"
    )


def row_exists(dataset, expert, student, seed, phase):
    if not CSV_PATH.exists():
        return False
    try:
        df = pd.read_csv(CSV_PATH)
        mask = (
            (df["dataset"]       == dataset) &
            (df["expert_model"]  == expert)  &
            (df["student_model"] == student) &
            (df["seed"]          == seed)    &
            (df["phase"]         == phase)   &
            (df["transfer_mse"].notna())     &
            (df["transfer_mse"]  != "")
        )
        return bool(mask.any())
    except Exception:
        return False


# =============================================================================
# PHASE BOUNDARY LOADING (velocity method — see h_detect_phase_boundary.py)
# =============================================================================

def load_phase_boundaries() -> dict:
    """Load detected T+ values: {(dataset, model, seed): boundary_epoch_or_None}."""
    boundaries = {}

    if not BOUNDARY_CSV.exists():
        print(
            f"[WARN] {BOUNDARY_CSV} not found. "
            f"Run h_detect_phase_boundary.py first to auto-detect T+.\n"
            f"       All experts will be skipped."
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

    # RNG state right after expert training — restore this before each
    # segment's distillation call so synthetic_init is reproducible and
    # comparable between early_phase and late_phase.
    rng_state = torch.get_rng_state()
    return recorder, n_pairs, rng_state


# =============================================================================
# TRAJECTORY SEGMENTATION
# =============================================================================

def build_segment_recorder(full_trajectory, phase, boundary):
    """Return a SimpleRecorder whose .trajectory is restricted to one phase.

    early_phase — checkpoints with step in [0, boundary]
    late_phase  — checkpoints with step in [boundary, expert_epochs]

    The boundary checkpoint is shared (inclusive) by both segments.
    """
    if phase == "early_phase":
        segment = [c for c in full_trajectory if c["step"] <= boundary]
    else:
        segment = [c for c in full_trajectory if c["step"] >= boundary]

    recorder = SimpleRecorder(record_every=1)
    recorder.trajectory = segment
    return recorder


# =============================================================================
# DISTILLATION — standard MTT (parameter matching) on a trajectory segment
# =============================================================================

def distill_segment_mtt(model_name, segment_recorder, data, cfg, device, seed):
    """Standard MTT — parameter matching, restricted to a trajectory segment.

    Caller must restore the RNG state captured right after expert training
    (see train_expert) before calling this, so synthetic_init is reproducible
    and comparable between early_phase and late_phase.
    """
    sl = data["seq_len"]
    pl = data["pred_len"]
    nf = data["in_features"]

    def factory():
        return make_model(model_name, sl, pl, nf)

    initializer    = RandomSampleInitializer()
    synthetic_init = initializer.initialize_sequence(data["raw_train"], cfg["n_synthetic"])

    # No `device` kwarg — defaults to "cpu", matching experiment_matrix.py's
    # MTTDistiller construction exactly.
    distiller = MTTDistiller(
        initializer=initializer,
        matcher=MSEMatcher(),
        model_factory=factory,
        expert_recorder=segment_recorder,
        expert_epochs=cfg["trajectory_gap"],
        syn_batch_size=cfg["batch_size"],
        synthetic_lr=cfg["synthetic_lr"],
        student_lr=cfg["student_lr"],
        student_steps=cfg["student_steps"],
        snapshot_student_steps=cfg["snapshot_student_steps"],
        seq_len=sl,
        pred_len=pl,
    )

    return distiller.distill(
        synthetic_init=synthetic_init,
        n_steps=cfg["n_distill_steps"],
        val_data=data["val_data"],
    )


# =============================================================================
# EVALUATION (same protocol as experiment_matrix.py / h1_cross_arch_baseline.py)
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
    """Train student on synthetic windows with early stopping on real val; return test MSE.

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
    print("H3 PHASE-SEGMENT MTT — EARLY-PHASE vs LATE-PHASE SYNTHETIC DATA")
    print("=" * 80)

    # {(dataset, model, seed): boundary_epoch_or_None}, keyed by EXPERT model
    phase_boundaries = load_phase_boundaries()

    for dataset_name in ACTIVE_DATASETS:

        data = load_dataset(dataset_name, cfg)

        # real_mse is the same for a given (student, seed) regardless of
        # which expert/phase generated the synthetic data.
        real_mse_cache: dict = {}   # (student_name, seed) -> float

        for expert_name in EXPERT_MODELS:

            for seed in SEEDS:

                boundary = phase_boundaries.get((dataset_name, expert_name, seed))

                print("\n" + "=" * 60)
                print(
                    f"Dataset={dataset_name} | Expert={expert_name} | "
                    f"Seed={seed} | boundary={boundary}"
                )
                print("=" * 60)

                if boundary is None:
                    print(
                        f"  [SKIP] No boundary detected for {expert_name} on "
                        f"{dataset_name} (seed={seed}) — cannot segment trajectory."
                    )
                    continue

                phases_needed = ["early_phase", "late_phase"]

                if all(
                    row_exists(dataset_name, expert_name, s, seed, phase)
                    for s in ACTIVE_MODELS
                    for phase in phases_needed
                ):
                    print("  [RESUME] All students/phases already in CSV — skipping.")
                    continue

                # ---------------------------------------------------------
                # Train expert (shared trajectory for both phases)
                # ---------------------------------------------------------
                try:
                    recorder, n_pairs, post_expert_rng_state = train_expert(
                        expert_name, data, cfg, device, seed
                    )
                except Exception as exc:
                    print(f"  [FAIL] Expert training: {exc}")
                    for student_name in ACTIVE_MODELS:
                        for phase in phases_needed:
                            if not row_exists(dataset_name, expert_name, student_name, seed, phase):
                                append_csv({
                                    "experiment":     "phase_segment_mtt",
                                    "dataset":        dataset_name,
                                    "expert_model":   expert_name,
                                    "student_model":  student_name,
                                    "seed":           seed,
                                    "phase":          phase,
                                    "boundary_epoch": boundary,
                                    "notes":          f"expert_training_failed: {str(exc)[:120]}",
                                })
                    continue

                full_traj = recorder.trajectory

                # ---------------------------------------------------------
                # For each phase: build the segment recorder, distill, evaluate
                # ---------------------------------------------------------
                for phase in phases_needed:

                    if all(
                        row_exists(dataset_name, expert_name, s, seed, phase)
                        for s in ACTIVE_MODELS
                    ):
                        print(f"\n  [RESUME] {phase}: all students already in CSV — skipping.")
                        continue

                    segment_recorder = build_segment_recorder(full_traj, phase, boundary)
                    n_ckpts = len(segment_recorder.trajectory)

                    print(f"\n  -- Phase: {phase}  ({n_ckpts} checkpoints) --")

                    if n_ckpts < cfg["trajectory_gap"] + 1:
                        print(
                            f"  [SKIP] Too few checkpoints ({n_ckpts}) for "
                            f"trajectory_gap={cfg['trajectory_gap']}. Need >= {cfg['trajectory_gap'] + 1}."
                        )
                        for student_name in ACTIVE_MODELS:
                            if not row_exists(dataset_name, expert_name, student_name, seed, phase):
                                append_csv({
                                    "experiment":        "phase_segment_mtt",
                                    "dataset":           dataset_name,
                                    "expert_model":      expert_name,
                                    "student_model":     student_name,
                                    "seed":              seed,
                                    "phase":             phase,
                                    "boundary_epoch":    boundary,
                                    "n_pairs_available": n_ckpts,
                                    "notes":             f"skipped: only {n_ckpts} checkpoints available",
                                })
                        continue

                    # Restore RNG state to right after expert training so
                    # synthetic_init is reproducible/comparable between phases.
                    torch.set_rng_state(post_expert_rng_state)

                    try:
                        synthetic_seq = distill_segment_mtt(
                            expert_name, segment_recorder, data, cfg, device, seed
                        )
                    except Exception as exc:
                        print(f"  [FAIL] {phase} distillation: {exc}")
                        for student_name in ACTIVE_MODELS:
                            if not row_exists(dataset_name, expert_name, student_name, seed, phase):
                                append_csv({
                                    "experiment":        "phase_segment_mtt",
                                    "dataset":           dataset_name,
                                    "expert_model":      expert_name,
                                    "student_model":     student_name,
                                    "seed":              seed,
                                    "phase":             phase,
                                    "boundary_epoch":    boundary,
                                    "n_pairs_available": n_ckpts,
                                    "notes":             f"distillation_failed: {str(exc)[:120]}",
                                })
                        continue

                    # -------------------------------------------------
                    # Evaluate every student on this phase's synthetic data
                    # -------------------------------------------------
                    for student_name in ACTIVE_MODELS:

                        if row_exists(dataset_name, expert_name, student_name, seed, phase):
                            print(f"    [RESUME] {student_name} already done.")
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

                        try:
                            transfer_mse = eval_on_synthetic(
                                student_name, synthetic_seq, data, cfg, device, seed,
                            )
                        except Exception as exc:
                            print(f"    [FAIL] {phase} synthetic eval: {exc}")
                            append_csv({
                                "experiment":        "phase_segment_mtt",
                                "dataset":           dataset_name,
                                "expert_model":      expert_name,
                                "student_model":     student_name,
                                "seed":              seed,
                                "phase":             phase,
                                "boundary_epoch":    boundary,
                                "real_mse":          real_mse,
                                "n_pairs_available": n_ckpts,
                                "notes":             f"synthetic_eval_failed: {str(exc)[:120]}",
                            })
                            continue

                        mse_ratio = (
                            transfer_mse / real_mse
                            if (real_mse > 0 and not np.isnan(real_mse))
                            else float("nan")
                        )

                        print(
                            f"    [{phase}] real={real_mse:.6f} | "
                            f"transfer={transfer_mse:.6f} | ratio={mse_ratio:.3f}"
                        )

                        append_csv({
                            "experiment":        "phase_segment_mtt",
                            "dataset":           dataset_name,
                            "expert_model":      expert_name,
                            "student_model":     student_name,
                            "seed":              seed,
                            "phase":             phase,
                            "boundary_epoch":    boundary,
                            "real_mse":          real_mse,
                            "transfer_mse":      transfer_mse,
                            "mse_ratio":         mse_ratio,
                            "n_pairs_available": n_ckpts,
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
    # EXPERT_MODELS[:] = ["DLinear"]
    # ACTIVE_MODELS[:] = ["DLinear", "CNN"]
    # SEEDS[:] = [42]
    # DISTILL_CONFIG["expert_epochs"]   = 5
    # DISTILL_CONFIG["n_distill_steps"] = 10

    main()
