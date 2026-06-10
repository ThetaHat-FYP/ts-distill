"""
Phase Boundary Detection — Method A: Gradient Magnitude & Parameter Distance
=============================================================================

Automatically identifies T+ (the phase boundary) from the expert training
trajectory without any manual guessing.

Algorithm
---------
For each (dataset, model):
  1. Train the expert for expert_epochs and record weight checkpoints at
     every epoch using SimpleRecorder.
  2. For every consecutive checkpoint pair (theta_t, theta_{t+1}) compute
     the squared L2 distance across all parameters:

         delta_t = ||theta_t - theta_{t+M}||^2    (M = step_gap, default 1)

     This measures how far the expert's weights moved during epoch t.
  3. Smooth delta_t with a rolling average of window size W to reduce noise.
  4. Compute the reference movement from the first init_window epochs.
  5. Detect T+ (boundary epoch) as the FIRST epoch after the reference window
     where the smoothed movement drops below (threshold * reference):

         T+ = first t >= init_window  where  delta_smooth[t] < threshold * ref

     Default threshold = 0.05 means the expert has slowed to <= 5% of its
     initial movement speed — a 95% slowdown.

Outputs
-------
  results/phase_boundaries.csv      — one row per (dataset, model, seed)
                                      with the detected T+
  results/phase_boundary_deltas.csv — one row per (dataset, model, seed, epoch)
                                      with raw and smoothed delta values
                                      (use this for plotting the curves)

Run
---
  python -m example.experiments.h_detect_phase_boundary
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent.parent))

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader
from ts_distill.models.factory import create_model
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder


# =============================================================================
# CONFIG
# =============================================================================

ACTIVE_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
ACTIVE_MODELS   = ["DLinear", "LSTM", "MLP", "CNN"]

SEEDS = [42]

MODEL_CONFIGS = {
    "DLinear": {"individual": False},
    "LSTM":    {"hidden_dim": 16, "num_layer": 1},
    "MLP":     {},
    "CNN":     {},
}

DATASET_CONFIGS = {
    "ETTh1": {
        "csv_path":    r"C:\fyp\ts-distill\example/ETTh1.csv",
        "split_mode":  "benchmark_borders",
        "border1s":    [0, 12 * 30 * 24 - 96,  12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s":    [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        "in_features": 7,
    },
    "ETTh2": {
        "csv_path":    r"C:\fyp\ts-distill\example/ETTh2.csv",
        "split_mode":  "benchmark_borders",
        "border1s":    [0, 12 * 30 * 24 - 96,  12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s":    [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        "in_features": 7,
    },
    "ETTm1": {
        "csv_path":    r"C:\fyp\ts-distill\example/ETTm1.csv",
        "split_mode":  "benchmark_borders",
        "border1s":    [0, 12 * 30 * 96 - 96,  12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s":    [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        "in_features": 7,
    },
    "ETTm2": {
        "csv_path":    r"C:\fyp\ts-distill\example/ETTm2.csv",
        "split_mode":  "benchmark_borders",
        "border1s":    [0, 12 * 30 * 96 - 96,  12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s":    [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        "in_features": 7,
    },
}

EXPERT_CONFIG = {
    "seq_len":         96,
    "pred_len":        96,
    "expert_epochs":   80,
    "expert_lr":       0.01,
    "expert_momentum": 0.9,
    "batch_size":      64,
}

BOUNDARY_CONFIG = {
    # M in the formula: distance between checkpoints M steps apart.
    # 1 = adjacent epochs (most granular, slightly noisier).
    "step_gap": 1,

    # Rolling average window applied to the raw delta sequence.
    "smoothing_window": 5,

    # Number of initial epochs used as the reference movement baseline.
    "init_window": 5,

    # T+ is where smoothed delta drops below (threshold * reference).
    # 0.05 = 95% slowdown relative to the initial phase.
    "threshold": 0.05,
}


# =============================================================================
# CSV PATHS
# =============================================================================

RESULTS_DIR   = Path(__file__).parent / "results"
BOUNDARY_CSV  = RESULTS_DIR / "phase_boundaries.csv"
DELTA_CSV     = RESULTS_DIR / "phase_boundary_deltas.csv"

BOUNDARY_COLS = [
    "dataset", "model", "seed",
    "boundary_epoch",      # detected T+ (None if no clear boundary found)
    "n_checkpoints",       # total checkpoints recorded
    "threshold",           # fraction used (e.g. 0.05)
    "smoothing_window",    # rolling window size W
    "init_window",         # reference window size
    "step_gap",            # M in the formula
    "ref_delta",           # mean delta of first init_window epochs (reference)
    "notes",
]

DELTA_COLS = [
    "dataset", "model", "seed",
    "epoch",               # 1-indexed epoch number
    "delta_raw",           # raw squared L2 distance ||theta_t - theta_{t+M}||^2
    "delta_smooth",        # smoothed with rolling window
    "delta_norm",          # delta_smooth / ref_delta (normalized to initial)
    "is_boundary",         # True for the detected T+ epoch
]


def append_boundary(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = BOUNDARY_CSV.exists()
    with open(BOUNDARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOUNDARY_COLS)
        if not exists:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in BOUNDARY_COLS})


def append_deltas(rows: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = DELTA_CSV.exists()
    with open(DELTA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_COLS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in DELTA_COLS})


def boundary_exists(dataset, model, seed):
    if not BOUNDARY_CSV.exists():
        return False
    try:
        df   = pd.read_csv(BOUNDARY_CSV)
        mask = (
            (df["dataset"] == dataset) &
            (df["model"]   == model)   &
            (df["seed"]    == seed)
        )
        return bool(mask.any())
    except Exception:
        return False


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(dataset_name, cfg):
    ds_cfg      = DATASET_CONFIGS[dataset_name]
    seq_len     = cfg["seq_len"]
    pred_len    = cfg["pred_len"]
    window_size = seq_len + pred_len

    df_raw = pd.read_csv(ds_cfg["csv_path"])
    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    train_start, train_end, *_ = get_data_splits(
        values=values,
        window_size=window_size,
        seq_len=seq_len,
        dataset_cfg=ds_cfg,
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data_scaled = scaler.transform(values)

    train_data = torch.tensor(
        make_windows(data_scaled[train_start:train_end], window_size),
        dtype=torch.float32,
    )

    return {"train_data": train_data, "seq_len": seq_len, "pred_len": pred_len}


# =============================================================================
# EXPERT TRAINING
# =============================================================================

def train_expert_and_record(model_name, dataset_name, data, cfg, device, seed):
    """Train expert and return the full SimpleRecorder trajectory."""
    torch.manual_seed(seed)

    ds_cfg      = DATASET_CONFIGS[dataset_name]
    in_features = ds_cfg["in_features"]

    model = create_model(
        model_type=model_name,
        seq_len=cfg["seq_len"],
        pred_len=cfg["pred_len"],
        in_features=in_features,
        model_kwargs=MODEL_CONFIGS[model_name],
    ).to(device)

    recorder = SimpleRecorder(record_every=1)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=cfg["expert_lr"],
            momentum=cfg["expert_momentum"],
        ),
        criterion=torch.nn.MSELoss(),
        device=device,
        seq_len=cfg["seq_len"],
    )

    trainer.fit(
        dataloader=MiniBatchLoader(data["train_data"], batch_size=cfg["batch_size"]),
        epochs=cfg["expert_epochs"],
        callbacks=[recorder, SimpleCallback()],
    )

    return recorder.get_trajectory()


# =============================================================================
# PARAMETER DISTANCE
# =============================================================================

def checkpoint_distance(ckpt_a: dict, ckpt_b: dict) -> float:
    """
    Compute the squared L2 distance between two weight checkpoints across
    all shared parameter tensors.

        d = sum_over_layers ||theta_a_layer - theta_b_layer||^2

    Only parameters present in both checkpoints are included (handles
    rare cases where buffer names differ between steps).
    """
    total = 0.0
    weights_a = ckpt_a["weights"]
    weights_b = ckpt_b["weights"]
    for name in weights_a:
        if name in weights_b:
            diff = weights_a[name].float() - weights_b[name].float()
            total += diff.pow(2).sum().item()
    return total


# =============================================================================
# PHASE BOUNDARY DETECTION
# =============================================================================

def compute_deltas(trajectory: list, step_gap: int = 1) -> tuple:
    """
    Compute delta_t = ||theta_t - theta_{t+step_gap}||^2 for each t.

    Returns:
        epochs  — list of epoch numbers (step of the LATER checkpoint)
        deltas  — list of raw squared L2 distances
    """
    epochs = []
    deltas = []

    for i in range(len(trajectory) - step_gap):
        ckpt_a = trajectory[i]
        ckpt_b = trajectory[i + step_gap]
        d = checkpoint_distance(ckpt_a, ckpt_b)
        epochs.append(ckpt_b["step"])   # label by the later step
        deltas.append(d)

    return epochs, deltas


def smooth(values: list, window: int) -> np.ndarray:
    """Apply a centred rolling average. Edge values use available data only."""
    arr = np.array(values, dtype=np.float64)
    n   = len(arr)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def detect_boundary(
    epochs:          list,
    deltas:          list,
    smoothing_window: int   = 5,
    init_window:      int   = 5,
    threshold:        float = 0.05,
) -> dict:
    """
    Detect T+ from a delta sequence.

    Args:
        epochs:           List of epoch numbers matching each delta.
        deltas:           Raw squared L2 distances per epoch.
        smoothing_window: Rolling average window size.
        init_window:      Number of initial epochs for the reference baseline.
        threshold:        T+ where smoothed delta < threshold * ref_delta.

    Returns:
        dict with keys: boundary_epoch, ref_delta, delta_smooth, delta_norm,
                        boundary_idx (index into epochs list, or None)
    """
    if len(deltas) < init_window + 1:
        return {
            "boundary_epoch": None,
            "boundary_idx":   None,
            "ref_delta":      float("nan"),
            "delta_smooth":   np.array(deltas),
            "delta_norm":     np.ones(len(deltas)),
        }

    delta_smooth = smooth(deltas, smoothing_window)

    # Reference: mean movement over the first init_window epochs
    ref_delta = float(delta_smooth[:init_window].mean())

    if ref_delta < 1e-30:
        return {
            "boundary_epoch": None,
            "boundary_idx":   None,
            "ref_delta":      ref_delta,
            "delta_smooth":   delta_smooth,
            "delta_norm":     np.zeros(len(deltas)),
        }

    delta_norm = delta_smooth / ref_delta

    # Search for T+: first epoch AFTER the init window where movement drops
    # below threshold (i.e., has slowed by more than (1 - threshold) * 100 %)
    boundary_epoch = None
    boundary_idx   = None
    for i in range(init_window, len(deltas)):
        if delta_norm[i] < threshold:
            boundary_epoch = epochs[i]
            boundary_idx   = i
            break

    return {
        "boundary_epoch": boundary_epoch,
        "boundary_idx":   boundary_idx,
        "ref_delta":      ref_delta,
        "delta_smooth":   delta_smooth,
        "delta_norm":     delta_norm,
    }


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(device, expert_cfg, boundary_cfg):

    step_gap        = boundary_cfg["step_gap"]
    smoothing_window = boundary_cfg["smoothing_window"]
    init_window     = boundary_cfg["init_window"]
    threshold       = boundary_cfg["threshold"]

    print("=" * 72)
    print("PHASE BOUNDARY DETECTION — Gradient Magnitude & Parameter Distance")
    print("=" * 72)
    print(f"  step_gap={step_gap}  smoothing_window={smoothing_window}  "
          f"init_window={init_window}  threshold={threshold}")
    print(f"  Rule: T+ = first epoch where delta_smooth < {threshold} * "
          f"mean(delta[0:{init_window}])")
    print()

    for dataset_name in ACTIVE_DATASETS:

        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'='*60}")

        data = load_dataset(dataset_name, expert_cfg)

        for model_name in ACTIVE_MODELS:

            for seed in SEEDS:

                print(f"\n  Model={model_name} | Seed={seed}")

                if boundary_exists(dataset_name, model_name, seed):
                    print("  [RESUME] Already in CSV — skipping.")
                    continue

                # ---------------------------------------------------------
                # Train expert and record trajectory
                # ---------------------------------------------------------
                try:
                    trajectory = train_expert_and_record(
                        model_name, dataset_name, data, expert_cfg, device, seed
                    )
                except Exception as exc:
                    print(f"  [FAIL] Expert training: {exc}")
                    append_boundary({
                        "dataset": dataset_name, "model": model_name,
                        "seed": seed, "notes": f"expert_failed: {str(exc)[:120]}",
                    })
                    continue

                n_checkpoints = len(trajectory)
                print(f"  Checkpoints recorded: {n_checkpoints}")

                # ---------------------------------------------------------
                # Compute delta sequence
                # ---------------------------------------------------------
                epochs, deltas = compute_deltas(trajectory, step_gap=step_gap)

                # ---------------------------------------------------------
                # Detect boundary
                # ---------------------------------------------------------
                result = detect_boundary(
                    epochs, deltas,
                    smoothing_window=smoothing_window,
                    init_window=init_window,
                    threshold=threshold,
                )

                boundary_epoch = result["boundary_epoch"]
                ref_delta      = result["ref_delta"]
                delta_smooth   = result["delta_smooth"]
                delta_norm     = result["delta_norm"]
                boundary_idx   = result["boundary_idx"]

                # ---------------------------------------------------------
                # Print summary
                # ---------------------------------------------------------
                if boundary_epoch is not None:
                    print(f"  T+ detected at epoch {boundary_epoch}  "
                          f"(delta_norm={delta_norm[boundary_idx]:.4f})")
                else:
                    print(f"  T+ NOT detected — movement never dropped below "
                          f"{threshold * 100:.0f}% of initial.")
                    print(f"  Final delta_norm = {delta_norm[-1]:.4f}  "
                          f"(threshold = {threshold:.2f})")

                print(f"  ref_delta = {ref_delta:.6e}")
                print(f"  delta_norm range: [{delta_norm.min():.4f}, "
                      f"{delta_norm.max():.4f}]")

                # Print the normalized delta curve in a compact form
                print(f"  Normalized deltas (epoch : norm_delta):")
                for i, (ep, nd) in enumerate(zip(epochs, delta_norm)):
                    marker = " <-- T+" if i == boundary_idx else ""
                    if i < init_window or i == boundary_idx or i % 10 == 0:
                        print(f"    epoch {ep:>3}: {nd:.4f}{marker}")

                # ---------------------------------------------------------
                # Write CSVs
                # ---------------------------------------------------------
                append_boundary({
                    "dataset":         dataset_name,
                    "model":           model_name,
                    "seed":            seed,
                    "boundary_epoch":  boundary_epoch if boundary_epoch is not None else "",
                    "n_checkpoints":   n_checkpoints,
                    "threshold":       threshold,
                    "smoothing_window": smoothing_window,
                    "init_window":     init_window,
                    "step_gap":        step_gap,
                    "ref_delta":       round(ref_delta, 8),
                    "notes":           "" if boundary_epoch is not None
                                       else "no_boundary_found",
                })

                delta_rows = []
                for i, (ep, d_raw, d_sm, d_nm) in enumerate(
                    zip(epochs, deltas, delta_smooth, delta_norm)
                ):
                    delta_rows.append({
                        "dataset":      dataset_name,
                        "model":        model_name,
                        "seed":         seed,
                        "epoch":        ep,
                        "delta_raw":    round(float(d_raw), 8),
                        "delta_smooth": round(float(d_sm),  8),
                        "delta_norm":   round(float(d_nm),  8),
                        "is_boundary":  (i == boundary_idx),
                    })
                append_deltas(delta_rows)

    # Summary table
    print("\n\n" + "=" * 72)
    print("DETECTED PHASE BOUNDARIES SUMMARY")
    print("=" * 72)
    if BOUNDARY_CSV.exists():
        try:
            df = pd.read_csv(BOUNDARY_CSV)
            print(f"\n{'Dataset':<10} {'Model':<10} {'Seed':>5} {'T+':>8}  Notes")
            print("-" * 50)
            for _, row in df.iterrows():
                t_plus = row["boundary_epoch"] if pd.notna(row["boundary_epoch"]) else "---"
                notes  = row["notes"] if pd.notna(row["notes"]) else ""
                print(f"{row['dataset']:<10} {row['model']:<10} "
                      f"{int(row['seed']):>5} {str(t_plus):>8}  {notes}")
        except Exception as exc:
            print(f"  Could not read summary: {exc}")


# =============================================================================
# ENTRY
# =============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    run(device, EXPERT_CONFIG, BOUNDARY_CONFIG)
    print(f"\nBoundaries : {BOUNDARY_CSV}")
    print(f"Delta curves: {DELTA_CSV}")


if __name__ == "__main__":

    # ----------------------------------------------------------
    # QUICK SMOKE TEST (uncomment to verify the pipeline runs)
    # ----------------------------------------------------------
    # ACTIVE_DATASETS[:] = ["ETTh1"]
    # ACTIVE_MODELS[:] = ["LSTM"]
    # SEEDS[:] = [42]
    # EXPERT_CONFIG["expert_epochs"] = 10
    # BOUNDARY_CONFIG["init_window"] = 2

    main()
