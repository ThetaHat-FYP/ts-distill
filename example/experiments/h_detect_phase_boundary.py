"""
Phase Boundary Detection — Validation Loss Velocity
=====================================================

(Distinct from h_detect_phase_boundary_valloss.py's "Method B: Validation
Loss Plateau" — this script uses a velocity-threshold rule instead of a
best-so-far + patience plateau rule.)

Automatically identifies T+ (the phase boundary) from the expert training
trajectory without any manual guessing.

Algorithm
---------
For each (dataset, model):
  1. Train the expert for expert_epochs, evaluating on the validation split
     at the end of every epoch (model.eval() + torch.no_grad()) to record
     val_loss[t] = mean validation MSE after epoch t.
  2. Compute the optimization velocity between consecutive epochs:

         velocity_t = val_loss[t-1] - val_loss[t]

     This measures how much the validation loss improved during epoch t.
     Negative velocities (validation loss got worse) are clipped to 0.0 so
     that noisy fluctuations do not skew the baseline / boundary calculation.
  3. Smooth velocity_t with a centered rolling average of window size W to
     reduce noise.
  4. Compute the reference velocity from the first init_window epochs.
  5. Detect T+ (boundary epoch) as the FIRST epoch after the reference window
     where the smoothed velocity drops below (threshold * reference):

         T+ = first t >= init_window  where  velocity_smooth[t] < threshold * ref

     Default threshold = 0.10 means the expert's validation loss has slowed
     to <= 10% of its initial improvement speed — a 90% slowdown, i.e. the
     model has entered the "late phase" where it is mostly fine-tuning /
     converged rather than rapidly learning.

Outputs
-------
  results/phase_boundaries.csv      — one row per (dataset, model, seed)
                                      with the detected T+
  results/phase_boundary_deltas.csv — one row per (dataset, model, seed, epoch)
                                      with val_loss and raw/smoothed/normalized
                                      velocity values (use this for plotting)

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


# =============================================================================
# CONFIG
# =============================================================================

ACTIVE_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2","weather"]
ACTIVE_MODELS   = ["DLinear", "MLP", "CNN"]

SEEDS = [123]

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
    "weather": {
        "csv_path":    r"C:\fyp\ts-distill\example/weather.csv",
        "split_mode":  "ratios",
        "train_ratio": 0.7,
        "val_ratio":   0.1,
        "in_features": 21,
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
    # M in the original formula: distance between checkpoints M steps apart.
    # Kept for CSV/back-compat purposes; the velocity formula is always
    # computed between adjacent epochs (velocity_t = val_loss[t-1] - val_loss[t]).
    "step_gap": 1,

    # Rolling average window applied to the raw velocity sequence.
    "smoothing_window": 5,

    # Number of initial epochs used as the reference velocity baseline.
    "init_window": 5,

    # T+ is where smoothed velocity drops below (threshold * reference).
    # 0.10 = 90% slowdown relative to the initial phase.
    "threshold": 0.10,
}


# =============================================================================
# CSV PATHS
# =============================================================================

RESULTS_DIR   = Path(__file__).parent / "results"
BOUNDARY_CSV  = RESULTS_DIR / "phase_boundaries_new.csv"
DELTA_CSV     = RESULTS_DIR / "phase_boundary_deltas_new.csv"

BOUNDARY_COLS = [
    "dataset", "model", "seed",
    "boundary_epoch",      # detected T+ (None if no clear boundary found)
    "n_checkpoints",       # total epochs trained / val_loss values recorded
    "threshold",           # fraction used (e.g. 0.05)
    "smoothing_window",    # rolling window size W
    "init_window",         # reference window size
    "step_gap",            # M in the original formula (fixed at 1 here)
    "ref_delta",           # mean velocity of first init_window epochs (reference)
    "notes",
]

DELTA_COLS = [
    "dataset", "model", "seed",
    "epoch",               # 1-indexed epoch number (later epoch of the pair)
    "val_loss",            # validation MSE after this epoch
    "velocity_raw",        # val_loss[t-1] - val_loss[t], clipped to >= 0
    "velocity_smooth",     # smoothed with rolling window
    "velocity_norm",       # velocity_smooth / ref_velocity (normalized to initial)
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

    train_start, train_end, val_start, val_end, *_ = get_data_splits(
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
    val_data = torch.tensor(
        make_windows(data_scaled[val_start:val_end], window_size),
        dtype=torch.float32,
    )

    return {
        "train_data": train_data,
        "val_data":   val_data,
        "seq_len":    seq_len,
        "pred_len":   pred_len,
    }


# =============================================================================
# EXPERT TRAINING
# =============================================================================

def train_expert_and_record(model_name, dataset_name, data, cfg, device, seed):
    """Train expert and return the per-epoch validation MSE trace.

    At the end of every epoch, evaluates the model on the validation split
    via model.eval() + torch.no_grad() and records the mean validation MSE.
    """
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

    train_loader = MiniBatchLoader(data["train_data"], batch_size=cfg["batch_size"])
    val_loader   = MiniBatchLoader(data["val_data"],   batch_size=cfg["batch_size"])

    val_losses = []
    for _ in range(cfg["expert_epochs"]):
        trainer.train_epoch(train_loader)
        val_loss = trainer.eval_epoch(val_loader)
        val_losses.append(val_loss)

    return val_losses


# =============================================================================
# VALIDATION LOSS VELOCITY
# =============================================================================

def compute_velocity(val_losses: list) -> tuple:
    """
    Compute velocity_t = val_loss[t-1] - val_loss[t] for each consecutive
    pair of epochs. Negative velocities (val loss got worse) are clipped to
    0.0 so noisy fluctuations don't skew the baseline / boundary detection.

    Returns:
        epochs       — list of 1-indexed epoch numbers (the LATER epoch of
                        each pair)
        velocity_raw — list of clipped raw velocity values
    """
    epochs       = []
    velocity_raw = []

    for t in range(1, len(val_losses)):
        v = val_losses[t - 1] - val_losses[t]
        v = max(v, 0.0)
        epochs.append(t + 1)  # later epoch, 1-indexed
        velocity_raw.append(v)

    return epochs, velocity_raw


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
    epochs:           list,
    velocities:       list,
    smoothing_window: int   = 5,
    init_window:      int   = 5,
    threshold:        float = 0.05,
) -> dict:
    """
    Detect T+ from a velocity sequence.

    Args:
        epochs:           List of epoch numbers matching each velocity value.
        velocities:       Raw (clipped) velocity values per epoch.
        smoothing_window: Rolling average window size.
        init_window:      Number of initial epochs for the reference baseline.
        threshold:        T+ where smoothed velocity < threshold * ref_velocity.

    Returns:
        dict with keys: boundary_epoch, ref_velocity, velocity_smooth,
                        velocity_norm, boundary_idx (index into epochs list,
                        or None)
    """
    if len(velocities) < init_window + 1:
        return {
            "boundary_epoch":  None,
            "boundary_idx":    None,
            "ref_velocity":    float("nan"),
            "velocity_smooth": np.array(velocities),
            "velocity_norm":   np.ones(len(velocities)),
        }

    velocity_smooth = smooth(velocities, smoothing_window)

    # Reference: mean velocity over the first init_window epochs
    ref_velocity = float(velocity_smooth[:init_window].mean())

    if ref_velocity < 1e-30:
        return {
            "boundary_epoch":  None,
            "boundary_idx":    None,
            "ref_velocity":    ref_velocity,
            "velocity_smooth": velocity_smooth,
            "velocity_norm":   np.zeros(len(velocities)),
        }

    velocity_norm = velocity_smooth / ref_velocity

    # Search for T+: first epoch AFTER the init window where velocity drops
    # below threshold (i.e., validation loss has slowed by more than
    # (1 - threshold) * 100 %)
    boundary_epoch = None
    boundary_idx   = None
    for i in range(init_window, len(velocities)):
        if velocity_norm[i] < threshold:
            boundary_epoch = epochs[i]
            boundary_idx   = i
            break

    return {
        "boundary_epoch":  boundary_epoch,
        "boundary_idx":    boundary_idx,
        "ref_velocity":    ref_velocity,
        "velocity_smooth": velocity_smooth,
        "velocity_norm":   velocity_norm,
    }


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(device, expert_cfg, boundary_cfg):

    step_gap         = boundary_cfg["step_gap"]
    smoothing_window = boundary_cfg["smoothing_window"]
    init_window      = boundary_cfg["init_window"]
    threshold        = boundary_cfg["threshold"]

    print("=" * 72)
    print("PHASE BOUNDARY DETECTION — Validation Loss Velocity")
    print("=" * 72)
    print(f"  smoothing_window={smoothing_window}  "
          f"init_window={init_window}  threshold={threshold}")
    print(f"  Rule: T+ = first epoch where velocity_smooth < {threshold} * "
          f"mean(velocity[0:{init_window}])")
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
                # Train expert and record per-epoch validation loss
                # ---------------------------------------------------------
                try:
                    val_losses = train_expert_and_record(
                        model_name, dataset_name, data, expert_cfg, device, seed
                    )
                except Exception as exc:
                    print(f"  [FAIL] Expert training: {exc}")
                    append_boundary({
                        "dataset": dataset_name, "model": model_name,
                        "seed": seed, "notes": f"expert_failed: {str(exc)[:120]}",
                    })
                    continue

                n_checkpoints = len(val_losses)
                print(f"  Epochs trained: {n_checkpoints}")

                # ---------------------------------------------------------
                # Compute velocity sequence
                # ---------------------------------------------------------
                epochs, velocity_raw = compute_velocity(val_losses)

                # ---------------------------------------------------------
                # Detect boundary
                # ---------------------------------------------------------
                result = detect_boundary(
                    epochs, velocity_raw,
                    smoothing_window=smoothing_window,
                    init_window=init_window,
                    threshold=threshold,
                )

                boundary_epoch  = result["boundary_epoch"]
                ref_velocity    = result["ref_velocity"]
                velocity_smooth = result["velocity_smooth"]
                velocity_norm   = result["velocity_norm"]
                boundary_idx    = result["boundary_idx"]

                # ---------------------------------------------------------
                # Print summary
                # ---------------------------------------------------------
                if boundary_epoch is not None:
                    print(f"  T+ detected at epoch {boundary_epoch}  "
                          f"(velocity_norm={velocity_norm[boundary_idx]:.4f})")
                else:
                    print(f"  T+ NOT detected — velocity never dropped below "
                          f"{threshold * 100:.0f}% of initial.")
                    print(f"  Final velocity_norm = {velocity_norm[-1]:.4f}  "
                          f"(threshold = {threshold:.2f})")

                print(f"  ref_velocity = {ref_velocity:.6e}")
                print(f"  velocity_norm range: [{velocity_norm.min():.4f}, "
                      f"{velocity_norm.max():.4f}]")

                # Print the normalized velocity curve in a compact form
                print(f"  Normalized velocities (epoch : norm_velocity):")
                for i, (ep, nv) in enumerate(zip(epochs, velocity_norm)):
                    marker = " <-- T+" if i == boundary_idx else ""
                    if i < init_window or i == boundary_idx or i % 10 == 0:
                        print(f"    epoch {ep:>3}: {nv:.4f}{marker}")

                # ---------------------------------------------------------
                # Write CSVs
                # ---------------------------------------------------------
                append_boundary({
                    "dataset":          dataset_name,
                    "model":            model_name,
                    "seed":             seed,
                    "boundary_epoch":   boundary_epoch if boundary_epoch is not None else "",
                    "n_checkpoints":    n_checkpoints,
                    "threshold":        threshold,
                    "smoothing_window": smoothing_window,
                    "init_window":      init_window,
                    "step_gap":         step_gap,
                    "ref_delta":        round(ref_velocity, 8),
                    "notes":            "" if boundary_epoch is not None
                                        else "no_boundary_found",
                })

                delta_rows = []
                for i, (ep, v_raw, v_sm, v_nm) in enumerate(
                    zip(epochs, velocity_raw, velocity_smooth, velocity_norm)
                ):
                    delta_rows.append({
                        "dataset":         dataset_name,
                        "model":           model_name,
                        "seed":            seed,
                        "epoch":           ep,
                        "val_loss":        round(float(val_losses[ep - 1]), 8),
                        "velocity_raw":    round(float(v_raw), 8),
                        "velocity_smooth": round(float(v_sm),  8),
                        "velocity_norm":   round(float(v_nm),  8),
                        "is_boundary":     (i == boundary_idx),
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
