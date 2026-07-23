"""
Phase Boundary Detection — Method B: Validation Loss Plateau
==============================================================

Alternative to Method A (h_detect_phase_boundary.py, parameter-distance based).

Method A asks: "When do the expert's WEIGHTS stop moving?"
Method B asks: "When does the expert stop IMPROVING on real validation data?"

Both define a phase boundary T+, but from different signals. This script
detects T+ from the expert's validation-loss curve and writes it to a
separate CSV so it can be compared against Method A and swapped into
h_pred_vs_param_matching.py.

Algorithm
---------
For each (dataset, model):
  1. Train the expert for expert_epochs, recording weight checkpoints
     (SimpleRecorder, same as Method A) AND validation loss at every epoch.
  2. Smooth the val-loss curve with a rolling average of window W.
  3. Track the best (lowest) smoothed val loss seen so far.
  4. If a later epoch's smoothed val loss is not at least
     `min_delta_frac` better than the best-so-far, count it as
     "no improvement".
  5. T+ = the epoch index of the best-so-far value at the point where
     `patience` consecutive epochs have shown no improvement
     (i.e., the epoch where the plateau began).

Outputs
-------
  results/phase_boundaries_valloss.csv      — one row per (dataset, model, seed)
  results/phase_boundary_valloss_curves.csv — per-epoch val loss curve (for plotting)

Run
---
  python -m example.experiments.h_detect_phase_boundary_valloss
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
from torch.utils.data import DataLoader as TorchDataLoader
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder


# =============================================================================
# CONFIG
# =============================================================================

ACTIVE_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2","weather"]
ACTIVE_MODELS   = ["DLinear", "MLP", "CNN"]

SEEDS = [7,42,123]

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
    "eval_batch_size": 32,
}

# T+ = epoch of best smoothed val loss before a sustained plateau begins.
PLATEAU_CONFIG = {
    # Rolling average window applied to the raw val-loss sequence.
    "smoothing_window": 5,

    # Number of consecutive non-improving epochs that defines "plateau".
    "patience": 5,

    # An epoch only counts as "improvement" if it is at least this much
    # better (relative) than the best-so-far smoothed val loss.
    "min_delta_frac": 0.01,   # 1% relative improvement required
}


# =============================================================================
# CSV PATHS
# =============================================================================

RESULTS_DIR  = Path(__file__).parent / "results"
BOUNDARY_CSV = RESULTS_DIR / "all_seeds_phase_boundaries_valloss_new2.csv"
CURVE_CSV    = RESULTS_DIR / "all_seeds_phase_boundary_valloss_curves_new2.csv"

BOUNDARY_COLS = [
    "dataset", "model", "seed",
    "boundary_epoch",      # detected T+ (None if no plateau found)
    "n_checkpoints",       # total checkpoints recorded
    "best_val_loss",       # smoothed val loss at T+
    "final_val_loss",      # smoothed val loss at last epoch
    "smoothing_window",
    "patience",
    "min_delta_frac",
    "notes",
]

CURVE_COLS = [
    "dataset", "model", "seed",
    "epoch",               # 1-indexed epoch number
    "val_loss_raw",
    "val_loss_smooth",
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


def append_curve(rows: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = CURVE_CSV.exists()
    with open(CURVE_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_COLS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in CURVE_COLS})


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
# EXPERT TRAINING (manual loop — records weights + val loss per epoch)
# =============================================================================

def train_expert_and_record(model_name, dataset_name, data, cfg, device, seed):
    """
    Train expert for cfg['expert_epochs'] epochs.

    Returns:
        trajectory  — list of {'step': int, 'weights': dict} from SimpleRecorder
        val_losses  — list of float, val MSE recorded at the end of each epoch
                       (same length as trajectory[1:], i.e. one per epoch)
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

    train_loader = MiniBatchLoader(data["train_data"], batch_size=cfg["batch_size"])
    val_loader   = TorchDataLoader(
        data["val_data"], batch_size=cfg["eval_batch_size"], shuffle=False
    )

    recorder.on_train_begin(model)

    val_losses = []
    for epoch in range(cfg["expert_epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        recorder.on_epoch_end(model, epoch, train_loss)

        val_loss = trainer.eval_epoch(val_loader)
        val_losses.append(val_loss)

    return recorder.get_trajectory(), val_losses


# =============================================================================
# PLATEAU DETECTION
# =============================================================================

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


def detect_boundary_valloss(
    epochs:           list,
    val_losses:       list,
    smoothing_window: int   = 5,
    patience:         int   = 5,
    min_delta_frac:   float = 0.01,
) -> dict:
    """
    Detect T+ as the epoch of the best (lowest) smoothed val loss before a
    sustained plateau begins.

    An epoch is "improving" if its smoothed val loss is at least
    `min_delta_frac` (relative) below the best-so-far value. T+ is the
    epoch index of the best-so-far value at the moment `patience`
    consecutive non-improving epochs have been observed.

    Returns:
        dict with keys: boundary_epoch, boundary_idx, best_val_loss,
                        final_val_loss, val_loss_smooth
    """
    val_smooth = smooth(val_losses, smoothing_window)

    best_val      = val_smooth[0]
    best_idx      = 0
    no_improve    = 0
    boundary_idx  = None

    for i in range(1, len(val_smooth)):
        if val_smooth[i] < best_val * (1.0 - min_delta_frac):
            best_val   = val_smooth[i]
            best_idx   = i
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                boundary_idx = best_idx
                break

    boundary_epoch = epochs[boundary_idx] if boundary_idx is not None else None

    return {
        "boundary_epoch": boundary_epoch,
        "boundary_idx":   boundary_idx,
        "best_val_loss":  float(best_val),
        "final_val_loss": float(val_smooth[-1]),
        "val_loss_smooth": val_smooth,
    }


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(device, expert_cfg, plateau_cfg):

    smoothing_window = plateau_cfg["smoothing_window"]
    patience          = plateau_cfg["patience"]
    min_delta_frac    = plateau_cfg["min_delta_frac"]

    print("=" * 72)
    print("PHASE BOUNDARY DETECTION — Method B: Validation Loss Plateau")
    print("=" * 72)
    print(f"  smoothing_window={smoothing_window}  patience={patience}  "
          f"min_delta_frac={min_delta_frac}")
    print(f"  Rule: T+ = epoch of best smoothed val loss before "
          f"{patience} consecutive epochs with < {min_delta_frac*100:.1f}% "
          f"relative improvement")
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
                # Train expert and record trajectory + val loss curve
                # ---------------------------------------------------------
                try:
                    trajectory, val_losses = train_expert_and_record(
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
                # epochs are 1-indexed (epoch i finishes -> recorded as step i)
                epochs = list(range(1, len(val_losses) + 1))

                # ---------------------------------------------------------
                # Detect boundary
                # ---------------------------------------------------------
                result = detect_boundary_valloss(
                    epochs, val_losses,
                    smoothing_window=smoothing_window,
                    patience=patience,
                    min_delta_frac=min_delta_frac,
                )

                boundary_epoch = result["boundary_epoch"]
                boundary_idx   = result["boundary_idx"]
                val_smooth     = result["val_loss_smooth"]

                # ---------------------------------------------------------
                # Print summary
                # ---------------------------------------------------------
                if boundary_epoch is not None:
                    print(f"  T+ detected at epoch {boundary_epoch}  "
                          f"(val_loss_smooth={val_smooth[boundary_idx]:.6f})")
                else:
                    print(f"  T+ NOT detected — val loss kept improving "
                          f"throughout training.")

                print(f"  best_val_loss  = {result['best_val_loss']:.6f}")
                print(f"  final_val_loss = {result['final_val_loss']:.6f}")

                print(f"  Val loss curve (epoch : smoothed):")
                for i, (ep, vl) in enumerate(zip(epochs, val_smooth)):
                    marker = " <-- T+" if i == boundary_idx else ""
                    if i < smoothing_window or i == boundary_idx or i % 10 == 0:
                        print(f"    epoch {ep:>3}: {vl:.6f}{marker}")

                # ---------------------------------------------------------
                # Write CSVs
                # ---------------------------------------------------------
                append_boundary({
                    "dataset":          dataset_name,
                    "model":            model_name,
                    "seed":             seed,
                    "boundary_epoch":   boundary_epoch if boundary_epoch is not None else "",
                    "n_checkpoints":    n_checkpoints,
                    "best_val_loss":    round(result["best_val_loss"], 8),
                    "final_val_loss":   round(result["final_val_loss"], 8),
                    "smoothing_window": smoothing_window,
                    "patience":         patience,
                    "min_delta_frac":   min_delta_frac,
                    "notes":            "" if boundary_epoch is not None
                                        else "no_boundary_found",
                })

                curve_rows = []
                for i, (ep, vl_raw, vl_sm) in enumerate(
                    zip(epochs, val_losses, val_smooth)
                ):
                    curve_rows.append({
                        "dataset":        dataset_name,
                        "model":          model_name,
                        "seed":           seed,
                        "epoch":          ep,
                        "val_loss_raw":   round(float(vl_raw), 8),
                        "val_loss_smooth": round(float(vl_sm), 8),
                        "is_boundary":    (i == boundary_idx),
                    })
                append_curve(curve_rows)

    # Summary table
    print("\n\n" + "=" * 72)
    print("DETECTED PHASE BOUNDARIES SUMMARY (Method B — Val Loss Plateau)")
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
    run(device, EXPERT_CONFIG, PLATEAU_CONFIG)
    print(f"\nBoundaries  : {BOUNDARY_CSV}")
    print(f"Curves      : {CURVE_CSV}")


if __name__ == "__main__":
    main()
