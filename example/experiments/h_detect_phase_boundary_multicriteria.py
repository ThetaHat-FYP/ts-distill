"""
Phase Boundary Detection — Method C: Change-Point Detection on Parameter Movement
===================================================================================

Alternative to:
  - Method A (h_detect_phase_boundary.py)         — val-loss velocity threshold
  - Method B (h_detect_phase_boundary_valloss.py) — val-loss plateau (best-so-far
                                                     + patience)

Methods A and B both rely on a hand-picked threshold (e.g. "velocity has
dropped below 10% of its initial value") applied to a val-loss-derived
signal. In practice that threshold has to be re-tuned per dataset/model —
on the real weather runs it never triggered at all for some architectures.

Method C instead asks a purely statistical question of a single signal —
the expert's parameter (weight) movement between consecutive epochs,
||theta_t - theta_(t-1)||_2 (same squared-distance formula as
ts_distill.trajectory.matcher.mse_matcher.MSEMatcher) — and finds where its
*decay rate* changes, using change-point detection (PELT or Binary
Segmentation) with a piecewise-LINEAR-TREND cost on the LOG of the smoothed
curve. No manually chosen threshold is involved: the penalty that governs
how many change points are accepted is derived automatically from each
run's own signal variance and length via the standard BIC formula
(penalty = penalty_scale * (n_params + 1) * sigma^2 * log(n), n_params=2 for
a fitted line), so the same script works unmodified across datasets and
architectures.

Why log + linear-trend, not raw + mean-shift
---------------------------------------------
An earlier version of this detector used a piecewise-CONSTANT-MEAN cost on
the RAW parameter-movement curve, and picked the change point with the
single largest mean drop. That consistently fired around epoch 3-4 instead
of the real late-training knee, for two compounding reasons:
  - Parameter movement decays roughly exponentially. On the raw scale, the
    first few epochs (random-init transient) are or­ders of magnitude
    larger than the rest of the curve, so they dominate both the L2 cost
    and the variance term in the BIC penalty. A mean-shift model "spends"
    its one big level-shift there and has nothing left to justify a further
    split deeper in the (numerically tiny, but proportionally still
    informative) tail.
  - Picking the "biggest drop" is the wrong question for a monotonically
    decaying signal in the first place: any two-segment split of a smoothly
    decaying curve reduces mean-shift cost somewhat, so the method could
    report *a* boundary even when there is no real regime change — and the
    largest such artificial drop is almost always at the very start.
This version fixes both issues:
  - Log-transforming the curve turns exponential decay into an
    (approximately) LINEAR trend, so a genuine multiplicative slowdown late
    in training is on the same footing as the early transient instead of
    being swamped by it.
  - Fitting piecewise LINES (not piecewise constants) means a change point
    is only justified when the *decay rate* (slope in log-space) actually
    changes. A pure single-rate exponential decay needs zero extra
    segments — a straight line already fits it perfectly in log-space — so
    the method no longer manufactures a spurious boundary out of ordinary
    smooth convergence. It only reports a boundary when the curve
    genuinely bends: steep negative slope (rapid learning) flattening into
    a near-zero slope (stable convergence).
  - T+ is chosen as the LAST detected change point rather than the largest
    drop: everything before it (including any early transient split) is
    treated as still part of the "fast learning" super-regime, and the
    segment from the last change point to the end of training is, by
    construction, the single linear regime that best explains the tail —
    i.e. the phase that has actually settled into stable convergence.

Algorithm
---------
For each (dataset, model):
  1. Train the expert for expert_epochs, recording weight checkpoints
     (SimpleRecorder) and train/val loss (kept only for the curve CSV /
     plotting context — they do not drive detection).
  2. Compute raw parameter movement per epoch: ||theta_t - theta_(t-1)||_2.
  3. Log-transform it (log(x + eps)) and smooth with a rolling average of
     window `smoothing_window` — this is the detection signal. (The
     RAW-scale smoothed curve is kept separately for the CSV / plotting and
     for reporting mean_before/mean_after in interpretable units.)
  4. Run change-point detection (`algorithm`: "pelt" or "binseg") on the
     log-smoothed curve with a piecewise-linear-trend cost and a
     BIC-derived penalty. This partitions the curve into segments, each
     well-approximated by its own straight line (i.e. its own decay rate).
  5. T+ = the LAST detected change point, i.e. the epoch where the final,
     stable (near-zero-slope) linear regime begins. If no change point is
     found, the curve never showed a distinct rate change (e.g. pure smooth
     decay throughout) and no boundary is reported.

Outputs
-------
  results/weather_phase_boundaries_multicriteria.csv       — one row per (dataset, model, seed)
  results/weather_phase_boundary_multicriteria_curves.csv  — per-epoch signal values (for plotting)

Run
---
  python -m example.experiments.h_detect_phase_boundary_multicriteria
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

ACTIVE_DATASETS = ["ETTh1","ETTh2","ETTm1","ETTm2","weather"]
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

# T+ = the LAST change point in decay RATE (slope), detected on the log of
# the smoothed parameter-movement curve via a piecewise-linear-trend cost —
# i.e. the start of the final, stable (near-zero-slope) regime.
CHANGEPOINT_CONFIG = {
    # Rolling average window applied to the raw parameter-movement signal
    # before change-point detection.
    "smoothing_window": 5,

    # "pelt" (exact optimal partitioning) or "binseg" (greedy binary
    # segmentation). Both use the same BIC-derived penalty below.
    "algorithm": "pelt",

    # Minimum number of epochs per segment. Each segment now fits a line
    # (slope + intercept), which needs a few more points than a mean
    # estimate to be numerically stable — this is a numerical-stability
    # floor, not a tuned threshold, and does not need to change across
    # datasets/architectures.
    "min_segment_size": 4,

    # Multiplier on the BIC penalty (penalty = penalty_scale * (n_params+1)
    # * sigma^2 * log(n), computed per-run from that run's own signal).
    # 1.0 = standard BIC; left as a knob for sensitivity checks, not
    # per-dataset tuning.
    "penalty_scale": 1.0,
}


# =============================================================================
# CSV PATHS
# =============================================================================

RESULTS_DIR  = Path(__file__).parent / "results"
BOUNDARY_CSV = RESULTS_DIR / "new_phase_boundaries_multicriteria.csv"
CURVE_CSV    = RESULTS_DIR / "new_phase_boundary_multicriteria_curves.csv"

BOUNDARY_COLS = [
    "dataset", "model", "seed",
    "boundary_epoch",              # detected T+ (None if no change point found)
    "n_checkpoints",                # total epochs trained
    "algorithm",                    # "pelt" or "binseg"
    "penalty",                      # BIC-derived penalty used for this run
    "n_change_points",              # total change points detected in the curve
    "mean_before",                  # segment mean immediately before T+
    "mean_after",                   # segment mean immediately after T+
    "drop_ratio",                   # mean_before / mean_after at T+
    "smoothing_window",
    "min_segment_size",
    "penalty_scale",
    "notes",
]

CURVE_COLS = [
    "dataset", "model", "seed",
    "epoch",                        # 1-indexed epoch number
    "train_loss",
    "val_loss",
    "param_movement_raw", "param_movement_smooth",
    "segment_id",                   # index of the piecewise segment this epoch belongs to
    "is_boundary",                  # True for the detected T+ epoch
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
# EXPERT TRAINING (manual loop — records weights + train loss + val loss)
# =============================================================================

def train_expert_and_record(model_name, dataset_name, data, cfg, device, seed):
    """
    Train expert for cfg['expert_epochs'] epochs.

    Returns:
        trajectory   — list of {'step': int, 'weights': dict} from SimpleRecorder
                        (trajectory[0] is the random-init snapshot, trajectory[i]
                        is the snapshot after epoch i)
        train_losses — list of float, mean training loss per epoch
        val_losses   — list of float, val MSE recorded at the end of each epoch
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

    train_losses = []
    val_losses   = []
    for epoch in range(cfg["expert_epochs"]):
        train_loss = trainer.train_epoch(train_loader)
        recorder.on_epoch_end(model, epoch, train_loss)
        train_losses.append(train_loss)

        val_loss = trainer.eval_epoch(val_loader)
        val_losses.append(val_loss)

    return recorder.get_trajectory(), train_losses, val_losses


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================

def compute_param_movement(trajectory: list) -> tuple:
    """
    L2 distance between consecutive weight checkpoints for every epoch:
    ||theta_epoch - theta_(epoch-1)||_2. trajectory[i] holds the weights
    after epoch i (trajectory[0] is the random-init snapshot), so this is
    well-defined for epoch = 1 .. len(trajectory) - 1.

    Uses the same sum-of-squared-differences formula as
    ts_distill.trajectory.matcher.mse_matcher.MSEMatcher.

    Returns:
        epochs — 1-indexed epoch numbers, [1, 2, ..., N]
        raw    — L2 distance per epoch, aligned with `epochs`
    """
    n = len(trajectory) - 1
    epochs = list(range(1, n + 1))
    raw = []
    for ep in epochs:
        w_now  = trajectory[ep]["weights"]
        w_prev = trajectory[ep - 1]["weights"]
        dist_sq = sum(
            torch.sum((w_now[k] - w_prev[k]) ** 2).item() for k in w_now
        )
        raw.append(dist_sq ** 0.5)
    return epochs, raw


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


# =============================================================================
# CHANGE-POINT DETECTION (PELT / Binary Segmentation, piecewise-LINEAR cost)
# =============================================================================
#
# `ruptures` cannot be installed on this machine (no MSVC build tools for its
# C extension), so both algorithms are implemented directly here. Both use a
# piecewise-linear-TREND segment cost (residual sum of squares of the best-fit
# line over the segment):
#
#     cost(a, b) = sum((y[a:b] - fitted_line(a:b)) ** 2)
#
# rather than a piecewise-constant-mean cost. This is what lets the detector
# find where the DECAY RATE changes (fast-learning slope -> stable-convergence
# slope) instead of where the raw LEVEL jumps (which, on an exponentially
# decaying curve, is almost always the very first few epochs). See the module
# docstring for the full reasoning. Segment cost is computed in O(1) via
# prefix sums (closed-form simple linear regression), and both algorithms use
# the same BIC-derived penalty (see bic_penalty below) — no hand-tuned
# threshold.

def _sum_int(n: int) -> float:
    """sum_{i=0}^{n-1} i, closed form."""
    return n * (n - 1) / 2.0


def _sum_sq(n: int) -> float:
    """sum_{i=0}^{n-1} i**2, closed form."""
    return n * (n - 1) * (2 * n - 1) / 6.0


def _segment_cost_linear(
    prefix_y: np.ndarray, prefix_yy: np.ndarray, prefix_xy: np.ndarray, a: int, b: int
) -> float:
    """
    Residual sum of squares of the best-fit line (ordinary least squares,
    x = position index, y = signal value) over segment [a, b). Uses the
    standard centred-sums identity: RSS = Syy_c - Sxy_c^2 / Sxx_c.
    """
    n_seg = b - a
    if n_seg < 2:
        return 0.0

    Sy  = prefix_y[b]  - prefix_y[a]
    Syy = prefix_yy[b] - prefix_yy[a]
    Sxy = prefix_xy[b] - prefix_xy[a]
    Sx  = _sum_int(b) - _sum_int(a)
    Sxx = _sum_sq(b)  - _sum_sq(a)

    Sxx_c = Sxx - (Sx ** 2) / n_seg
    Syy_c = Syy - (Sy ** 2) / n_seg
    if Sxx_c <= 1e-12:
        # Degenerate (shouldn't happen for n_seg >= 2): fall back to
        # variance-only cost.
        return float(max(Syy_c, 0.0))

    Sxy_c = Sxy - (Sx * Sy) / n_seg
    rss = Syy_c - (Sxy_c ** 2) / Sxx_c
    return float(max(rss, 0.0))


def bic_penalty(signal: np.ndarray, penalty_scale: float = 1.0, n_params: int = 2) -> float:
    """
    Standard BIC penalty for adding a segment: (n_params + 1) * sigma^2 *
    log(n), where sigma^2 is the signal's own variance and n its length.
    n_params=2 for a linear-trend model (slope + intercept). Computed fresh
    from each run's own signal, so it automatically adapts across
    datasets/architectures instead of requiring a per-dataset threshold.
    """
    n = len(signal)
    if n < 2:
        return 0.0
    sigma2 = float(np.var(signal))
    return penalty_scale * (n_params + 1) * sigma2 * np.log(n)


def pelt_changepoints(signal: np.ndarray, penalty: float, min_size: int = 4) -> list:
    """
    Exact optimal partitioning under a piecewise-linear-trend cost (PELT
    recursion, Killick et al. 2012, generalised from mean-shift to
    trend-shift). Expert-training curves here are short (<= ~100 epochs), so
    this runs the dynamic program without PELT's pruning step — the result
    is mathematically identical to pruned PELT, just without the O(n) speed
    optimization that only matters at much larger n.

    Returns a sorted list of change-point indices (each in [min_size, n)),
    where index t means "segment boundary between position t-1 and t".
    """
    n = len(signal)
    idx        = np.arange(n, dtype=np.float64)
    prefix_y   = np.concatenate(([0.0], np.cumsum(signal)))
    prefix_yy  = np.concatenate(([0.0], np.cumsum(np.square(signal))))
    prefix_xy  = np.concatenate(([0.0], np.cumsum(idx * signal)))

    F    = np.full(n + 1, np.inf)
    F[0] = -penalty
    back = np.zeros(n + 1, dtype=int)

    for t in range(1, n + 1):
        for s in range(0, t - min_size + 1):
            if not np.isfinite(F[s]):
                continue
            val = F[s] + _segment_cost_linear(prefix_y, prefix_yy, prefix_xy, s, t) + penalty
            if val < F[t]:
                F[t] = val
                back[t] = s

    cps = []
    t = n
    while t > 0:
        s = back[t]
        if s > 0:
            cps.append(s)
        t = s
    cps.reverse()
    return cps


def binseg_changepoints(signal: np.ndarray, penalty: float, min_size: int = 4) -> list:
    """
    Greedy binary segmentation with a piecewise-linear-trend cost: repeatedly
    split the segment at the point that most reduces total residual, stopping
    when the best available split's gain no longer exceeds the BIC penalty.
    """
    n = len(signal)
    idx        = np.arange(n, dtype=np.float64)
    prefix_y   = np.concatenate(([0.0], np.cumsum(signal)))
    prefix_yy  = np.concatenate(([0.0], np.cumsum(np.square(signal))))
    prefix_xy  = np.concatenate(([0.0], np.cumsum(idx * signal)))

    def cost(a, b):
        return _segment_cost_linear(prefix_y, prefix_yy, prefix_xy, a, b)

    cps   = []
    stack = [(0, n)]
    while stack:
        a, b = stack.pop()
        if b - a < 2 * min_size:
            continue
        base = cost(a, b)
        best_gain = 0.0
        best_t    = None
        for t in range(a + min_size, b - min_size + 1):
            gain = base - (cost(a, t) + cost(t, b))
            if gain > best_gain:
                best_gain = gain
                best_t    = t
        if best_t is not None and best_gain > penalty:
            cps.append(best_t)
            stack.append((a, best_t))
            stack.append((best_t, b))
    cps.sort()
    return cps


def select_transition_breakpoint(raw_smoothed: np.ndarray, change_points: list) -> dict:
    """
    T+ = the LAST detected change point, i.e. the start of the final linear
    regime that the trend model needed no further splits to explain. Any
    earlier change points (e.g. from the initial random-init transient) are
    treated as still part of the "fast learning" super-regime; only the
    final regime — the one that persists all the way to the end of
    training — counts as "settled into stable convergence".

    mean_before / mean_after are reported in the ORIGINAL (not log-space)
    parameter-movement units, computed on `raw_smoothed`, purely for
    interpretability in the output CSV — they do not drive the selection.

    Returns dict with keys: idx (change-point index, or None), mean_before,
    mean_after, drop_ratio.
    """
    if not change_points:
        return {"idx": None, "mean_before": None, "mean_after": None, "drop_ratio": None}

    boundary   = change_points[-1]
    prev_bound = change_points[-2] if len(change_points) >= 2 else 0

    seg_before = raw_smoothed[prev_bound:boundary]
    seg_after  = raw_smoothed[boundary:]
    mb = float(seg_before.mean()) if len(seg_before) else float("nan")
    ma = float(seg_after.mean())  if len(seg_after)  else float("nan")
    ratio = (mb / ma) if ma > 1e-30 else None

    return {"idx": boundary, "mean_before": mb, "mean_after": ma, "drop_ratio": ratio}


def detect_boundary_changepoint(
    trajectory:        list,
    smoothing_window:  int   = 5,
    algorithm:         str   = "pelt",
    min_segment_size:  int   = 4,
    penalty_scale:     float = 1.0,
) -> dict:
    """
    Detect T+ via change-point detection on the LOG of the smoothed
    parameter-movement curve, using a piecewise-linear-trend cost (so the
    detector responds to changes in decay RATE, not raw level — see module
    docstring). boundary_epoch is the first epoch of the final, stable
    linear regime — the last change point found — consistent with Methods
    A/B's "T+ = first epoch of the slow phase" convention.
    """
    epochs, raw = compute_param_movement(trajectory)
    raw_arr = np.asarray(raw, dtype=np.float64)

    # RAW-scale smoothed curve: kept for the curve CSV / plotting and for
    # reporting mean_before/mean_after in interpretable (non-log) units.
    param_movement_smooth = smooth(raw, smoothing_window)

    # LOG-scale smoothed curve: the actual detection signal. Log-transforming
    # first (before smoothing) turns the roughly-exponential decay into a
    # roughly-linear one and stops the huge early-epoch magnitudes from
    # dominating the rolling average of later, numerically smaller epochs.
    eps = 1e-8
    log_smooth = smooth(list(np.log(raw_arr + eps)), smoothing_window)

    penalty = bic_penalty(log_smooth, penalty_scale, n_params=2)

    if algorithm == "pelt":
        change_points = pelt_changepoints(log_smooth, penalty, min_size=min_segment_size)
    elif algorithm == "binseg":
        change_points = binseg_changepoints(log_smooth, penalty, min_size=min_segment_size)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r} (expected 'pelt' or 'binseg')")

    selection = select_transition_breakpoint(param_movement_smooth, change_points)
    cp_idx = selection["idx"]

    if cp_idx is None:
        boundary_epoch, boundary_idx = None, None
    else:
        boundary_idx   = cp_idx
        boundary_epoch = epochs[cp_idx]

    # Segment id per epoch, for the curve CSV / plotting.
    bounds = [0] + list(change_points) + [len(param_movement_smooth)]
    segment_id = np.zeros(len(param_movement_smooth), dtype=int)
    for seg_i in range(len(bounds) - 1):
        segment_id[bounds[seg_i]:bounds[seg_i + 1]] = seg_i

    return {
        "epochs":                epochs,
        "param_movement_raw":    raw,
        "param_movement_smooth": param_movement_smooth,
        "boundary_epoch":        boundary_epoch,
        "boundary_idx":          boundary_idx,
        "algorithm":             algorithm,
        "penalty":               penalty,
        "change_points":         change_points,
        "mean_before":           selection["mean_before"],
        "mean_after":            selection["mean_after"],
        "drop_ratio":            selection["drop_ratio"],
        "segment_id":            segment_id,
    }


# =============================================================================
# MAIN LOOP
# =============================================================================

def run(device, expert_cfg, cp_cfg):

    smoothing_window  = cp_cfg["smoothing_window"]
    algorithm          = cp_cfg["algorithm"]
    min_segment_size   = cp_cfg["min_segment_size"]
    penalty_scale       = cp_cfg["penalty_scale"]

    print("=" * 72)
    print("PHASE BOUNDARY DETECTION — Method C: Change-Point Detection")
    print("=" * 72)
    print(f"  algorithm={algorithm}  smoothing_window={smoothing_window}  "
          f"min_segment_size={min_segment_size}  penalty_scale={penalty_scale}")
    print(f"  Rule: T+ = last decay-rate change point on log(smoothed param movement) "
          f"via piecewise-linear-trend cost (BIC-derived penalty, no manual threshold)")
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
                # Train expert and record trajectory + train/val loss curves
                # ---------------------------------------------------------
                try:
                    trajectory, train_losses, val_losses = train_expert_and_record(
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

                # ---------------------------------------------------------
                # Detect boundary
                # ---------------------------------------------------------
                result = detect_boundary_changepoint(
                    trajectory,
                    smoothing_window=smoothing_window,
                    algorithm=algorithm,
                    min_segment_size=min_segment_size,
                    penalty_scale=penalty_scale,
                )

                epochs          = result["epochs"]
                boundary_epoch  = result["boundary_epoch"]
                boundary_idx    = result["boundary_idx"]
                segment_id      = result["segment_id"]

                # ---------------------------------------------------------
                # Print summary
                # ---------------------------------------------------------
                if boundary_epoch is not None:
                    print(f"  T+ detected at epoch {boundary_epoch}  "
                          f"(mean_before={result['mean_before']:.6e}, "
                          f"mean_after={result['mean_after']:.6e}, "
                          f"drop_ratio={result['drop_ratio']:.2f}x)")
                else:
                    print(f"  T+ NOT detected — no change point showed a mean drop "
                          f"({len(result['change_points'])} change point(s) found total).")

                print(f"  penalty (BIC)  = {result['penalty']:.6e}")
                print(f"  change points  = {result['change_points']}")

                print(f"  Parameter movement (epoch : raw / smooth : segment):")
                for i, ep in enumerate(epochs):
                    marker = " <-- T+" if i == boundary_idx else ""
                    if i == 0 or i == boundary_idx or i % 10 == 0:
                        print(f"    epoch {ep:>3}: "
                              f"{result['param_movement_raw'][i]:.6e} / "
                              f"{result['param_movement_smooth'][i]:.6e} : "
                              f"seg{segment_id[i]}{marker}")

                # ---------------------------------------------------------
                # Write CSVs
                # ---------------------------------------------------------
                append_boundary({
                    "dataset":            dataset_name,
                    "model":              model_name,
                    "seed":               seed,
                    "boundary_epoch":     boundary_epoch if boundary_epoch is not None else "",
                    "n_checkpoints":      n_checkpoints,
                    "algorithm":          algorithm,
                    "penalty":            round(result["penalty"], 8),
                    "n_change_points":    len(result["change_points"]),
                    "mean_before":        round(result["mean_before"], 8) if result["mean_before"] is not None else "",
                    "mean_after":         round(result["mean_after"], 8) if result["mean_after"] is not None else "",
                    "drop_ratio":         round(result["drop_ratio"], 6) if result["drop_ratio"] is not None else "",
                    "smoothing_window":   smoothing_window,
                    "min_segment_size":   min_segment_size,
                    "penalty_scale":      penalty_scale,
                    "notes":              "" if boundary_epoch is not None
                                          else "no_transition_breakpoint_found",
                })

                curve_rows = []
                for i, ep in enumerate(epochs):
                    curve_rows.append({
                        "dataset":               dataset_name,
                        "model":                 model_name,
                        "seed":                  seed,
                        "epoch":                 ep,
                        "train_loss":            round(float(train_losses[ep - 1]), 8),
                        "val_loss":              round(float(val_losses[ep - 1]), 8),
                        "param_movement_raw":    round(float(result["param_movement_raw"][i]), 8),
                        "param_movement_smooth": round(float(result["param_movement_smooth"][i]), 8),
                        "segment_id":            int(segment_id[i]),
                        "is_boundary":           (i == boundary_idx),
                    })
                append_curve(curve_rows)

    # Summary table
    print("\n\n" + "=" * 72)
    print("DETECTED PHASE BOUNDARIES SUMMARY (Method C — Change-Point Detection)")
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
    run(device, EXPERT_CONFIG, CHANGEPOINT_CONFIG)
    print(f"\nBoundaries  : {BOUNDARY_CSV}")
    print(f"Curves      : {CURVE_CSV}")


if __name__ == "__main__":
    main()
