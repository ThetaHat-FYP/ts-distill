"""
H2 Phase Isolation Experiment
==============================

Hypothesis H2:
  The cross-architecture bias grows as more late-phase checkpoints are used.
  If distillation is restricted to EARLY checkpoints (before the detected
  overfitting boundary), cross-architecture MSE will be LOWER than when
  restricted to LATE checkpoints (after the boundary).

Test cases (from research guide):
  - ETTh1  x LSTM expert  (boundary = epoch 48)
  - ETTm1  x LSTM expert  (boundary = epoch 48)
  - ETTh1  x CNN  expert  (boundary = epoch 58)

Three distillation conditions per test case:
  E (early_only) : checkpoint pairs from epochs [0, boundary-1]
  L (late_only)  : checkpoint pairs from epochs [boundary+1, 80]
  F (full_mtt)   : all checkpoint pairs (replication of H1 baseline)

Key question: does Condition L produce worse off-diagonal mse_ratio than E?

Output: results/h2_phase_isolation.csv
  One row written immediately per (test_case, condition, student).

NO changes to the original MTT algorithm.  A local FilteredRecorder wraps
the SimpleRecorder trajectory and exposes only the desired epoch slice.

Run:
  python -m example.experiments.h2_phase_isolation
"""

import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader

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
# FilteredRecorder  —  wraps a SimpleRecorder trajectory slice
# =============================================================================

class FilteredRecorder:
    """
    Presents a filtered view of a SimpleRecorder's trajectory to MTTDistiller.

    Only checkpoints whose 'step' value falls within [min_step, max_step] are
    visible.  MTTDistiller calls sample_checkpoint_pair() on this object, so
    the distillation is restricted to the chosen epoch range without any
    modification to the original MTT code.

    Args:
        full_trajectory : recorder.trajectory from a completed SimpleRecorder.
        min_step        : include checkpoints with step >= min_step (None = no lower bound).
        max_step        : include checkpoints with step <= max_step (None = no upper bound).
    """

    def __init__(
        self,
        full_trajectory: List[Dict],
        min_step: Optional[int] = None,
        max_step: Optional[int] = None,
    ):
        self.trajectory = [
            ckpt for ckpt in full_trajectory
            if (min_step is None or ckpt["step"] >= min_step)
            and (max_step is None or ckpt["step"] <= max_step)
        ]
        self.min_step = min_step
        self.max_step = max_step

    def __len__(self) -> int:
        return len(self.trajectory)

    def sample_checkpoint_pair(self, step_gap: int = 10) -> Tuple[Dict, Dict]:
        if len(self.trajectory) < 2:
            raise ValueError(
                f"FilteredRecorder has only {len(self.trajectory)} checkpoint(s) "
                f"in range [{self.min_step}, {self.max_step}] — need >= 2 for MTT."
            )
        actual_gap = min(step_gap, len(self.trajectory) - 1)
        start_idx  = random.randint(0, len(self.trajectory) - actual_gap - 1)
        return self.trajectory[start_idx], self.trajectory[start_idx + actual_gap]


# =============================================================================
# CONFIG
# =============================================================================

ACTIVE_MODELS = ["DLinear", "LSTM", "MLP", "CNN"]

MODEL_CONFIGS = {
    "DLinear": {"individual": False},
    "LSTM":    {"hidden_dim": 16, "num_layer": 1},
    "MLP":     {},
    "CNN":     {},
}

# Phase boundaries detected in the interim report.
PHASE_BOUNDARIES = {
    "LSTM":    48,
    "CNN":     58,
    "MLP":     30,
    "DLinear": None,   # No classical boundary found
}

# H2 test cases as specified by the research guide.
# Each entry defines one (dataset, expert) pair to run all 3 conditions on.
H2_TEST_CASES = [
    {"dataset": "ETTh1", "expert": "LSTM", "boundary": PHASE_BOUNDARIES["LSTM"]},
    {"dataset": "ETTm1", "expert": "LSTM", "boundary": PHASE_BOUNDARIES["LSTM"]},
    {"dataset": "ETTh1", "expert": "CNN",  "boundary": PHASE_BOUNDARIES["CNN"]},
]

DATASET_CONFIGS = {
    "ETTh1": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTh1.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12*30*24 - 96, 12*30*24 + 4*30*24 - 96],
        "border2s":   [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24],
        "in_features": 7,
    },
    "ETTh2": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTh2.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12*30*24 - 96, 12*30*24 + 4*30*24 - 96],
        "border2s":   [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24],
        "in_features": 7,
    },
    "ETTm1": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTm1.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12*30*96 - 96, 12*30*96 + 4*30*96 - 96],
        "border2s":   [12*30*96, 12*30*96 + 4*30*96, 12*30*96 + 8*30*96],
        "in_features": 7,
    },
    "ETTm2": {
        "csv_path":   r"C:\fyp\ts-distill\example/ETTm2.csv",
        "split_mode": "benchmark_borders",
        "border1s":   [0, 12*30*96 - 96, 12*30*96 + 4*30*96 - 96],
        "border2s":   [12*30*96, 12*30*96 + 4*30*96, 12*30*96 + 8*30*96],
        "in_features": 7,
    },
}

DISTILL_CONFIG = {
    "seq_len":  96,
    "pred_len": 96,

    # Expert training
    "expert_epochs":   80,
    "expert_lr":       0.01,
    "expert_momentum": 0.9,

    # MTT distillation
    "n_distill_steps":       300,
    "n_synthetic":           384,
    "synthetic_lr":          5.0,
    "student_lr":            0.01,
    "student_steps":         20,
    "snapshot_student_steps": 50,
    "trajectory_gap":        5,
    "batch_size":            64,

    # Student evaluation
    "eval_max_epochs":    300,
    "eval_lr":            0.001,
    "eval_batch_size":    32,
    "early_stop_patience": 15,
}

SEED = 42


# =============================================================================
# CSV
# =============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "h2_phase_isolation.csv"

CSV_COLUMNS = [
    "experiment", "dataset", "expert_model", "student_model", "seed",
    "method",             # early_only / late_only / full_mtt
    "phase_boundary",     # epoch used as the E/L split
    "n_ckpts_condition",  # checkpoints available in this condition
    "real_mse", "transfer_mse", "mse_ratio",
    "n_pairs_available",
    "notes",
]


def _append_csv(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})
    print(
        f"  [CSV] {row['dataset']} | {row['expert_model']}->{row['student_model']}"
        f" | {row['method']} | ratio={row.get('mse_ratio', 'N/A')}"
    )


def _row_exists(dataset: str, expert: str, student: str, method: str) -> bool:
    if not CSV_PATH.exists():
        return False
    try:
        raw = pd.read_csv(CSV_PATH, header=None)
        if str(raw.iloc[0, 0]) == "experiment":
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.read_csv(CSV_PATH, header=None, names=CSV_COLUMNS[:len(raw.columns)])
        mask = (
            (df["dataset"]       == dataset) &
            (df["expert_model"]  == expert)  &
            (df["student_model"] == student) &
            (df["method"]        == method)  &
            (df["transfer_mse"].notna()) &
            (df["transfer_mse"]  != "")
        )
        return bool(mask.any())
    except Exception:
        return False


# =============================================================================
# HELPERS
# =============================================================================

def _make_model(name: str, sl: int, pl: int, nf: int) -> torch.nn.Module:
    return create_model(
        model_type=name, seq_len=sl, pred_len=pl,
        in_features=nf, model_kwargs=MODEL_CONFIGS[name],
    )


def _load_dataset(dataset_name: str, cfg: dict) -> dict:
    ds  = DATASET_CONFIGS[dataset_name]
    sl  = cfg["seq_len"]
    pl  = cfg["pred_len"]
    nf  = ds["in_features"]
    ws  = sl + pl

    df_raw = pd.read_csv(ds["csv_path"])
    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values=values, window_size=ws, seq_len=sl, dataset_cfg=ds,
    )

    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    scaled = scaler.transform(values)

    train_data = torch.tensor(make_windows(scaled[train_start:train_end], ws), dtype=torch.float32)
    val_data   = torch.tensor(make_windows(scaled[val_start:val_end],     ws), dtype=torch.float32)
    test_data  = torch.tensor(make_windows(scaled[test_start:test_end],   ws), dtype=torch.float32)
    raw_train  = torch.tensor(scaled[train_start:train_end],                   dtype=torch.float32)

    return {
        "train_data": train_data, "val_data": val_data,
        "test_data":  test_data,  "raw_train": raw_train,
        "seq_len": sl, "pred_len": pl, "in_features": nf, "window_size": ws,
    }


def _train_expert_and_record(
    expert_name: str, data: dict, cfg: dict, device: str, seed: int,
) -> SimpleRecorder:
    """Train expert on real data and return its full trajectory recorder."""
    sl, pl, nf = data["seq_len"], data["pred_len"], data["in_features"]
    torch.manual_seed(seed)

    model   = _make_model(expert_name, sl, pl, nf).to(device)
    recorder = SimpleRecorder(record_every=1)

    trainer = Trainer(
        model     = model,
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg["expert_lr"], momentum=cfg["expert_momentum"]
        ),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = sl,
    )
    loader = MiniBatchLoader(data["train_data"], batch_size=cfg["batch_size"])
    trainer.fit(
        dataloader = loader,
        epochs     = cfg["expert_epochs"],
        callbacks  = [recorder, SimpleCallback()],
    )
    return recorder


def _distill_with_recorder(
    expert_name: str,
    filtered_recorder,          # FilteredRecorder or SimpleRecorder
    data: dict,
    cfg: dict,
    device: str,
    seed: int,
) -> Tuple[torch.Tensor, int]:
    """Run MTT distillation using the given recorder (full or filtered)."""
    sl, pl, nf = data["seq_len"], data["pred_len"], data["in_features"]
    torch.manual_seed(seed)

    def make_model():
        return _make_model(expert_name, sl, pl, nf).to(device)

    init     = RandomSampleInitializer()
    syn_init = init.initialize_sequence(data["raw_train"], cfg["n_synthetic"])

    distiller = MTTDistiller(
        initializer             = init,
        matcher                 = MSEMatcher(),
        model_factory           = make_model,
        expert_recorder         = filtered_recorder,
        expert_epochs           = cfg["trajectory_gap"],
        syn_batch_size          = cfg["batch_size"],
        synthetic_lr            = cfg["synthetic_lr"],
        student_lr              = cfg["student_lr"],
        student_steps           = cfg["student_steps"],
        snapshot_student_steps  = cfg["snapshot_student_steps"],
        seq_len                 = sl,
        pred_len                = pl,
        device                  = device,
    )
    synthetic_seq = distiller.distill(
        synthetic_init = syn_init,
        n_steps        = cfg["n_distill_steps"],
        val_data       = data["val_data"],
    )
    return synthetic_seq, len(filtered_recorder.trajectory)


def _eval_on_real(student_name: str, data: dict, cfg: dict, device: str) -> float:
    """Train student on real data with early stopping; return test MSE."""
    sl, pl, nf = data["seq_len"], data["pred_len"], data["in_features"]
    torch.manual_seed(SEED)
    model = _make_model(student_name, sl, pl, nf).to(device)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
        criterion=torch.nn.MSELoss(),
        device=device, seq_len=sl,
    )
    trainer.fit(
        dataloader = TorchDataLoader(data["train_data"], batch_size=cfg["eval_batch_size"], shuffle=True),
        epochs     = cfg["eval_max_epochs"],
        val_loader = TorchDataLoader(data["val_data"],   batch_size=cfg["eval_batch_size"], shuffle=False),
        patience   = cfg["early_stop_patience"],
    )
    ev = Evaluator(seq_len=sl, batch_size=cfg["eval_batch_size"])
    return ev.test_on_real(model, data["test_data"].to(device))["MSE"]


def _eval_on_synthetic(
    student_name: str, synthetic_seq: torch.Tensor,
    data: dict, cfg: dict, device: str,
) -> float:
    """Train student on synthetic data with real-val early stopping; return test MSE."""
    sl, pl, nf, ws = (
        data["seq_len"], data["pred_len"], data["in_features"], data["window_size"]
    )
    syn_windows = torch.tensor(
        make_windows(synthetic_seq.cpu().numpy(), ws), dtype=torch.float32
    )
    torch.manual_seed(SEED)
    model = _make_model(student_name, sl, pl, nf).to(device)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["eval_lr"]),
        criterion=torch.nn.MSELoss(),
        device=device, seq_len=sl,
    )
    trainer.fit(
        dataloader = TorchDataLoader(syn_windows,        batch_size=cfg["eval_batch_size"], shuffle=True),
        epochs     = cfg["eval_max_epochs"],
        val_loader = TorchDataLoader(data["val_data"],   batch_size=cfg["eval_batch_size"], shuffle=False),
        patience   = cfg["early_stop_patience"],
    )
    ev = Evaluator(seq_len=sl, batch_size=cfg["eval_batch_size"])
    return ev.test_on_real(model, data["test_data"].to(device))["MSE"]


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_h2(cfg: dict = None, device: str = None):
    if cfg is None:
        cfg = DISTILL_CONFIG
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Output: {CSV_PATH}\n")

    real_mse_cache: dict = {}   # {(dataset, student): float}

    for tc in H2_TEST_CASES:
        dataset_name = tc["dataset"]
        expert_name  = tc["expert"]
        boundary     = tc["boundary"]

        print(f"\n{'='*70}")
        print(f"Test case: {dataset_name} x {expert_name} expert  (boundary=epoch {boundary})")
        print(f"{'='*70}")

        try:
            data = _load_dataset(dataset_name, cfg)
        except Exception as exc:
            print(f"  [SKIP] Cannot load {dataset_name}: {exc}")
            continue

        # ── Train expert once; record full 80-epoch trajectory ──────────────
        print(f"\n  Training {expert_name} expert for {cfg['expert_epochs']} epochs...")
        try:
            recorder = _train_expert_and_record(expert_name, data, cfg, device, SEED)
        except Exception as exc:
            print(f"  [FAIL] Expert training: {exc}")
            continue

        full_traj = recorder.trajectory
        print(f"  Trajectory: {len(full_traj)} checkpoints  "
              f"(steps {full_traj[0]['step']}..{full_traj[-1]['step']})")

        # ── Build the three filtered recorders ──────────────────────────────
        #   E: steps [0, boundary-1]  (before boundary)
        #   L: steps [boundary+1, max]  (after boundary, exclusive)
        #   F: full trajectory
        conditions = {
            "early_only": FilteredRecorder(full_traj, min_step=0,          max_step=boundary - 1),
            "late_only":  FilteredRecorder(full_traj, min_step=boundary + 1, max_step=None),
            "full_mtt":   FilteredRecorder(full_traj, min_step=0,          max_step=None),
        }

        for cond_name, filt_rec in conditions.items():
            n_ckpts = len(filt_rec)
            print(f"\n  -- Condition: {cond_name}  ({n_ckpts} checkpoints) --")

            if n_ckpts < cfg["trajectory_gap"] + 1:
                print(f"  [SKIP] Too few checkpoints ({n_ckpts}) for "
                      f"trajectory_gap={cfg['trajectory_gap']}. Need >= {cfg['trajectory_gap']+1}.")
                for student_name in ACTIVE_MODELS:
                    _append_csv({
                        "experiment": "h2_phase_isolation",
                        "dataset": dataset_name, "expert_model": expert_name,
                        "student_model": student_name, "seed": SEED,
                        "method": cond_name, "phase_boundary": boundary,
                        "n_ckpts_condition": n_ckpts,
                        "notes": f"skipped: only {n_ckpts} checkpoints available",
                    })
                continue

            # ── Distil synthetic data for this condition ─────────────────
            all_done = all(
                _row_exists(dataset_name, expert_name, s, cond_name)
                for s in ACTIVE_MODELS
            )
            if all_done:
                print(f"  [RESUME] All 4 students already in CSV for {cond_name}.")
                continue

            try:
                synthetic_seq, _ = _distill_with_recorder(
                    expert_name, filt_rec, data, cfg, device, SEED
                )
            except Exception as exc:
                print(f"  [FAIL] Distillation ({cond_name}): {exc}")
                for student_name in ACTIVE_MODELS:
                    if not _row_exists(dataset_name, expert_name, student_name, cond_name):
                        _append_csv({
                            "experiment": "h2_phase_isolation",
                            "dataset": dataset_name, "expert_model": expert_name,
                            "student_model": student_name, "seed": SEED,
                            "method": cond_name, "phase_boundary": boundary,
                            "n_ckpts_condition": n_ckpts,
                            "notes": f"distillation_failed: {str(exc)[:120]}",
                        })
                continue

            # ── Evaluate each student architecture ───────────────────────
            for student_name in ACTIVE_MODELS:
                if _row_exists(dataset_name, expert_name, student_name, cond_name):
                    print(f"    [RESUME] {student_name} already done.")
                    continue

                # Real MSE — compute once per (dataset, student), cache it
                key = (dataset_name, student_name)
                if key not in real_mse_cache:
                    print(f"    Computing real baseline for {student_name}...")
                    try:
                        real_mse_cache[key] = _eval_on_real(student_name, data, cfg, device)
                        print(f"      real_mse = {real_mse_cache[key]:.6f}")
                    except Exception as exc:
                        print(f"      [FAIL] real eval: {exc}")
                        real_mse_cache[key] = float("nan")

                real_mse = real_mse_cache[key]

                print(f"    Evaluating student: {student_name}")
                try:
                    transfer_mse = _eval_on_synthetic(student_name, synthetic_seq, data, cfg, device)
                    mse_ratio = (
                        transfer_mse / real_mse
                        if (real_mse > 0 and not np.isnan(real_mse))
                        else float("nan")
                    )
                    _append_csv({
                        "experiment":       "h2_phase_isolation",
                        "dataset":          dataset_name,
                        "expert_model":     expert_name,
                        "student_model":    student_name,
                        "seed":             SEED,
                        "method":           cond_name,
                        "phase_boundary":   boundary,
                        "n_ckpts_condition": n_ckpts,
                        "real_mse":         real_mse,
                        "transfer_mse":     transfer_mse,
                        "mse_ratio":        mse_ratio,
                        "n_pairs_available": n_ckpts,
                        "notes":            "",
                    })
                except Exception as exc:
                    print(f"    [FAIL] {student_name} eval: {exc}")
                    _append_csv({
                        "experiment":       "h2_phase_isolation",
                        "dataset":          dataset_name,
                        "expert_model":     expert_name,
                        "student_model":    student_name,
                        "seed":             SEED,
                        "method":           cond_name,
                        "phase_boundary":   boundary,
                        "n_ckpts_condition": n_ckpts,
                        "notes":            f"eval_failed: {str(exc)[:120]}",
                    })

    print(f"\n{'='*70}")
    print(f"H2 experiment complete.  Results: {CSV_PATH}")
    _print_summary()


def _print_summary():
    if not CSV_PATH.exists():
        return
    try:
        raw = pd.read_csv(CSV_PATH, header=None)
        if str(raw.iloc[0, 0]) == "experiment":
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.read_csv(CSV_PATH, header=None, names=CSV_COLUMNS[:len(raw.columns)])

        df["mse_ratio"] = pd.to_numeric(df["mse_ratio"], errors="coerce")

        print("\n--- H2 Summary: off-diagonal mse_ratio by condition ---")
        print(f"{'Test case':<25} {'Condition':<12} {'Off-diag mean':>14} {'Same-arch':>10}")
        print("-" * 65)

        for tc in H2_TEST_CASES:
            ds, ex = tc["dataset"], tc["expert"]
            sub = df[(df["dataset"] == ds) & (df["expert_model"] == ex)]
            for cond in ["early_only", "late_only", "full_mtt"]:
                c = sub[sub["method"] == cond]
                off = c[c["student_model"] != ex]["mse_ratio"].dropna()
                same = c[c["student_model"] == ex]["mse_ratio"].dropna()
                off_m  = f"{off.mean():.3f}"  if len(off)  else "—"
                same_m = f"{same.mean():.3f}" if len(same) else "—"
                print(f"  {ds} x {ex:<10} {cond:<12} {off_m:>14} {same_m:>10}")
    except Exception as exc:
        print(f"  [summary error] {exc}")


if __name__ == "__main__":
    run_h2()
