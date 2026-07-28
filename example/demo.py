"""
End-to-End Pipeline Demo
========================
A single, runnable demonstration of the full ts_distill library:

    1. Load a CSV dataset          (ts_distill CSVDataLoader)
    2. Choose an INITIALIZER       (random | geometry | uncertainty)   <- switchable
    3. Train an expert + distil    (MTT trajectory matching)
    4. Apply the POST-FIX          (FFT amplitude correction, strength alpha)
    5. Predict the BEST MIX RATIO  (hybrid mixer + optimal-ratio finder)
    6. Mix at that ratio           -> final synthetic+real dataset + accuracy

Everything is imported from the built library in `src/`; nothing is
re-implemented here. Configuration comes from `ts_distill.config`.

Run from the PROJECT ROOT (the folder containing `example/` and `src/`):

    python -m example.end_to_end_demo

Edit the SWITCHES block below to change dataset, model, initializer, or
post-fix strength.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Make the project root importable (so `import ts_distill...` works) ────────
sys.path.append(str(Path(__file__).parent.parent))

# ── Everything below is imported from the built library ──────────────────────
from ts_distill.config import (
    DEFAULT_CONFIG, DATASET_CONFIGS, MODEL_CONFIGS, phase_boundary_config,
)

from ts_distill.data_pipeline.data_loader.csv_loader import CSVDataLoader
from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader

from ts_distill.models.factory import create_model
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback

from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.distillation_algorithm.phase_aware_mtt import PhaseAwareMTTDistiller

# Phase-aware matching: val-loss curve recorder + plateau detector for T+.
from ts_distill.trainer.callback.val_loss_callback import ValLossRecorderCallback
from ts_distill.trajectory.phase_detector.valloss_detector import ValLossPlateauDetector

# The three switchable initializers.
from ts_distill.distillation_core.initializer.random_sample_initializer import RandomSampleInitializer
from ts_distill.distillation_core.initializer.geommetry_sequence_initializer import GeometrySequenceInitializer
from ts_distill.distillation_core.initializer.uncertainty_sequence_initializer import UncertaintySampleInitializer

# Post-fix (temporal correction).
from ts_distill.post_processing import FFTAmplitudePostFix

# Standard evaluator (for the real baseline).
from ts_distill.evaluation.evaluation import Evaluator

# Hybrid mixer + best-ratio prediction.
from ts_distill.evaluation.hybrid_evaluation.hybrid import BaseHybridEvaluator, RandomAnchorSelector
from ts_distill.evaluation.hybrid_evaluation.optimal_ratio_finder import find_optimal_ratio


# =============================================================================
# CONFIGURATION — edit these, then run.  Each run appends one row to the CSV,
# so you can sweep datasets / models / methods and collect results over time.
# =============================================================================

# ── 1. What to run (change between runs to build the results table) ──────────
DATASET       = 'ETTh2'     # any key in DATASET_CONFIGS
MODEL         = 'DLinear'       # 'MLP' | 'DLinear' | 'CNN' | 'LSTM'
INITIALIZER   = 'random'    # 'random' | 'geometric' | 'uncertainty'
POSTFIX_ALPHA = 0.7      # FFT post-fix strength in [0, 1]  (0.0 disables it)
SEED          = 42

# ── 1b. Phase-aware matching (same option as experiment_matrix.py) ───────────
# False (default): standard MTT — parameter matching throughout. Keeps the
#   distillation RNG stream identical to experiment_inti.py (exact-match path).
# True: PhaseAwareMTTDistiller — parameter matching early, prediction matching
#   late. The phase boundary T+ is detected on the fly from the expert's
#   validation-loss curve (ValLossPlateauDetector). This adds one val pass per
#   expert epoch, which consumes torch RNG, so numbers will NOT match the
#   standard-MTT path — expected, it is a different method.
USE_PHASE_AWARE_MATCHING = True

# The detector settings live in the library (ts_distill.config), because they
# are tuned per expert architecture and every script needs the same values.
# `phase_boundary_config(MODEL)` returns the entry for this run's architecture,
# falling back to a default for anything without a dedicated one (e.g. LSTM).
# To override a single knob for an experiment, edit it here rather than in the
# library — the returned dict is a copy, so this cannot leak into other runs:
#
#     PHASE_BOUNDARY_OVERRIDES = {'burn_in_epochs': 10}
#
PHASE_BOUNDARY_OVERRIDES = {}

# ── 2. Hybrid mixer sweep grid ───────────────────────────────────────────────
# 0.0 = pure synthetic. The real baseline is measured separately from the
# trained expert (STEP 2), so a 100%-real point is not needed here.
MIXING_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.5)

# ── 3. Pipeline hyperparameters (override the library defaults) ──────────────
# Small = fast demo. For real/thesis results use the "full" values in comments.
# NOTE: the distillation learning rates below MUST match experiment_inti.py's
# DISTILL_CONFIG, or the demo will distil differently and give different MSE.
# In particular synthetic_lr=5.0 makes distillation actually move the synthetic
# data; a tiny value (e.g. 0.01) barely distils, leaving synthetic ≈ its real
# init block and giving an artificially low transfer MSE.
PIPELINE = {
    'expert_epochs':       80,    # full: 80
    'n_distill_steps':     300,    # full: 300
    'synthetic_lr':        5.0,    # MUST match experiment_inti (was inheriting 0.01)
    'student_lr':          0.01,   # match experiment_inti
    'student_steps':        20,    # match experiment_inti
    'eval_max_epochs':     300,    # full: 300
    'early_stop_patience':  10,    # full: 10
}

# ── 4. Output — where each run's result row is appended ──────────────────────
LOG_RESULTS = True
RESULTS_CSV = Path(__file__).parent / 'results' / 'end_to_end_results.csv'

# ─────────────────────────────────────────────────────────────────────────────
# Registry so INITIALIZER string maps to a class. All share the same
# initialize_sequence(raw_train_data, n_synthetic) interface.
# 'geometric' is accepted as an alias for 'geometry' to avoid a common typo.
INITIALIZERS = {
    'random':      RandomSampleInitializer,
    'geometric':   GeometrySequenceInitializer,
    'uncertainty': UncertaintySampleInitializer,
}
if INITIALIZER not in INITIALIZERS:
    raise ValueError(
        f"INITIALIZER='{INITIALIZER}' is not valid. "
        f"Choose one of: {sorted(set(INITIALIZERS))}."
    )

# Merge library defaults with the pipeline overrides above.
cfg = {**DEFAULT_CONFIG, **PIPELINE}


def banner(text):
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def append_result_csv(csv_path, row: dict, key_cols: list):
    """
    Append one result row to csv_path. If a row with the same key already
    exists (same dataset, model, initializer, postfix_alpha, seed), it is
    REPLACED — so re-running the same config overwrites instead of duplicating.
    New columns from changed MIXING_RATIOS are aligned automatically (NaN-filled).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        # Back-compat: older CSVs may predate some key columns (e.g. 'matching').
        # Add any missing ones so the key comparison below never KeyErrors.
        for k in key_cols:
            if k not in df_old.columns:
                df_old[k] = ''
        incoming = tuple(str(row[k]) for k in key_cols)
        keep = df_old[key_cols].astype(str).apply(tuple, axis=1) != incoming
        df_out = pd.concat([df_old[keep], df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)


def main():
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_cfg = DATASET_CONFIGS[DATASET]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])
    in_features = dataset_cfg.get('in_features', cfg['in_features'])
    window_size = seq_len + pred_len

    def make_model():
        return create_model(MODEL, seq_len, pred_len, in_features, MODEL_CONFIGS[MODEL])

    # ── STEP 1: Load the CSV with the library's CSV loader ───────────────────
    banner(f"STEP 1  Load dataset  ({DATASET})")
    loader = CSVDataLoader()
    df = loader.load_data(dataset_cfg['csv_path'])
    values = df.iloc[:, 1:].values.astype(np.float32)   # drop the date column
    print(f"  Loaded {values.shape[0]} rows x {values.shape[1]} channels from {dataset_cfg['csv_path']}")

    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values=values, window_size=window_size, seq_len=seq_len, dataset_cfg=dataset_cfg
    )
    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])       # fit on TRAIN only (no leakage)
    data = scaler.transform(values)

    train_windows  = torch.tensor(make_windows(data[train_start:train_end], window_size), dtype=torch.float32)
    val_windows    = torch.tensor(make_windows(data[val_start:val_end],     window_size), dtype=torch.float32)
    test_windows   = torch.tensor(make_windows(data[test_start:test_end],   window_size), dtype=torch.float32)
    raw_train_data = torch.tensor(data[train_start:train_end], dtype=torch.float32)
    raw_val_data   = torch.tensor(data[val_start:val_end],     dtype=torch.float32)
    print(f"  Windows -> train {len(train_windows)}, val {len(val_windows)}, test {len(test_windows)}")

    # ── STEP 2: Train expert ──────────────────────────────────────────────────
    # The expert is trained FIRST, before the initializer. CRUCIAL: nothing may
    # consume torch RNG between here and the distillation, or the initializer's
    # random draw (and hence the whole distillation) shifts, and the synthetic no
    # longer matches experiment_inti. In particular the real-baseline MSE is NOT
    # computed here — evaluating a model iterates a DataLoader, which draws one
    # torch RNG value. It is measured AFTER distillation instead (same value,
    # since the expert is already trained and unchanged).
    banner(f"STEP 2  Train expert ({cfg['expert_epochs']} ep)")
    # Shared val loader — used for the expert val-loss curve (phase-aware boundary
    # detection) and for eval-time early stopping later. Constructing it draws no
    # RNG; only iterating does (which, in phase-aware mode, happens per epoch).
    eval_val_loader = DataLoader(val_windows, batch_size=cfg['eval_batch_size'], shuffle=False)

    recorder       = SimpleRecorder(record_every=1)
    expert_model   = make_model()
    expert_trainer = Trainer(
        model     = expert_model,
        optimizer = torch.optim.SGD(expert_model.parameters(),
                                    lr=cfg['expert_lr'], momentum=cfg['expert_momentum']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    expert_callbacks = [recorder, SimpleCallback()]

    # Only record the per-epoch val-loss curve when phase-aware matching is on —
    # it costs one extra val pass per epoch, otherwise unused.
    #
    # CRITICAL: iterating the val DataLoader draws torch RNG. Fired at each
    # epoch end, that would shift the NEXT epoch's MiniBatchLoader randperm →
    # different expert weights → different real_mse. real_mse must be
    # config-independent (expert depends only on SEED + data + expert config,
    # never on whether phase-aware is toggled). So snapshot/restore the RNG
    # state around the eval, leaving the expert training stream untouched.
    val_loss_recorder = None
    if USE_PHASE_AWARE_MATCHING:
        def _rng_safe_val_eval():
            rng_state = torch.get_rng_state()
            try:
                return expert_trainer.eval_epoch(eval_val_loader)
            finally:
                torch.set_rng_state(rng_state)
        val_loss_recorder = ValLossRecorderCallback(eval_fn=_rng_safe_val_eval)
        expert_callbacks.append(val_loss_recorder)

    expert_trainer.fit(
        dataloader = MiniBatchLoader(train_windows, batch_size=cfg['batch_size']),
        epochs     = cfg['expert_epochs'],
        callbacks  = expert_callbacks,
    )

    # ── STEP 2b: Detect phase boundary T+ (phase-aware matching only) ─────────
    phase_boundary = None
    boundary_cfg   = None
    if USE_PHASE_AWARE_MATCHING:
        # Config is chosen by the EXPERT architecture, matching experiment_matrix.
        boundary_cfg = {**phase_boundary_config(MODEL), **PHASE_BOUNDARY_OVERRIDES}
        boundary_result = ValLossPlateauDetector(**boundary_cfg).detect(
            val_loss_recorder.val_losses
        )
        phase_boundary = boundary_result['boundary_epoch']
        if phase_boundary is not None:
            print(f"  [PhaseDetector] T+ = {phase_boundary}  (cfg for {MODEL}: "
                  f"burn_in={boundary_cfg['burn_in_epochs']}, "
                  f"patience={boundary_cfg['patience']}, "
                  f"min_delta_frac={boundary_cfg['min_delta_frac']})")
            print(f"  [PhaseDetector] best_val_loss={boundary_result['best_val_loss']:.6f}, "
                  f"final_val_loss={boundary_result['final_val_loss']:.6f}")
        else:
            print("  [PhaseDetector] No plateau found — phase_aware_mtt runs as "
                  "parameter matching throughout (same as standard MTT).")

    # ── STEP 3: Choose initializer + distil ──────────────────────────────────
    banner(f"STEP 3  Initializer ({INITIALIZER}) + distil ({cfg['n_distill_steps']} steps)")
    initializer    = INITIALIZERS[INITIALIZER]()
    synthetic_init = initializer.initialize_sequence(raw_train_data, cfg['n_synthetic'])
    print(f"  Synthetic seed shape: {tuple(synthetic_init.shape)}")

    distiller_kwargs = dict(
        initializer            = initializer,
        matcher                = MSEMatcher(),
        model_factory          = make_model,
        expert_recorder        = recorder,
        expert_epochs          = cfg['trajectory_gap'],
        syn_batch_size         = cfg['batch_size'],
        synthetic_lr           = cfg['synthetic_lr'],
        student_lr             = cfg['student_lr'],
        student_steps          = cfg['student_steps'],
        snapshot_student_steps = cfg['snapshot_student_steps'],
        seq_len                = seq_len,
        pred_len               = pred_len,
    )
    if USE_PHASE_AWARE_MATCHING:
        distiller = PhaseAwareMTTDistiller(**distiller_kwargs, phase_boundary=phase_boundary)
    else:
        distiller = MTTDistiller(**distiller_kwargs)
    synthetic_distilled = distiller.distill(
        synthetic_init = synthetic_init,
        n_steps        = cfg['n_distill_steps'],
        val_data       = val_windows,
    )
    print(f"  Distilled synthetic shape: {tuple(synthetic_distilled.shape)}")

    # ── Synthetic-only MSE — EXACT experiment_inti protocol (its Step 4b) ─────
    # Measured on the RAW distilled synthetic (before any post-fix) with a fresh
    # probe: manual_seed(1) -> Adam -> early stopping on the real val set ->
    # tested on the real test set. This reproduces experiment_inti.py exactly, so
    # the number matches it. It does NOT come from the hybrid mixer's hybrid_0.
    # For the match to hold, SEED and the distillation config must equal
    # experiment_inti's (SEED=123, INITIALIZER='random', synthetic_lr=5.0).
    syn_windows_raw = torch.tensor(
        make_windows(synthetic_distilled.detach().cpu().numpy(), window_size),
        dtype=torch.float32,
    )
    torch.manual_seed(1)
    syn_model = make_model()
    Trainer(
        model     = syn_model,
        optimizer = torch.optim.Adam(syn_model.parameters(), lr=cfg['eval_lr']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    ).fit(
        dataloader = DataLoader(syn_windows_raw, batch_size=cfg['eval_batch_size'], shuffle=True),
        epochs     = cfg['eval_max_epochs'],
        val_loader = eval_val_loader,
        patience   = cfg['early_stop_patience'],
    )
    synthetic_only_mse = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size']).test_on_real(
        syn_model, test_windows.to(device)
    )['MSE']
    print(f"  Synthetic-only MSE (experiment_inti protocol): {synthetic_only_mse:.6f}")

    # ── Real baseline MSE — measured AFTER distillation on purpose ────────────
    # Evaluating a model iterates a DataLoader, which draws one torch RNG value;
    # doing it before the initializer would shift the distillation RNG and break
    # the match with experiment_inti. The expert is already trained and
    # unchanged, so this is the same real baseline, just computed here. It stays
    # config-independent (expert trained before the initializer in STEP 2).
    real_mse = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size']).test_on_real(
        expert_model, test_windows.to(device)
    )['MSE']
    print(f"  Real baseline MSE (expert on test set): {real_mse:.6f}")

    # ── STEP 4: Post-fix (FFT amplitude correction) ──────────────────────────
    banner(f"STEP 4  Post-fix  (FFT amplitude, alpha={POSTFIX_ALPHA})")
    fft_before = fft_after = None
    if POSTFIX_ALPHA and POSTFIX_ALPHA > 0.0:
        synthetic_final = torch.tensor(
            FFTAmplitudePostFix(alpha=POSTFIX_ALPHA).apply(synthetic_distilled, raw_train_data),
            dtype=torch.float32,
        )
        fft_before = FFTAmplitudePostFix.fft_distance(synthetic_distilled, raw_train_data)
        fft_after  = FFTAmplitudePostFix.fft_distance(synthetic_final,     raw_train_data)
        print(f"  fft_distance: {fft_before:.4f}  ->  {fft_after:.4f}  (lower = closer to real)")
    else:
        synthetic_final = synthetic_distilled.detach()
        print("  Post-fix disabled (alpha=0). Using raw distilled data.")

    # ── STEP 5: Hybrid mixer — predict the best real/synthetic ratio ─────────
    banner("STEP 5  Hybrid mixer  ->  predict best mix ratio")
    hybrid = BaseHybridEvaluator(
        anchor_selector     = RandomAnchorSelector(),
        seq_len             = seq_len,
        batch_size          = cfg['eval_batch_size'],
        device              = device,
        eval_max_epochs     = cfg['eval_max_epochs'],
        early_stop_patience = cfg['early_stop_patience'],
        eval_lr             = cfg['eval_lr'],
    )
    test_loader = DataLoader(test_windows, batch_size=cfg['eval_batch_size'], shuffle=False)

    mix_results = hybrid.evaluate_mixing(
        synthetic_data   = synthetic_final,
        real_train_data  = raw_train_data,
        real_test_loader = test_loader,
        model_fn         = make_model,
        window_size      = window_size,
        mixing_ratios    = MIXING_RATIOS,
        real_val_data    = raw_val_data,     # enables early stopping per ratio
    )

    # Turn {'hybrid_0': {'MSE':..}, ...} into (ratio%, mse) lists for the finder.
    points = sorted((int(k.split('_')[1]), v['MSE']) for k, v in mix_results.items())
    ratios = [p[0] for p in points]
    mses   = [p[1] for p in points]

    print(f"\n  {'real %':>7}  {'test MSE':>10}")
    print("  " + "-" * 20)
    for r, m in zip(ratios, mses):
        finite_tag = "" if np.isfinite(m) else "  <- diverged (NaN)"
        print(f"  {r:>7}  {m:>10.6f}{finite_tag}")

    # Drop any ratio whose probe diverged (NaN/inf MSE): find_optimal_ratio's
    # interpolation requires finite values. Divergence usually hits one ratio
    # (an unstable model, or amplitude-inflated post-fixed synthetic); the
    # remaining finite points still define the trade-off curve.
    finite  = [(r, m) for r, m in zip(ratios, mses) if np.isfinite(m)]
    dropped = [r for r, m in zip(ratios, mses) if not np.isfinite(m)]
    if dropped:
        print(f"  [warn] dropped non-finite MSE at ratios {dropped}% (probe diverged)")
    if len(finite) < 2:
        raise RuntimeError(
            "Too few finite MSE points to pick a ratio — most probes diverged. "
            "Try a more stable model, a lower eval_lr, fewer distill steps, or a "
            "smaller postfix alpha."
        )

    f_ratios = [r for r, _ in finite]
    f_mses   = [m for _, m in finite]
    best = find_optimal_ratio(f_ratios, f_mses)      # balances accuracy + compression
    best_ratio_pct = int(best['nearest_tested_ratio'])
    print(f"\n  Predicted best ratio r* = {best['r_star']:.1f}%  "
          f"(nearest tested = {best_ratio_pct}% real)")

    # ── STEP 6: Mix at the best ratio -> final dataset + accuracy ─────────────
    banner(f"STEP 6  Final dataset  (mix at {best_ratio_pct}% real)")
    final_dataset = hybrid.hybridmixture(
        synthetic_data  = synthetic_final,
        real_train_data = raw_train_data,
        real_ratio      = best_ratio_pct / 100.0,
        window_size     = window_size,
    )
    final_mse = mix_results[f"hybrid_{best_ratio_pct}"]['MSE']
    # NOTE: synthetic_only_mse was already computed above via the exact
    # experiment_inti protocol (raw distilled, seed(1) probe) — do NOT overwrite
    # it with the hybrid mixer's hybrid_0 point.

    print(f"  Final training set: {len(final_dataset)} windows "
          f"({best_ratio_pct}% real + {100 - best_ratio_pct}% synthetic)")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    banner("SUMMARY")
    print(f"  Dataset / Model / Initializer : {DATASET} / {MODEL} / {INITIALIZER}")
    print(f"  Post-fix alpha                : {POSTFIX_ALPHA}")
    print(f"  Real baseline   MSE (expert)  : {real_mse:.6f}   <- fixed lower bound")
    print(f"  Synthetic-only  MSE (inti eval): {synthetic_only_mse:.6f}   <- matches experiment_inti")
    print(f"  Best mix ratio                : {best_ratio_pct}% real")
    print(f"  FINAL hybrid    MSE           : {final_mse:.6f}")
    print(f"  Final synthetic+real dataset  : {len(final_dataset)} windows ready for use")
    print()

    # ── LOG: append one row to the results CSV ───────────────────────────────
    if LOG_RESULTS:
        row = {
            'dataset':            DATASET,
            'model':              MODEL,
            'initializer':        INITIALIZER,
            'matching':           'phase_aware_mtt' if USE_PHASE_AWARE_MATCHING else 'param_mtt',
            'phase_boundary':     phase_boundary if phase_boundary is not None else '',
            # The detector config is per-architecture, so record the knobs that
            # produced this T+ — otherwise rows from different models are not
            # comparable.
            'phase_burn_in':      boundary_cfg['burn_in_epochs']  if boundary_cfg else '',
            'phase_patience':     boundary_cfg['patience']        if boundary_cfg else '',
            'phase_min_delta':    boundary_cfg['min_delta_frac']  if boundary_cfg else '',
            'postfix_alpha':      POSTFIX_ALPHA,
            'seed':               SEED,
            'expert_epochs':      cfg['expert_epochs'],
            'n_distill_steps':    cfg['n_distill_steps'],
            'real_mse':           round(real_mse, 6),
            'synthetic_only_mse': round(synthetic_only_mse, 6),
            'best_ratio_pct':     best_ratio_pct,
            'r_star':             round(float(best['r_star']), 2),
            'compression_pct':    round(float(best['compression_pct']), 2),
            'final_hybrid_mse':   round(final_mse, 6),
            'fft_before':         round(fft_before, 6) if fft_before is not None else None,
            'fft_after':          round(fft_after, 6) if fft_after is not None else None,
        }
        # Full mixing curve: one column per tested ratio.
        for r, m in zip(ratios, mses):
            row[f'mse_r{r}'] = round(m, 6)

        key_cols = ['dataset', 'model', 'initializer', 'matching', 'postfix_alpha', 'seed']
        append_result_csv(RESULTS_CSV, row, key_cols)
        print(f"  [log] result row appended -> {RESULTS_CSV}\n")


if __name__ == '__main__':
    main()
