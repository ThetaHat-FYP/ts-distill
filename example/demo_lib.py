"""
ts_distill — Library Showcase
=============================
The shortest complete tour of the public API. Every import below comes from the
top-level `ts_distill` namespace, which is the interface an installed user gets:

    pip install ts-distill
    python demo_lib.py

Stages
------
    1. Load + split       CSVDataLoader, get_data_splits, make_windows
    2. Train an expert    Trainer + SimpleRecorder     (records the trajectory)
    3. Initialise         RandomSample / Geometry / UncertaintySample
    4. Distil             MTTDistiller or PhaseAwareMTTDistiller
    5. Measure utility    Evaluator                    (probe trained on synthetic)
    6. Post-fix           FFTAmplitudePostFix          (restore the spectrum)
    7. Measure fidelity   MetricAggregator             (9 temporal metrics)
    8. Predict r* + mix   predict_r_star               (no sweep) + hybridmixture
    9. Validate r*        evaluate_mixing              (optional oracle)

Predicting vs. measuring r*
---------------------------
The optimal real/synthetic mixing ratio r* can be obtained two ways:

  predict_r_star()   scores four cheap properties of data you already have and
                     maps them to r*. Costs NO extra training. This is the
                     method the library is for, and what stage 8 uses.

  evaluate_mixing()  trains one probe per candidate ratio and reads off the
                     minimum. Expensive, but it is ground truth — so stage 9
                     runs it only to VERIFY the prediction. Set
                     VALIDATE_WITH_SWEEP = False to skip it.

This file is a *showcase*, tuned to finish in a few minutes on CPU. It is not the
reproduction script — for full-length runs with result logging see `demo.py`, and
for the dataset x model matrix see `experiments/experiment_matrix.py`.

Edit the SETTINGS block, then run from anywhere:

    python example/demo_lib.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make the in-repo `src/` importable so the demo runs without `pip install`.
# An installed user does not need this line.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# ── The entire public API, from one namespace ────────────────────────────────
from ts_distill import (
    __version__,
    configure_logging,
    # data
    CSVDataLoader, get_data_splits, make_windows, MiniBatchLoader,
    # models + training
    create_model, Trainer, SimpleRecorder, SimpleCallback,
    # distillation
    MSEMatcher, MTTDistiller, PhaseAwareMTTDistiller, ValLossPlateauDetector,
    ValLossRecorderCallback,
    RandomSampleInitializer, GeometrySequenceInitializer, UncertaintySampleInitializer,
    # post-processing + evaluation + metrics
    FFTAmplitudePostFix, Evaluator, BaseHybridEvaluator, RandomAnchorSelector,
    find_optimal_ratio, MetricAggregator,
    # config
    DEFAULT_CONFIG, DATASET_CONFIGS, DATASET_PERIODS, MODEL_CONFIGS,
    resolve_csv_path, phase_boundary_config,
)

# The sweep-free r* predictor. Not re-exported at top level because it is a
# research-stage API; import it from its own module.
from ts_distill.evaluation.hybrid_evaluation.ratio_predictor import predict_r_star


# =============================================================================
# SETTINGS
# =============================================================================

# Where the benchmark CSVs live. The library only stores each dataset's FILE
# NAME, never a path — once installed it has no idea where your data sits — so
# the directory is declared here and joined via resolve_csv_path().
# Point this anywhere: '/data/ett', 'C:/datasets', ...
DATA_DIR = Path(__file__).resolve().parent      # this example/ folder

DATASET     = 'ETTm2'          # any key in DATASET_CONFIGS
MODEL       = 'DLinear'            # 'MLP' | 'DLinear' | 'CNN' | 'LSTM'
INITIALIZER = 'random'         # 'random' | 'geometry' | 'uncertainty'
ALPHA       = 0             # post-fix strength in [0, 1]; 0 disables it
SEED        = 42

# Phase-aware matching: parameter matching early, prediction matching after the
# detected boundary T+. Costs one extra validation pass per expert epoch.
USE_PHASE_AWARE = False

# T+ is found by ValLossPlateauDetector, whose settings are tuned PER EXPERT
# ARCHITECTURE and live in the library (ts_distill.config), so every script uses
# the same values. `phase_boundary_config(MODEL)` returns this run's entry, with
# a fallback for architectures that have no dedicated one (e.g. LSTM).
# To tweak a single knob for one experiment, put it here instead of editing the
# library — the returned dict is a copy, so this cannot leak into other runs:
#
#     PHASE_BOUNDARY_OVERRIDES = {'burn_in_epochs': 10}
#
PHASE_BOUNDARY_OVERRIDES = {}

# Kept small so the showcase finishes quickly. Thesis-scale values in comments.
OVERRIDES = {
    'expert_epochs':       80,    # full: 80
    'n_distill_steps':     300,    # full: 300
    'synthetic_lr':        5.0,    # MUST match experiment_inti (was inheriting 0.01)
    'student_lr':          0.01,   # match experiment_inti
    'student_steps':        20,    # match experiment_inti
    'eval_max_epochs':     300,    # full: 300
    'early_stop_patience':  10,    # full: 10
}

# The pipeline PREDICTS r* from four cheap measures (stage 8) — no sweep.
# Set this True to additionally run the measured sweep and check the prediction
# against it. That costs one probe training per ratio, so it is the expensive
# part of this script; turn it off to see the predictor's cost in isolation.
VALIDATE_WITH_SWEEP = True
MIXING_RATIOS = (0.0, 0.1, 0.2, 0.3)   # ratios probed ONLY when validating

INITIALIZERS = {
    'random':      RandomSampleInitializer,
    'geometry':    GeometrySequenceInitializer,
    'uncertainty': UncertaintySampleInitializer,
}


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main() -> None:
    configure_logging('INFO')      # library is silent until asked
    banner(f"ts_distill v{__version__}  -  library showcase")

    if INITIALIZER not in INITIALIZERS:
        raise ValueError(f"INITIALIZER must be one of {sorted(INITIALIZERS)}")

    cfg    = {**DEFAULT_CONFIG, **OVERRIDES}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device={device}  dataset={DATASET}  model={MODEL}  "
          f"init={INITIALIZER}  alpha={ALPHA}  seed={SEED}")

    torch.manual_seed(SEED)

    dataset_cfg = DATASET_CONFIGS[DATASET]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])
    in_features = dataset_cfg.get('in_features', cfg['in_features'])
    window_size = seq_len + pred_len

    def make_model():
        return create_model(MODEL, seq_len, pred_len, in_features, MODEL_CONFIGS[MODEL])

    # ── 1. Load and split ────────────────────────────────────────────────────
    banner("1  Load + split")
    csv_path = resolve_csv_path(DATASET, DATA_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Put the {DATASET} benchmark CSV there, "
            f"or point DATA_DIR at the folder that holds it."
        )

    values = CSVDataLoader().load_data(str(csv_path)).iloc[:, 1:].values.astype(np.float32)
    print(f"  loaded {values.shape[0]} rows x {values.shape[1]} channels")

    tr_s, tr_e, va_s, va_e, te_s, te_e = get_data_splits(
        values=values, window_size=window_size, seq_len=seq_len, dataset_cfg=dataset_cfg,
    )

    # Standardise using TRAIN statistics only, so no test information leaks in.
    mean = values[tr_s:tr_e].mean(axis=0)
    std  = values[tr_s:tr_e].std(axis=0)
    std[std == 0] = 1.0
    data = (values - mean) / std

    train_windows = torch.tensor(make_windows(data[tr_s:tr_e], window_size), dtype=torch.float32)
    val_windows   = torch.tensor(make_windows(data[va_s:va_e], window_size), dtype=torch.float32)
    test_windows  = torch.tensor(make_windows(data[te_s:te_e], window_size), dtype=torch.float32)
    raw_train     = torch.tensor(data[tr_s:tr_e], dtype=torch.float32)
    raw_val       = torch.tensor(data[va_s:va_e], dtype=torch.float32)
    print(f"  windows -> train {len(train_windows)}, val {len(val_windows)}, "
          f"test {len(test_windows)}")

    # ── 2. Train the expert and record its trajectory ────────────────────────
    # Distillation matches the expert's PATH through parameter space, so the
    # trajectory has to be recorded during training, not reconstructed after.
    banner(f"2  Train expert ({cfg['expert_epochs']} epochs)")
    val_loader = DataLoader(val_windows, batch_size=cfg['eval_batch_size'], shuffle=False)

    recorder       = SimpleRecorder(record_every=1)
    expert         = make_model()
    expert_trainer = Trainer(
        model     = expert,
        optimizer = torch.optim.SGD(expert.parameters(),
                                    lr=cfg['expert_lr'], momentum=cfg['expert_momentum']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = seq_len,
    )
    callbacks = [recorder, SimpleCallback()]

    # The val-loss curve is only needed to locate the phase boundary T+.
    # Iterating the val loader consumes torch RNG, which would shift every later
    # random draw (including the initializer's), so the state is restored around
    # it — that keeps results identical whether or not phase-aware is enabled.
    val_recorder = None
    if USE_PHASE_AWARE:
        def rng_safe_val_eval():
            state = torch.get_rng_state()
            try:
                return expert_trainer.eval_epoch(val_loader)
            finally:
                torch.set_rng_state(state)
        val_recorder = ValLossRecorderCallback(eval_fn=rng_safe_val_eval)
        callbacks.append(val_recorder)

    expert_trainer.fit(
        dataloader = MiniBatchLoader(train_windows, batch_size=cfg['batch_size']),
        epochs     = cfg['expert_epochs'],
        callbacks  = callbacks,
    )

    # The detector settings come from the library and are selected by the EXPERT
    # architecture — a single setting that finds a good boundary for one model
    # pushes others so late that phase-aware matching degenerates into plain
    # parameter matching.
    phase_boundary = None
    boundary_cfg   = None
    if USE_PHASE_AWARE:
        boundary_cfg = {**phase_boundary_config(MODEL), **PHASE_BOUNDARY_OVERRIDES}
        detected = ValLossPlateauDetector(**boundary_cfg).detect(val_recorder.val_losses)
        phase_boundary = detected['boundary_epoch']
        print(f"  phase boundary T+ = {phase_boundary}  (cfg for {MODEL}: "
              f"burn_in={boundary_cfg['burn_in_epochs']}, "
              f"patience={boundary_cfg['patience']}, "
              f"min_delta_frac={boundary_cfg['min_delta_frac']})")

    # ── 3. Initialise the synthetic sequence ─────────────────────────────────
    # All three strategies return ONE contiguous block of n_synthetic timesteps;
    # they differ only in which block they choose.
    banner(f"3  Initialise ({INITIALIZER})")
    initializer    = INITIALIZERS[INITIALIZER]()
    synthetic_init = initializer.initialize_sequence(raw_train, cfg['n_synthetic'])
    print(f"  synthetic seed shape: {tuple(synthetic_init.shape)}")

    # ── 4. Distil ────────────────────────────────────────────────────────────
    banner(f"4  Distil ({cfg['n_distill_steps']} steps, "
           f"{'phase-aware' if USE_PHASE_AWARE else 'standard'} MTT)")
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
    distiller = (
        PhaseAwareMTTDistiller(**distiller_kwargs, phase_boundary=phase_boundary)
        if USE_PHASE_AWARE else
        MTTDistiller(**distiller_kwargs)
    )
    synthetic = distiller.distill(
        synthetic_init = synthetic_init,
        n_steps        = cfg['n_distill_steps'],
        val_data       = val_windows,
    )
    print(f"  distilled shape: {tuple(synthetic.shape)}  "
          f"(vs {len(train_windows)} real windows)")

    # ── 5. Utility: train a probe on the synthetic data, score it on real ────
    # This is the number distillation actually optimises for.
    banner("5  Utility - probe trained on synthetic, tested on real")
    evaluator = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size'])

    def probe_mse(training_windows, seed: int) -> float:
        """Train a fresh model on `training_windows`, return its real-test MSE."""
        torch.manual_seed(seed)
        model = make_model()
        Trainer(
            model     = model,
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['eval_lr']),
            criterion = torch.nn.MSELoss(),
            device    = device,
            seq_len   = seq_len,
        ).fit(
            dataloader = DataLoader(training_windows,
                                    batch_size=cfg['eval_batch_size'], shuffle=True),
            epochs     = cfg['eval_max_epochs'],
            val_loader = val_loader,
            patience   = cfg['early_stop_patience'],
        )
        return evaluator.test_on_real(model, test_windows.to(device))['MSE']

    syn_windows = torch.tensor(
        make_windows(synthetic.detach().cpu().numpy(), window_size), dtype=torch.float32,
    )
    synthetic_mse = probe_mse(syn_windows, seed=1)

    # The real baseline uses the already-trained expert, so it is unaffected by
    # any distillation setting — the fixed lower bound to compare against.
    real_mse = evaluator.test_on_real(expert, test_windows.to(device))['MSE']

    print(f"  real-data MSE      : {real_mse:.6f}   <- lower bound")
    print(f"  synthetic-data MSE : {synthetic_mse:.6f}")
    print(f"  performance kept   : {real_mse / synthetic_mse * 100:.1f}%")
    print(f"  compression        : {cfg['n_synthetic']}/{len(train_windows)} "
          f"({cfg['n_synthetic'] / len(train_windows) * 100:.2f}%)")

    # ── 6. Post-fix: restore the amplitude spectrum ──────────────────────────
    # Distillation optimises utility and wrecks the frequency content. The
    # post-fix keeps the synthetic PHASE and blends the AMPLITUDE toward the
    # real data's segment-averaged envelope, so alpha trades utility for
    # temporal realism after the fact.
    banner(f"6  Post-fix (FFT amplitude, alpha={ALPHA})")
    if ALPHA > 0:
        fixed = torch.tensor(
            FFTAmplitudePostFix(alpha=ALPHA).apply(synthetic, raw_train),
            dtype=torch.float32,
        )
        before = FFTAmplitudePostFix.fft_distance(synthetic, raw_train)
        after  = FFTAmplitudePostFix.fft_distance(fixed,     raw_train)
        print(f"  fft_distance: {before:.4f} -> {after:.4f}  "
              f"({(1 - after / before) * 100:.1f}% closer to real)")

        fixed_windows = torch.tensor(
            make_windows(fixed.detach().cpu().numpy(), window_size), dtype=torch.float32,
        )
        fixed_mse = probe_mse(fixed_windows, seed=1)
        print(f"  utility cost: {synthetic_mse:.6f} -> {fixed_mse:.6f} "
              f"({(fixed_mse / synthetic_mse - 1) * 100:+.1f}%)")
    else:
        fixed     = synthetic.detach()
        fixed_mse = synthetic_mse
        print("  disabled (alpha=0)")

    # ── 7. Temporal fidelity metrics ─────────────────────────────────────────
    # Utility alone does not say whether the synthetic data still LOOKS like the
    # real series. These nine metrics do.
    banner("7  Temporal fidelity metrics")
    period = DATASET_PERIODS[DATASET]
    n_lags = min(max(2 * period, cfg['acf_n_lags']), cfg['n_synthetic'] - 1)

    _, metrics = MetricAggregator(period=period, n_lags=n_lags).compute_all(
        real                = raw_train.cpu().numpy(),
        synthetic           = fixed.detach().cpu().numpy(),
        real_mse            = real_mse,
        transfer_mse        = fixed_mse,
        dataset             = DATASET,
        model               = MODEL,
        distillation_method = 'phase_aware_mtt' if USE_PHASE_AWARE else 'mtt',
        n_distill_steps     = cfg['n_distill_steps'],
    )
    for key in ('fft_distance', 'freq_rank_error', 'peak_mag_ratio',
                'acf_short', 'acf_long', 'trend_error', 'variance_diff',
                'cross_corr_error'):
        if key in metrics and metrics[key] is not None:
            print(f"  {key:<18}: {metrics[key]:.4f}")

    # ── 8. PREDICT the optimal mixing ratio — no sweep ───────────────────────
    # Pure synthetic is rarely optimal; mixing a little real data back in beats
    # both extremes. The question is HOW MUCH, and the expensive answer is to
    # train a probe at every candidate ratio and read off the minimum.
    #
    # predict_r_star() skips that entirely. It scores four cheap properties of
    # the data you already have — how badly distillation degraded utility, how
    # far the synthetic distribution drifted, how much temporal structure was
    # lost, and how sensitive this architecture is — and maps their weighted sum
    # to r*. Every input below was already computed by stage 6, so this costs
    # no extra training at all.
    banner("8  Predict optimal mixing ratio (no sweep)")
    predicted_pct = predict_r_star(
        synth_mse  = fixed_mse,
        real_mse   = real_mse,
        model_name = MODEL,
        real_train = raw_train.cpu().numpy(),
        synthetic  = fixed.detach().cpu().numpy(),
        # Per-channel scoring matters once a dataset has many channels
        # (weather has 21); for the 7-channel ETT family the global variant
        # is equivalent and cheaper.
        channel_aware = in_features > 8,
        verbose       = True,      # prints the four measures and the mapping
    )
    print(f"\n  -> predicted r* = {predicted_pct}% real "
          f"({100 - predicted_pct}% synthetic retained)")

    # Build the final dataset straight from the prediction.
    hybrid = BaseHybridEvaluator(
        anchor_selector     = RandomAnchorSelector(),
        seq_len             = seq_len,
        batch_size          = cfg['eval_batch_size'],
        device              = device,
        eval_max_epochs     = cfg['eval_max_epochs'],
        early_stop_patience = cfg['early_stop_patience'],
        eval_lr             = cfg['eval_lr'],
    )
    final_set = hybrid.hybridmixture(
        synthetic_data  = fixed,
        real_train_data = raw_train,
        real_ratio      = predicted_pct / 100.0,
        window_size     = window_size,
    )
    print(f"  final dataset: {len(final_set)} windows "
          f"({predicted_pct}% real + {100 - predicted_pct}% synthetic)")

    # ── 9. OPTIONAL: verify the prediction against a measured sweep ──────────
    # This is the expensive ground truth the predictor exists to avoid — one
    # probe trained per ratio. Run it only to validate the predictor (which is
    # what the thesis needs); a normal user of the library stops at stage 8.
    measured_pct = measured_mse = None
    if VALIDATE_WITH_SWEEP:
        banner("9  Validate - measured ratio sweep (expensive oracle)")
        mix_results = hybrid.evaluate_mixing(
            synthetic_data   = fixed,
            real_train_data  = raw_train,
            real_test_loader = DataLoader(test_windows,
                                          batch_size=cfg['eval_batch_size'], shuffle=False),
            model_fn         = make_model,
            window_size      = window_size,
            mixing_ratios    = MIXING_RATIOS,
            real_val_data    = raw_val,
        )

        points = sorted((int(k.split('_')[1]), v['MSE']) for k, v in mix_results.items())
        print(f"\n  {'real %':>7}  {'test MSE':>10}")
        print("  " + "-" * 21)
        for ratio, mse in points:
            print(f"  {ratio:>7}  {mse:>10.6f}"
                  + ("" if np.isfinite(mse) else "  <- diverged"))

        # A probe can diverge to NaN; the interpolator needs finite values, and
        # the remaining points still describe the trade-off curve.
        finite = [(r, m) for r, m in points if np.isfinite(m)]
        if len(finite) < 2:
            raise RuntimeError("Too few finite MSE points — most probes diverged.")

        best         = find_optimal_ratio([r for r, _ in finite], [m for _, m in finite])
        measured_pct = int(best['nearest_tested_ratio'])
        measured_mse = mix_results[f'hybrid_{measured_pct}']['MSE']
        print(f"\n  measured r*  : {best['r_star']:.1f}%  "
              f"(nearest tested {measured_pct}%)")
        print(f"  predicted r* : {predicted_pct}%")
        print(f"  gap          : {abs(predicted_pct - best['r_star']):.1f} "
              f"percentage points")

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("SUMMARY")
    print(f"  dataset / model / init : {DATASET} / {MODEL} / {INITIALIZER}")
    print(f"  matching               : "
          f"{'phase_aware_mtt' if USE_PHASE_AWARE else 'mtt'}")
    if phase_boundary is not None:
        print(f"  phase boundary T+      : {phase_boundary}  "
              f"(detector cfg for {MODEL})")
    print(f"  real-data MSE          : {real_mse:.6f}   <- lower bound")
    print(f"  synthetic-only MSE     : {synthetic_mse:.6f}")
    print(f"  after post-fix (a={ALPHA}) : {fixed_mse:.6f}")
    print(f"  PREDICTED r*           : {predicted_pct}% real   <- no sweep needed")
    if measured_pct is not None:
        print(f"  measured  r*           : {measured_pct}% real   "
              f"(oracle, {len(MIXING_RATIOS)} probe trainings)")
        print(f"  hybrid MSE @ measured  : {measured_mse:.6f}")
    print(f"  final dataset          : {len(final_set)} windows "
          f"({predicted_pct}% real + {100 - predicted_pct}% synthetic)")
    print()


if __name__ == '__main__':
    main()
