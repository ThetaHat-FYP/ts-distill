"""
ts_distill — library showcase.

Full pipeline in one file: load -> train expert -> distil -> post-fix ->
measure -> predict mix ratio -> build the final dataset.

Run:  python example/demo_lib.py     (edit SETTINGS below)
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Use the in-repo src/ so this runs without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from ts_distill import (
    __version__, configure_logging,
    CSVDataLoader, get_data_splits, make_windows, MiniBatchLoader,
    create_model, Trainer, SimpleRecorder, SimpleCallback,
    MSEMatcher, MTTDistiller, PhaseAwareMTTDistiller,
    ValLossPlateauDetector, ValLossRecorderCallback,
    RandomSampleInitializer, GeometrySequenceInitializer, UncertaintySampleInitializer,
    FFTAmplitudePostFix, Evaluator, BaseHybridEvaluator, RandomAnchorSelector,
    find_optimal_ratio, MetricAggregator,
    DEFAULT_CONFIG, DATASET_CONFIGS, DATASET_PERIODS, MODEL_CONFIGS,
    resolve_csv_path, phase_boundary_config,
)
from ts_distill.evaluation.hybrid_evaluation.ratio_predictor import predict_r_star


# ── SETTINGS ─────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).resolve().parent   # folder holding the benchmark CSVs
DATASET     = 'ETTh2'
MODEL       = 'DLinear'                          # MLP | DLinear | CNN | LSTM
INITIALIZER = 'random'                           # random | geometry | uncertainty
ALPHA       = 0.7                                # post-fix strength, 0 disables
SEED        = 42

USE_PHASE_AWARE          = False   # parameter matching early, prediction matching after T+
PHASE_BOUNDARY_OVERRIDES = {}      # e.g. {'burn_in_epochs': 10}

VALIDATE_WITH_SWEEP = True                  # also run the expensive oracle sweep
MIXING_RATIOS       = (0.0, 0.1, 0.2, 0.3)  # only used when validating

OVERRIDES = {
    'expert_epochs':       80,
    'n_distill_steps':    300,
    'synthetic_lr':       5.0,   # must be large or the synthetic data barely moves
    'student_lr':        0.01,
    'student_steps':       20,
    'eval_max_epochs':    300,
    'early_stop_patience': 10,
}

INITIALIZERS = {
    'random':      RandomSampleInitializer,
    'geometry':    GeometrySequenceInitializer,
    'uncertainty': UncertaintySampleInitializer,
}


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def main():
    configure_logging('INFO')
    banner(f"ts_distill v{__version__}  -  library showcase")

    if INITIALIZER not in INITIALIZERS:
        raise ValueError(f"INITIALIZER must be one of {sorted(INITIALIZERS)}")

    cfg    = {**DEFAULT_CONFIG, **OVERRIDES}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    print(f"  device={device}  dataset={DATASET}  model={MODEL}  "
          f"init={INITIALIZER}  alpha={ALPHA}  seed={SEED}")

    dataset_cfg = DATASET_CONFIGS[DATASET]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])
    in_features = dataset_cfg.get('in_features', cfg['in_features'])
    window_size = seq_len + pred_len

    def make_model():
        return create_model(MODEL, seq_len, pred_len, in_features, MODEL_CONFIGS[MODEL])

    # ── 1. Load + split ──────────────────────────────────────────────────────
    banner("1  Load + split")
    csv_path = resolve_csv_path(DATASET, DATA_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Point DATA_DIR at your CSVs.")

    values = CSVDataLoader().load_data(str(csv_path)).iloc[:, 1:].values.astype(np.float32)
    tr_s, tr_e, va_s, va_e, te_s, te_e = get_data_splits(
        values=values, window_size=window_size, seq_len=seq_len, dataset_cfg=dataset_cfg,
    )

    # Standardise on TRAIN statistics only, so test data does not leak in.
    mean, std = values[tr_s:tr_e].mean(axis=0), values[tr_s:tr_e].std(axis=0)
    std[std == 0] = 1.0
    data = (values - mean) / std

    train_windows = torch.tensor(make_windows(data[tr_s:tr_e], window_size), dtype=torch.float32)
    val_windows   = torch.tensor(make_windows(data[va_s:va_e], window_size), dtype=torch.float32)
    test_windows  = torch.tensor(make_windows(data[te_s:te_e], window_size), dtype=torch.float32)
    raw_train     = torch.tensor(data[tr_s:tr_e], dtype=torch.float32)
    raw_val       = torch.tensor(data[va_s:va_e], dtype=torch.float32)
    print(f"  loaded {values.shape[0]} rows x {values.shape[1]} channels")
    print(f"  windows -> train {len(train_windows)}, val {len(val_windows)}, test {len(test_windows)}")

    # ── 2. Train the expert, recording its trajectory ────────────────────────
    banner(f"2  Train expert ({cfg['expert_epochs']} epochs)")
    val_loader = DataLoader(val_windows, batch_size=cfg['eval_batch_size'], shuffle=False)

    recorder       = SimpleRecorder(record_every=1)   # MTT matches this path, not the endpoint
    expert         = make_model()
    expert_trainer = Trainer(
        model=expert,
        optimizer=torch.optim.SGD(expert.parameters(),
                                  lr=cfg['expert_lr'], momentum=cfg['expert_momentum']),
        criterion=torch.nn.MSELoss(), device=device, seq_len=seq_len,
    )
    callbacks = [recorder, SimpleCallback()]

    val_recorder = None
    if USE_PHASE_AWARE:
        # The val pass consumes torch RNG; restore it so results match the non-phase-aware run.
        def rng_safe_val_eval():
            state = torch.get_rng_state()
            try:
                return expert_trainer.eval_epoch(val_loader)
            finally:
                torch.set_rng_state(state)
        val_recorder = ValLossRecorderCallback(eval_fn=rng_safe_val_eval)
        callbacks.append(val_recorder)

    expert_trainer.fit(
        dataloader=MiniBatchLoader(train_windows, batch_size=cfg['batch_size']),
        epochs=cfg['expert_epochs'], callbacks=callbacks,
    )

    # Detector settings are tuned per expert architecture and come from the library.
    phase_boundary = None
    if USE_PHASE_AWARE:
        boundary_cfg = {**phase_boundary_config(MODEL), **PHASE_BOUNDARY_OVERRIDES}
        phase_boundary = ValLossPlateauDetector(**boundary_cfg).detect(
            val_recorder.val_losses)['boundary_epoch']
        print(f"  phase boundary T+ = {phase_boundary}  (cfg for {MODEL}: {boundary_cfg})")

    # ── 3. Initialise ────────────────────────────────────────────────────────
    # Every strategy picks ONE contiguous block; they differ only in which one.
    banner(f"3  Initialise ({INITIALIZER})")
    initializer    = INITIALIZERS[INITIALIZER]()
    synthetic_init = initializer.initialize_sequence(raw_train, cfg['n_synthetic'])

    # ── 4. Distil ────────────────────────────────────────────────────────────
    banner(f"4  Distil ({cfg['n_distill_steps']} steps, "
           f"{'phase-aware' if USE_PHASE_AWARE else 'standard'} MTT)")
    distiller_kwargs = dict(
        initializer=initializer, matcher=MSEMatcher(), model_factory=make_model,
        expert_recorder=recorder, expert_epochs=cfg['trajectory_gap'],
        syn_batch_size=cfg['batch_size'], synthetic_lr=cfg['synthetic_lr'],
        student_lr=cfg['student_lr'], student_steps=cfg['student_steps'],
        snapshot_student_steps=cfg['snapshot_student_steps'],
        seq_len=seq_len, pred_len=pred_len,
    )
    distiller = (PhaseAwareMTTDistiller(**distiller_kwargs, phase_boundary=phase_boundary)
                 if USE_PHASE_AWARE else MTTDistiller(**distiller_kwargs))
    synthetic = distiller.distill(
        synthetic_init=synthetic_init, n_steps=cfg['n_distill_steps'], val_data=val_windows,
    )
    print(f"  distilled shape: {tuple(synthetic.shape)}  (vs {len(train_windows)} real windows)")

    # ── 5. Utility ───────────────────────────────────────────────────────────
    banner("5  Utility - probe trained on synthetic, tested on real")
    evaluator = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size'])

    def probe_mse(training_windows, seed):
        """Train a fresh probe on the given windows, return its real-test MSE."""
        torch.manual_seed(seed)
        model = make_model()
        Trainer(
            model=model, optimizer=torch.optim.Adam(model.parameters(), lr=cfg['eval_lr']),
            criterion=torch.nn.MSELoss(), device=device, seq_len=seq_len,
        ).fit(
            dataloader=DataLoader(training_windows, batch_size=cfg['eval_batch_size'], shuffle=True),
            epochs=cfg['eval_max_epochs'], val_loader=val_loader,
            patience=cfg['early_stop_patience'],
        )
        return evaluator.test_on_real(model, test_windows.to(device))['MSE']

    syn_windows   = torch.tensor(
        make_windows(synthetic.detach().cpu().numpy(), window_size), dtype=torch.float32)
    synthetic_mse = probe_mse(syn_windows, seed=1)
    real_mse      = evaluator.test_on_real(expert, test_windows.to(device))['MSE']

    print(f"  real-data MSE      : {real_mse:.6f}   <- lower bound")
    print(f"  synthetic-data MSE : {synthetic_mse:.6f}")
    print(f"  performance kept   : {real_mse / synthetic_mse * 100:.1f}%")
    print(f"  compression        : {cfg['n_synthetic']}/{len(train_windows)} "
          f"({cfg['n_synthetic'] / len(train_windows) * 100:.2f}%)")

    # ── 6. Post-fix ──────────────────────────────────────────────────────────
    # Keeps the synthetic phase, blends amplitude toward the real spectrum.
    banner(f"6  Post-fix (FFT amplitude, alpha={ALPHA})")
    if ALPHA > 0:
        fixed  = torch.tensor(FFTAmplitudePostFix(alpha=ALPHA).apply(synthetic, raw_train),
                              dtype=torch.float32)
        before = FFTAmplitudePostFix.fft_distance(synthetic, raw_train)
        after  = FFTAmplitudePostFix.fft_distance(fixed,     raw_train)
        fixed_windows = torch.tensor(
            make_windows(fixed.detach().cpu().numpy(), window_size), dtype=torch.float32)
        fixed_mse = probe_mse(fixed_windows, seed=1)
        print(f"  fft_distance: {before:.4f} -> {after:.4f} "
              f"({(1 - after / before) * 100:.1f}% closer to real)")
        print(f"  utility cost: {synthetic_mse:.6f} -> {fixed_mse:.6f} "
              f"({(fixed_mse / synthetic_mse - 1) * 100:+.1f}%)")
    else:
        fixed, fixed_mse = synthetic.detach(), synthetic_mse
        print("  disabled (alpha=0)")

    # ── 7. Temporal fidelity ─────────────────────────────────────────────────
    banner("7  Temporal fidelity metrics")
    period = DATASET_PERIODS[DATASET]
    n_lags = min(max(2 * period, cfg['acf_n_lags']), cfg['n_synthetic'] - 1)
    _, metrics = MetricAggregator(period=period, n_lags=n_lags).compute_all(
        real=raw_train.cpu().numpy(), synthetic=fixed.detach().cpu().numpy(),
        real_mse=real_mse, transfer_mse=fixed_mse,
        dataset=DATASET, model=MODEL,
        distillation_method='phase_aware_mtt' if USE_PHASE_AWARE else 'mtt',
        n_distill_steps=cfg['n_distill_steps'],
    )
    for key in ('fft_distance', 'freq_rank_error', 'peak_mag_ratio', 'acf_short',
                'acf_long', 'trend_error', 'variance_diff', 'cross_corr_error'):
        if metrics.get(key) is not None:
            print(f"  {key:<18}: {metrics[key]:.4f}")

    # ── 8. Predict r* and build the final dataset ────────────────────────────
    # Four cheap measures of data we already have -> r*, with no probe training.
    banner("8  Predict optimal mixing ratio (no sweep)")
    predicted_pct = predict_r_star(
        synth_mse=fixed_mse, real_mse=real_mse, model_name=MODEL,
        real_train=raw_train.cpu().numpy(), synthetic=fixed.detach().cpu().numpy(),
        channel_aware=in_features > 8, verbose=True,
    )
    print(f"\n  -> predicted r* = {predicted_pct}% real")

    hybrid = BaseHybridEvaluator(
        anchor_selector=RandomAnchorSelector(), seq_len=seq_len,
        batch_size=cfg['eval_batch_size'], device=device,
        eval_max_epochs=cfg['eval_max_epochs'],
        early_stop_patience=cfg['early_stop_patience'], eval_lr=cfg['eval_lr'],
    )
    final_set = hybrid.hybridmixture(
        synthetic_data=fixed, real_train_data=raw_train,
        real_ratio=predicted_pct / 100.0, window_size=window_size,
    )

    # Report actual counts: all synthetic windows are kept, real ones are added on top.
    n_syn_final  = len(make_windows(fixed.detach().cpu().numpy(), window_size))
    n_real_final = len(final_set) - n_syn_final
    real_pct     = n_real_final / len(final_set) * 100
    print(f"  final dataset: {len(final_set)} windows = {n_real_final} real + "
          f"{n_syn_final} synthetic ({real_pct:.1f}% real by count; knob r={predicted_pct}%)")

    # ── 9. Validate the prediction against a measured sweep (optional) ───────
    measured_pct = measured_mse = None
    if VALIDATE_WITH_SWEEP:
        banner("9  Validate - measured ratio sweep (expensive oracle)")
        mix_results = hybrid.evaluate_mixing(
            synthetic_data=fixed, real_train_data=raw_train,
            real_test_loader=DataLoader(test_windows, batch_size=cfg['eval_batch_size'],
                                        shuffle=False),
            model_fn=make_model, window_size=window_size,
            mixing_ratios=MIXING_RATIOS, real_val_data=raw_val,
        )
        points = sorted((int(k.split('_')[1]), v['MSE']) for k, v in mix_results.items())
        print(f"\n  {'real %':>7}  {'test MSE':>10}")
        print("  " + "-" * 21)
        for ratio, mse in points:
            print(f"  {ratio:>7}  {mse:>10.6f}" + ("" if np.isfinite(mse) else "  <- diverged"))

        # A diverged probe gives NaN, which the interpolator cannot accept.
        finite = [(r, m) for r, m in points if np.isfinite(m)]
        if len(finite) < 2:
            raise RuntimeError("Too few finite MSE points - most probes diverged.")

        best         = find_optimal_ratio([r for r, _ in finite], [m for _, m in finite])
        measured_pct = int(best['nearest_tested_ratio'])
        measured_mse = mix_results[f'hybrid_{measured_pct}']['MSE']
        print(f"\n  measured r*  : {best['r_star']:.1f}% (nearest tested {measured_pct}%)")
        print(f"  predicted r* : {predicted_pct}%")
        print(f"  gap          : {abs(predicted_pct - best['r_star']):.1f} percentage points")

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("SUMMARY")
    print(f"  dataset / model / init : {DATASET} / {MODEL} / {INITIALIZER}")
    print(f"  matching               : {'phase_aware_mtt' if USE_PHASE_AWARE else 'mtt'}")
    if phase_boundary is not None:
        print(f"  phase boundary T+      : {phase_boundary}")
    print(f"  real-data MSE          : {real_mse:.6f}   <- lower bound")
    print(f"  synthetic-only MSE     : {synthetic_mse:.6f}")
    print(f"  after post-fix (a={ALPHA}) : {fixed_mse:.6f}")
    print(f"  PREDICTED r*           : {predicted_pct}% real   <- no sweep needed")
    if measured_pct is not None:
        print(f"  measured  r*           : {measured_pct}% real   (oracle)")
        print(f"  hybrid MSE @ measured  : {measured_mse:.6f}")
    print(f"  final dataset          : {len(final_set)} windows = {n_real_final} real + "
          f"{n_syn_final} synthetic ({real_pct:.1f}% real by count)")
    print()


if __name__ == '__main__':
    main()
