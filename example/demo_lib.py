"""
ts_distill — library showcase.

Feed in a dataset, run the four core stages, save both outputs.

    initialise -> distil -> post-fix -> hybrid mix

Writes to example/results/:
    synthetic_<dataset>_<model>_<init>_a<alpha>_seed<n>.csv   the distilled block
    demo_results.csv                                          one row per run

Run:  python example/demo_lib.py     (edit SETTINGS below)
"""

import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import path: use the in-repo src/ so this runs without `pip install`.
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
# Ratio predictor: not in the top-level namespace, it is a research-stage API.
from ts_distill.evaluation.hybrid_evaluation.ratio_predictor import predict_r_star


# ── SETTINGS ─────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).resolve().parent   # folder holding the benchmark CSVs
DATASET     = 'ETTh1'                            # any key in DATASET_CONFIGS
MODEL       = 'DLinear'                          # MLP | DLinear | CNN | LSTM
INITIALIZER = 'random'                           # random | geometry | uncertainty
ALPHA       = 0                                # post-fix strength, 0 disables
SEED        = 42                                 # fixes every random draw below

# Phase-aware: off = match weights throughout; on = match outputs after T+.
USE_PHASE_AWARE          = False
PHASE_BOUNDARY_OVERRIDES = {}      # tweak one detector knob, e.g. {'burn_in_epochs': 10}

# Validation: the sweep measures the true best ratio, to check the prediction.
# Costs one probe training per ratio, so it is the slow part of this script.
VALIDATE_WITH_SWEEP = True
MIXING_RATIOS       = (0.0, 0.1, 0.2, 0.3)  # only used when validating

# ── Outputs ──────────────────────────────────────────────────────────────────
OUT_DIR      = Path(__file__).resolve().parent / 'results'
RESULTS_CSV  = OUT_DIR / 'demo_results.csv'   # one row per run, appended
SAVE_ORIGINAL_UNITS = True                    # un-scale the synthetic before saving

# Hyperparameters: these override the library defaults for this run.
OVERRIDES = {
    'expert_epochs':       80,
    'n_distill_steps':    300,
    'synthetic_lr':       5.0,   # must be large or the synthetic data barely moves
    'student_lr':        0.01,
    'student_steps':       20,
    'eval_max_epochs':    300,
    'early_stop_patience': 10,
}

# Initializer choices: all three pick ONE block, differing only in which one.
INITIALIZERS = {
    'random':      RandomSampleInitializer,      # random position
    'geometry':    GeometrySequenceInitializer,  # most typical (medoid)
    'uncertainty': UncertaintySampleInitializer, # most volatile
}


def banner(text):
    """Print a section header."""
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def save_synthetic(block, channel_names, mean, std, path):
    """Write the distilled block to CSV, one row per synthetic timestep."""
    arr = block.detach().cpu().numpy()
    # Un-scale: the pipeline works in standardised space, so multiply the
    # training mean/std back in to get readable values in the dataset's units.
    if SAVE_ORIGINAL_UNITS:
        arr = arr * std + mean
    df = pd.DataFrame(arr, columns=channel_names)
    df.insert(0, 'step', np.arange(len(arr)))    # step column = position in time
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        # Locked file: same fallback as append_result, so a long run is not lost.
        path = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(path, index=False)
        print(f"  [warn] target was locked - wrote {path.name} instead")
    return path


def append_result(row, path, key_cols):
    """Append one result row; a re-run with the same key replaces its old row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # Back-compat: a CSV written by an older version may lack a key column.
        for k in key_cols:
            if k not in old.columns:
                old[k] = ''
        # De-duplicate: drop any existing row with the same config, keep the rest.
        key  = tuple(str(row[k]) for k in key_cols)
        keep = old[key_cols].astype(str).apply(tuple, axis=1) != key
        new  = pd.concat([old[keep], new], ignore_index=True)
    try:
        new.to_csv(path, index=False)
    except PermissionError:
        # Locked file: usually the CSV is open in Excel. Write beside it rather
        # than throwing away a run that took minutes to produce.
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        new.to_csv(alt, index=False)
        print(f"  [warn] {path.name} is locked (open in Excel?) - wrote {alt.name} instead")
        return alt
    return path


def main():
    configure_logging('INFO')        # library is silent until asked
    banner(f"ts_distill v{__version__}  -  library showcase")

    # Guard: fail early on a typo rather than deep inside the pipeline.
    if INITIALIZER not in INITIALIZERS:
        raise ValueError(f"INITIALIZER must be one of {sorted(INITIALIZERS)}")

    cfg    = {**DEFAULT_CONFIG, **OVERRIDES}     # library defaults + our overrides
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Seed all three generators. torch alone is NOT enough: the expert recorder
    # picks checkpoint pairs with Python's `random`, so leaving it unseeded makes
    # the distilled block differ on every run even at a fixed SEED.
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"  device={device}  dataset={DATASET}  model={MODEL}  "
          f"init={INITIALIZER}  alpha={ALPHA}  seed={SEED}")

    # Shapes: a dataset may override the global look-back / horizon / channels.
    dataset_cfg = DATASET_CONFIGS[DATASET]
    seq_len     = dataset_cfg.get('seq_len',     cfg['seq_len'])     # steps model sees
    pred_len    = dataset_cfg.get('pred_len',    cfg['pred_len'])    # steps it predicts
    in_features = dataset_cfg.get('in_features', cfg['in_features']) # channels
    window_size = seq_len + pred_len                                 # one training sample

    def make_model():
        """Build a fresh model. Called for the expert, the students and the probes."""
        return create_model(MODEL, seq_len, pred_len, in_features, MODEL_CONFIGS[MODEL])

    # ── 1. Load + split ──────────────────────────────────────────────────────
    banner("1  Load + split")
    csv_path = resolve_csv_path(DATASET, DATA_DIR)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Point DATA_DIR at your CSVs.")

    df_raw        = CSVDataLoader().load_data(str(csv_path))
    channel_names = list(df_raw.columns[1:])          # column 0 is the date
    values        = df_raw.iloc[:, 1:].values.astype(np.float32)

    # Split first: we need the train boundary before we can scale anything.
    tr_s, tr_e, va_s, va_e, te_s, te_e = get_data_splits(
        values=values, window_size=window_size, seq_len=seq_len, dataset_cfg=dataset_cfg,
    )

    # Scale second, using TRAIN rows only. Using the whole file would leak test
    # statistics into training and make the results look better than they are.
    mean, std = values[tr_s:tr_e].mean(axis=0), values[tr_s:tr_e].std(axis=0)
    std[std == 0] = 1.0                               # guard a constant channel
    data = (values - mean) / std

    # Window third: cut each split into overlapping training samples.
    train_windows = torch.tensor(make_windows(data[tr_s:tr_e], window_size), dtype=torch.float32)
    val_windows   = torch.tensor(make_windows(data[va_s:va_e], window_size), dtype=torch.float32)
    test_windows  = torch.tensor(make_windows(data[te_s:te_e], window_size), dtype=torch.float32)

    # Un-windowed copies: the distiller and the metrics need continuous series.
    raw_train = torch.tensor(data[tr_s:tr_e], dtype=torch.float32)
    raw_val   = torch.tensor(data[va_s:va_e], dtype=torch.float32)

    print(f"  loaded {values.shape[0]} rows x {values.shape[1]} channels")
    print(f"  windows -> train {len(train_windows)}, val {len(val_windows)}, test {len(test_windows)}")

    # ── 2. Train the expert, recording its trajectory ────────────────────────
    # Distillation copies HOW the expert learned, so the path has to be recorded
    # while training happens. It cannot be recovered from the finished model.
    banner(f"2  Train expert ({cfg['expert_epochs']} epochs)")
    val_loader = DataLoader(val_windows, batch_size=cfg['eval_batch_size'], shuffle=False)

    recorder       = SimpleRecorder(record_every=1)   # saves weights every epoch
    expert         = make_model()
    expert_trainer = Trainer(
        model=expert,
        # SGD, not Adam: it gives a smooth path that is easier to match.
        optimizer=torch.optim.SGD(expert.parameters(),
                                  lr=cfg['expert_lr'], momentum=cfg['expert_momentum']),
        criterion=torch.nn.MSELoss(), device=device, seq_len=seq_len,
    )
    callbacks = [recorder, SimpleCallback()]

    # Val curve: only needed to locate T+, so only recorded in phase-aware mode.
    val_recorder = None
    if USE_PHASE_AWARE:
        def rng_safe_val_eval():
            # RNG guard: iterating a loader draws a random number, which would
            # shift every later draw and change the synthetic data. Save and
            # restore the state so results match the non-phase-aware run.
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

    # Phase boundary T+: the epoch where the expert stops improving. Detector
    # settings differ per architecture, so they come from the library.
    phase_boundary = None
    if USE_PHASE_AWARE:
        boundary_cfg = {**phase_boundary_config(MODEL), **PHASE_BOUNDARY_OVERRIDES}
        phase_boundary = ValLossPlateauDetector(**boundary_cfg).detect(
            val_recorder.val_losses)['boundary_epoch']
        print(f"  phase boundary T+ = {phase_boundary}  (cfg for {MODEL}: {boundary_cfg})")

    # ── 3. Initialise ────────────────────────────────────────────────────────
    # Starting point: one continuous block of real data that distillation will
    # reshape. Which block affects how much results swing between seeds.
    banner(f"3  Initialise ({INITIALIZER})")
    initializer    = INITIALIZERS[INITIALIZER]()
    synthetic_init = initializer.initialize_sequence(raw_train, cfg['n_synthetic'])

    # ── 4. Distil ────────────────────────────────────────────────────────────
    # The data is what gets optimised here, not a model. Students are trained a
    # few steps on the synthetic block, compared against the expert, then thrown
    # away — the gradient flows back into the block itself.
    banner(f"4  Distil ({cfg['n_distill_steps']} steps, "
           f"{'phase-aware' if USE_PHASE_AWARE else 'standard'} MTT)")
    distiller_kwargs = dict(
        initializer=initializer, matcher=MSEMatcher(), model_factory=make_model,
        expert_recorder=recorder,
        expert_epochs=cfg['trajectory_gap'],        # gap between matched checkpoints
        syn_batch_size=cfg['batch_size'],
        synthetic_lr=cfg['synthetic_lr'],           # step size for the DATA
        student_lr=cfg['student_lr'],               # step size inside the inner loop
        student_steps=cfg['student_steps'],         # inner-loop length; main cost dial
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
    # Does it work? Train a fresh model on the distilled data, score on real data.
    banner("5  Utility - probe trained on synthetic, tested on real")
    evaluator = Evaluator(seq_len=seq_len, batch_size=cfg['eval_batch_size'])

    def probe_mse(training_windows, seed):
        """Train a fresh probe on the given windows, return its real-test MSE."""
        torch.manual_seed(seed)     # same start every call, so runs are comparable
        model = make_model()
        Trainer(
            # Adam here: a probe only needs to converge, its path does not matter.
            model=model, optimizer=torch.optim.Adam(model.parameters(), lr=cfg['eval_lr']),
            criterion=torch.nn.MSELoss(), device=device, seq_len=seq_len,
        ).fit(
            dataloader=DataLoader(training_windows, batch_size=cfg['eval_batch_size'], shuffle=True),
            epochs=cfg['eval_max_epochs'], val_loader=val_loader,
            patience=cfg['early_stop_patience'],    # stop early, restore best weights
        )
        return evaluator.test_on_real(model, test_windows.to(device))['MSE']

    syn_windows   = torch.tensor(
        make_windows(synthetic.detach().cpu().numpy(), window_size), dtype=torch.float32)
    synthetic_mse = probe_mse(syn_windows, seed=1)

    # Real baseline: reuse the already-trained expert, so this number never
    # changes with distillation settings. It is the lower bound to beat.
    real_mse = evaluator.test_on_real(expert, test_windows.to(device))['MSE']

    print(f"  real-data MSE      : {real_mse:.6f}   <- lower bound")
    print(f"  synthetic-data MSE : {synthetic_mse:.6f}")
    print(f"  performance kept   : {real_mse / synthetic_mse * 100:.1f}%")
    print(f"  compression        : {cfg['n_synthetic']}/{len(train_windows)} "
          f"({cfg['n_synthetic'] / len(train_windows) * 100:.2f}%)")

    # ── 6. Post-fix ──────────────────────────────────────────────────────────
    # Repair rhythms: distillation wrecks the data's cycles. This pushes them
    # back toward the real ones while keeping the shape distillation learned.
    banner(f"6  Post-fix (FFT amplitude, alpha={ALPHA})")
    if ALPHA > 0:
        fixed  = torch.tensor(FFTAmplitudePostFix(alpha=ALPHA).apply(synthetic, raw_train),
                              dtype=torch.float32)
        # Before/after: this distance should drop by exactly alpha.
        before = FFTAmplitudePostFix.fft_distance(synthetic, raw_train)
        after  = FFTAmplitudePostFix.fft_distance(fixed,     raw_train)
        # Re-probe: repairing rhythms usually costs a little accuracy.
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
    # Is it still a time series? Utility alone cannot answer that.
    banner("7  Temporal fidelity metrics")
    period = DATASET_PERIODS[DATASET]                 # e.g. 24 = one day, hourly data
    n_lags = min(max(2 * period, cfg['acf_n_lags']), cfg['n_synthetic'] - 1)
    _, metrics = MetricAggregator(period=period, n_lags=n_lags).compute_all(
        real=raw_train.cpu().numpy(), synthetic=fixed.detach().cpu().numpy(),
        real_mse=real_mse, transfer_mse=fixed_mse,    # utility is passed in, not measured
        dataset=DATASET, model=MODEL,
        distillation_method='phase_aware_mtt' if USE_PHASE_AWARE else 'mtt',
        n_distill_steps=cfg['n_distill_steps'],
    )
    for key in ('fft_distance', 'freq_rank_error', 'peak_mag_ratio', 'acf_short',
                'acf_long', 'trend_error', 'variance_diff', 'cross_corr_error'):
        if metrics.get(key) is not None:
            print(f"  {key:<18}: {metrics[key]:.4f}")

    # ── 8. Predict r* and build the final dataset ────────────────────────────
    # How much real data to add back? Predicted from four cheap properties of
    # data we already have — no probe training, unlike the sweep in step 9.
    banner("8  Predict optimal mixing ratio (no sweep)")
    predicted_pct = predict_r_star(
        synth_mse=fixed_mse, real_mse=real_mse, model_name=MODEL,
        real_train=raw_train.cpu().numpy(), synthetic=fixed.detach().cpu().numpy(),
        channel_aware=in_features > 8,    # per-channel scoring pays off past ~8 channels
        verbose=True,                     # prints the four measures and the mapping
    )
    print(f"\n  -> predicted r* = {predicted_pct}% real")

    hybrid = BaseHybridEvaluator(
        anchor_selector=RandomAnchorSelector(),   # picks WHICH real windows to add
        seq_len=seq_len,
        batch_size=cfg['eval_batch_size'], device=device,
        eval_max_epochs=cfg['eval_max_epochs'],
        early_stop_patience=cfg['early_stop_patience'], eval_lr=cfg['eval_lr'],
    )
    final_set = hybrid.hybridmixture(
        synthetic_data=fixed, real_train_data=raw_train,
        real_ratio=predicted_pct / 100.0, window_size=window_size,
    )

    # Actual counts: every synthetic window is kept and real ones are added on
    # top, so the mixture is far more real-heavy than the ratio knob suggests.
    n_syn_final  = len(make_windows(fixed.detach().cpu().numpy(), window_size))
    n_real_final = len(final_set) - n_syn_final
    real_pct     = n_real_final / len(final_set) * 100
    print(f"  final dataset: {len(final_set)} windows = {n_real_final} real + "
          f"{n_syn_final} synthetic ({real_pct:.1f}% real by count; knob r={predicted_pct}%)")

    # ── 9. Validate the prediction against a measured sweep (optional) ───────
    # Ground truth: try every ratio and retrain. Expensive — this is what the
    # predictor above exists to avoid. Run it only to check the prediction.
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

        # NaN filter: a probe can diverge (CNN especially), and the curve fitter
        # below rejects non-finite values. The remaining points still work.
        finite = [(r, m) for r, m in points if np.isfinite(m)]
        if len(finite) < 2:
            raise RuntimeError("Too few finite MSE points - most probes diverged.")

        best         = find_optimal_ratio([r for r, _ in finite], [m for _, m in finite])
        measured_pct = int(best['nearest_tested_ratio'])
        measured_mse = mix_results[f'hybrid_{measured_pct}']['MSE']
        print(f"\n  measured r*  : {best['r_star']:.1f}% (nearest tested {measured_pct}%)")
        print(f"  predicted r* : {predicted_pct}%")
        print(f"  gap          : {abs(predicted_pct - best['r_star']):.1f} percentage points")

    # ── 10. Save outputs ─────────────────────────────────────────────────────
    banner("10  Save outputs")
    # Tag: encodes the config so sweeping settings never overwrites a result.
    tag = f"{DATASET}_{MODEL}_{INITIALIZER}_a{ALPHA}_seed{SEED}"

    # File 1 — the distilled block itself, the artifact this pipeline produces.
    syn_path = save_synthetic(fixed, channel_names, mean, std,
                              OUT_DIR / f'synthetic_{tag}.csv')
    print(f"  synthetic block -> {syn_path}")
    print(f"    {cfg['n_synthetic']} timesteps x {in_features} channels"
          f"{' (original units)' if SAVE_ORIGINAL_UNITS else ' (standardised)'}")

    # File 2 — one row summarising this run, appended to a growing table.
    row = {
        # what was run
        'run_at':          datetime.now().isoformat(timespec='seconds'),
        'dataset':         DATASET,
        'model':           MODEL,
        'initializer':     INITIALIZER,
        'matching':        'phase_aware_mtt' if USE_PHASE_AWARE else 'mtt',
        'alpha':           ALPHA,
        'seed':            SEED,
        'phase_boundary':  phase_boundary if phase_boundary is not None else '',
        'expert_epochs':   cfg['expert_epochs'],
        'n_distill_steps': cfg['n_distill_steps'],
        'n_synthetic':     cfg['n_synthetic'],
        # utility — does it train a good model?
        'real_mse':        round(real_mse, 6),
        'synthetic_mse':   round(synthetic_mse, 6),
        'postfix_mse':     round(fixed_mse, 6),
        'performance_kept_pct': round(real_mse / fixed_mse * 100, 2),
        'compression_pct': round(cfg['n_synthetic'] / len(train_windows) * 100, 4),
        # mixing — predicted vs measured, the pair that validates the predictor
        'predicted_r_star': predicted_pct,
        'measured_r_star':  measured_pct if measured_pct is not None else '',
        'r_star_gap_pp':    round(abs(predicted_pct - measured_pct), 2)
                            if measured_pct is not None else '',
        'hybrid_mse':       round(measured_mse, 6) if measured_mse is not None else '',
        'final_windows':    len(final_set),
        'final_real':       n_real_final,
        'final_synthetic':  n_syn_final,
        'synthetic_csv':    syn_path.name,     # links this row to its block
        # fidelity — is it still a time series?
        **{k: round(float(metrics[k]), 6)
           for k in ('fft_distance', 'freq_rank_error', 'peak_mag_ratio', 'acf_short',
                     'acf_long', 'trend_error', 'variance_diff', 'cross_corr_error')
           if metrics.get(k) is not None},
    }
    res_path = append_result(
        row, RESULTS_CSV,
        # Key: these six identify a run. Re-running the same six replaces the row.
        key_cols=['dataset', 'model', 'initializer', 'matching', 'alpha', 'seed'])
    print(f"  results row     -> {res_path}")

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
    print(f"  saved                  : {syn_path.name}  +  {res_path.name}")
    print()


if __name__ == '__main__':
    main()
