"""
ratio_predictor.py  (FIXED)
============================
Predicts the optimal mixing ratio r* for any new dataset-architecture
combination without running a full sweep.

Usage:
    from ratio_predictor import predict_r_star

    r_star = predict_r_star(
        synth_mse   = 0.493,
        real_mse    = 0.397,
        model_name  = 'DLinear',
        real_train  = raw_train_data,   # numpy array shape (N, features)
        synthetic   = synthetic_seq,    # numpy array shape (M, features)
    )
    print(f"Recommended r* = {r_star}%")

Standalone demo (annotates existing mixing curves with predicted r*):
    python -m example.experiments.ratio_predictor
    (run from anywhere — paths are resolved relative to this file, not cwd)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf as compute_acf

# Output/input data stays under example/experiments/ even though this module
# now lives in src/. __file__ is src/ts_distill/evaluation/hybrid_evaluation/
# ratio_predictor.py, so 4 parents up is the project root.
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[4] / 'example' / 'experiments'

# ─────────────────────────────────────────────────────────────────
# REGRESSION-BASED PREDICTOR
# ─────────────────────────────────────────────────────────────────

_REGRESSION_MODEL_CACHE: dict = {}
_DEFAULT_REGRESSION_TRAINING_PATH = _EXPERIMENTS_DIR / 'results' / 'h2_mixing_strategy_results.csv'


def _build_regression_features(
    synth_mse: float,
    real_mse: float,
    model_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Create a feature frame for the regression-based ratio predictor."""
    error_score = measure_error_ratio(synth_mse, real_mse)
    mse_gap_ratio = (max(real_mse, 1e-8) - max(synth_mse, 1e-8)) / max(real_mse, 1e-8)

    row = pd.DataFrame([
        {
            'synth_mse': float(synth_mse),
            'real_mse': float(real_mse),
            'error_ratio_score': float(error_score),
            'mse_gap_ratio': float(max(0.0, min(1.0, mse_gap_ratio))),
            'model': model_name,
            'dataset': dataset_name,
        }
    ])
    return row


def _fit_ratio_regression_model(training_data_path: Path | str | None = None):
    """Fit a simple linear regression model from historical ratio results."""
    path = Path(training_data_path or _DEFAULT_REGRESSION_TRAINING_PATH)
    key = str(path)

    if not path.exists():
        return None, None

    mtime = path.stat().st_mtime
    cached = _REGRESSION_MODEL_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1], cached[2]

    df = pd.read_csv(path)
    target_column = None
    for candidate in ('best_ratio', 'h2_r_star_pct', 'h2_r_star_discrete'):
        if candidate in df.columns:
            target_column = candidate
            break

    if target_column is None:
        return None, None

    training_df = df[['synth_mse', 'real_mse', 'dataset', 'model', target_column]].copy()
    training_df = training_df.dropna()
    if len(training_df) < 5:
        return None, None

    training_df['error_ratio_score'] = training_df.apply(
        lambda row: measure_error_ratio(float(row['synth_mse']), float(row['real_mse'])),
        axis=1,
    )
    training_df['mse_gap_ratio'] = (
        training_df['real_mse'] - training_df['synth_mse']
    ).clip(lower=0.0) / training_df['real_mse'].replace(0, np.nan).fillna(1.0)

    feature_df = pd.get_dummies(
        training_df[['dataset', 'model']], prefix=['dataset', 'model']
    )
    feature_df['synth_mse'] = training_df['synth_mse'].astype(float)
    feature_df['real_mse'] = training_df['real_mse'].astype(float)
    feature_df['error_ratio_score'] = training_df['error_ratio_score'].astype(float)
    feature_df['mse_gap_ratio'] = training_df['mse_gap_ratio'].astype(float)

    model = LinearRegression()
    model.fit(feature_df, training_df[target_column].astype(float))
    _REGRESSION_MODEL_CACHE[key] = (mtime, model, feature_df.columns.tolist())
    return model, feature_df.columns.tolist()


def predict_r_star_regression(
    synth_mse: float,
    real_mse: float,
    model_name: str,
    dataset_name: str,
    real_train: np.ndarray,
    synthetic: np.ndarray,
    training_data_path: Path | str | None = None,
    verbose: bool = True,
) -> int:
    """Predict the ratio using a simple regression model trained from prior sweep data."""
    model, feature_columns = _fit_ratio_regression_model(training_data_path)
    if model is None or feature_columns is None:
        if verbose:
            print('  [Regression] No suitable training data found; falling back to heuristic predictor.')
        return predict_r_star(
            synth_mse=synth_mse,
            real_mse=real_mse,
            model_name=model_name,
            real_train=real_train,
            synthetic=synthetic,
            verbose=False,
        )

    input_df = _build_regression_features(synth_mse, real_mse, model_name, dataset_name)

    feature_df = pd.get_dummies(
        input_df[['dataset', 'model']], prefix=['dataset', 'model']
    )
    feature_df['synth_mse'] = input_df['synth_mse'].astype(float)
    feature_df['real_mse'] = input_df['real_mse'].astype(float)
    feature_df['error_ratio_score'] = input_df['error_ratio_score'].astype(float)
    feature_df['mse_gap_ratio'] = input_df['mse_gap_ratio'].astype(float)

    input_df = feature_df.reindex(columns=feature_columns, fill_value=0.0)

    predicted_ratio = float(model.predict(input_df)[0])
    predicted_ratio = int(round(max(1, min(100, predicted_ratio))))

    if verbose:
        print(f'  [Regression] Predicted ratio = {predicted_ratio}%')

    return predicted_ratio


# ─────────────────────────────────────────────────────────────────
# MEASURE A — Distillation Error Ratio
# ─────────────────────────────────────────────────────────────────

def measure_error_ratio(synth_mse: float, real_mse: float) -> float:
    """
    How badly did distillation fail?

    Returns a score in [0, 1] where:
        0.0 = perfect distillation
        1.0 = catastrophic failure (error ratio >= 5x)
    """
    if real_mse <= 0:
        return 0.0

    error_ratio = synth_mse / real_mse

    # Normalise: cap contribution at error_ratio = 5x
    # Beyond 5x the distillation has completely failed
    # and we already know we need maximum real data
    score = min(error_ratio / 5.0, 1.0)

    return round(score, 4)


# ─────────────────────────────────────────────────────────────────
# MEASURE B — Distribution Divergence
# ─────────────────────────────────────────────────────────────────

def measure_divergence(
    real_train: np.ndarray,
    synthetic:  np.ndarray,
) -> float:
    """
    Did the synthetic data drift away from the real data distribution?

    Compares mean and standard deviation PER CHANNEL, then averages —
    this is a global (non-channel-specific) comparison; see
    measure_divergence_per_channel() below if you need channel-level
    detail for high-dimensional multivariate datasets.

    Returns a score in [0, 1] where:
        0.0 = synthetic perfectly matches real distribution
        1.0 = synthetic completely drifted away
    """
    real_mean = real_train.mean()
    real_std  = real_train.std() + 1e-8   # avoid division by zero

    syn_mean  = synthetic.mean()
    syn_std   = synthetic.std()

    # How much did the mean shift (normalised by real std)
    mean_shift = abs(syn_mean - real_mean) / real_std

    # How much did the spread change
    std_shift  = abs(1.0 - (syn_std / (real_std + 1e-8)))

    # Combined divergence score
    raw_divergence = mean_shift + std_shift

    # Normalise: cap at 2.0 (beyond that it is completely off)
    score = min(raw_divergence / 2.0, 1.0)

    return round(score, 4)


def measure_divergence_per_channel(
    real_train: np.ndarray,
    synthetic:  np.ndarray,
) -> float:
    """
    Channel-aware variant of measure_divergence().

    On multivariate datasets (e.g. 'weather' with 21 channels), a single
    global mean/std can mask one channel drifting badly while the
    aggregate looks fine. This computes the divergence score per channel
    and returns the WORST (max) channel score, which is more conservative
    and catches localized drift the global version misses.

    Falls back to the global measure_divergence() if input is 1-D.
    """
    if real_train.ndim == 1 or real_train.shape[-1] == 1:
        return measure_divergence(real_train, synthetic)

    n_channels = real_train.shape[-1]
    channel_scores = []
    for c in range(n_channels):
        real_c = real_train[:, c]
        syn_c  = synthetic[:, c]
        real_std_c = real_c.std() + 1e-8
        mean_shift = abs(syn_c.mean() - real_c.mean()) / real_std_c
        std_shift  = abs(1.0 - (syn_c.std() / real_std_c))
        channel_scores.append(min((mean_shift + std_shift) / 2.0, 1.0))

    return round(float(np.max(channel_scores)), 4)


# ─────────────────────────────────────────────────────────────────
# MEASURE C — Temporal Structure Preservation
# ─────────────────────────────────────────────────────────────────

def measure_structure_loss(
    real_train: np.ndarray,
    synthetic:  np.ndarray,
    n_lags:     int = 24,
) -> float:
    """
    Did the synthetic data preserve the temporal autocorrelation
    structure of the real data?

    Compares ACF of first feature channel at n_lags lags. For
    multivariate data, use measure_structure_loss_all_channels()
    instead — this single-channel version is kept for speed and
    backward compatibility.

    Returns a score in [0, 1] where:
        0.0 = perfect structure preservation
        1.0 = all temporal structure lost
    """
    try:
        # Use the first feature channel only
        real_series = real_train[:, 0]
        syn_series  = synthetic[:, 0]

        # Compute ACF at n_lags lags
        real_acf = compute_acf(real_series, nlags=n_lags, fft=True)
        syn_acf  = compute_acf(syn_series,  nlags=n_lags, fft=True)

        # Mean absolute difference between ACF profiles
        acf_diff = np.mean(np.abs(real_acf - syn_acf))

        # Normalise: max possible ACF diff is 2.0 (one is +1, other is -1)
        score = min(acf_diff / 2.0, 1.0)

    except Exception:
        # If ACF computation fails, assume moderate structure loss
        score = 0.5

    return round(score, 4)


def measure_structure_loss_all_channels(
    real_train: np.ndarray,
    synthetic:  np.ndarray,
    n_lags:     int = 24,
    max_channels: int = 8,
) -> float:
    """
    Multivariate variant of measure_structure_loss().

    Averages the ACF-difference score across up to max_channels channels
    (capped for speed on very wide datasets) instead of only checking
    channel 0. Falls back to the single-channel version on failure.
    """
    if real_train.ndim == 1 or real_train.shape[-1] == 1:
        return measure_structure_loss(real_train, synthetic, n_lags)

    n_channels = min(real_train.shape[-1], max_channels)
    scores = []
    for c in range(n_channels):
        try:
            real_acf = compute_acf(real_train[:, c], nlags=n_lags, fft=True)
            syn_acf  = compute_acf(synthetic[:, c],  nlags=n_lags, fft=True)
            acf_diff = np.mean(np.abs(real_acf - syn_acf))
            scores.append(min(acf_diff / 2.0, 1.0))
        except Exception:
            scores.append(0.5)

    return round(float(np.mean(scores)), 4) if scores else 0.5


# ─────────────────────────────────────────────────────────────────
# MEASURE D — Architecture Sensitivity (fixed lookup)
# ─────────────────────────────────────────────────────────────────

# Based on H2 findings:
#   DLinear: linear model, not sensitive to sequence structure loss
#   MLP:     independent windows, low sensitivity
#   CNN:     local patterns, moderate sensitivity
#   LSTM:    sequential memory, very sensitive to structure loss
ARCH_SENSITIVITY = {
    'DLinear': 0.2,
    'MLP':     0.3,
    'CNN':     0.5,
    'LSTM':    0.9,
}

def measure_arch_sensitivity(model_name: str) -> float:
    """
    How sensitive is this architecture to distillation quality loss?

    Returns a score in [0, 1] from the fixed lookup table.
    Unknown architectures default to 0.5 (moderate sensitivity).
    """
    return ARCH_SENSITIVITY.get(model_name, 0.5)


# ─────────────────────────────────────────────────────────────────
# COMBINE — Quality Demand Score
# ─────────────────────────────────────────────────────────────────

# Default weights — calibrated from 24-cell study
# (error_ratio, divergence, structure_loss, arch_sensitivity)
# NOTE: no fitting code ships alongside this constant. Treat these as a
# reasonable manually-chosen prior, not a statistically fit result, until
# an actual regression against h2_best_ratios.csv is run to justify them.
DEFAULT_WEIGHTS = (0.40, 0.25, 0.20, 0.15)

def compute_quality_demand(
    score_error_ratio:   float,
    score_divergence:    float,
    score_structure:     float,
    score_arch:          float,
    weights:             tuple = DEFAULT_WEIGHTS,
) -> float:
    """
    Combine four scores into one quality demand score.

    Quality demand answers:
        "How much real data does this cell need to recover quality?"

    Returns a score in [0, 1] where:
        0.0 = near-perfect distillation, need almost no real data
        1.0 = catastrophic failure, need maximum real data
    """
    w_err, w_div, w_str, w_arch = weights

    demand = (
        w_err  * score_error_ratio  +
        w_div  * score_divergence   +
        w_str  * score_structure    +
        w_arch * score_arch
    )

    # Clamp to [0, 1]
    return round(max(0.0, min(1.0, demand)), 4)


# ─────────────────────────────────────────────────────────────────
# PREDICT — Map Quality Demand to r* (architecture-aware)
# ─────────────────────────────────────────────────────────────────

# FIX 4: grid now spans the FULL 1-100 range instead of capping at 50,
# so a catastrophic-failure quality_demand can recommend pure real data.
RATIO_GRID = list(range(1, 101))

# FIX 3: minimum spread enforced on any architecture-specific range so
# quality_demand always has room to move the prediction. If the observed
# [min_best_r_pct, max_best_r_pct] from h2_per_arch_summary.csv is
# narrower than this, it gets symmetrically widened.
MIN_RANGE_SPREAD = 20

# FIX 1: generic fallback is now the full grid, not a hardcoded [1, 50].
# This is used only when no architecture-specific range is available yet
# (e.g. the very first run, before h2_experiment.py has ever produced
# h2_per_arch_summary.csv).
GENERIC_FALLBACK_RANGE = (1, 100)

H2_SUMMARY_PATH = _EXPERIMENTS_DIR / 'results' / 'h2_results' / 'h2_per_arch_summary.csv'

# Internal cache: {path_str: (mtime, {model: (r_min, r_max)})}
# Re-read only when the file's mtime changes — cheap, but never stale
# within a single process (FIX 2).
_ARCH_RANGE_CACHE: dict = {}

# Tracks which model names we've already warned about, so the warning
# for FIX 1 prints once per model per process rather than spamming.
_WARNED_MODELS: set = set()


def load_arch_ratio_range(path: Path = H2_SUMMARY_PATH, force_reload: bool = False) -> dict:
    """
    Build {model_name: (min_r_star, max_r_star)} from h2_per_arch_summary.csv.

    min_r = lowest best-ratio ever observed for this architecture (min_best_r_pct)
    max_r = highest best-ratio ever observed for this architecture (max_best_r_pct)

    FIX 2: Unlike the original (which cached this once at import time),
    this re-reads the file whenever its mtime has changed, so a fresh
    h2_experiment.py run is picked up automatically on the next call
    within the same process — no reload needed.

    Returns an empty dict if the CSV doesn't exist yet (e.g. h2_experiment.py
    hasn't been run) — predict_r_star() then falls back to
    GENERIC_FALLBACK_RANGE for every model, with a one-time warning.
    """
    path = Path(path)
    key = str(path)

    if not path.exists():
        _ARCH_RANGE_CACHE.pop(key, None)
        return {}

    mtime = path.stat().st_mtime
    cached = _ARCH_RANGE_CACHE.get(key)
    if cached is not None and not force_reload and cached[0] == mtime:
        return cached[1]

    df = pd.read_csv(path)
    ranges = {
        row['model']: (int(row['min_best_r_pct']), int(row['max_best_r_pct']))
        for _, row in df.iterrows()
    }
    _ARCH_RANGE_CACHE[key] = (mtime, ranges)
    return ranges


def _resolve_range(model_name: str, arch_ratio_range: dict) -> tuple:
    """
    Resolve the (r_min, r_max) range to use for a given model, applying
    FIX 1 (loud fallback warning) and FIX 3 (minimum spread enforcement).
    """
    if model_name and model_name in arch_ratio_range:
        r_min, r_max = arch_ratio_range[model_name]

        # FIX 3: widen degenerate or overly narrow ranges so quality_demand
        # still has an effect, instead of always returning the same ratio.
        spread = r_max - r_min
        if spread < MIN_RANGE_SPREAD:
            pad = (MIN_RANGE_SPREAD - spread) / 2
            r_min = max(1,   int(round(r_min - pad)))
            r_max = min(100, int(round(r_max + pad)))
            if r_max - r_min < MIN_RANGE_SPREAD:
                # still short (clamped against 1 or 100) — push the
                # other side out as far as it will go
                if r_min == 1:
                    r_max = min(100, r_min + MIN_RANGE_SPREAD)
                elif r_max == 100:
                    r_min = max(1, r_max - MIN_RANGE_SPREAD)

        return r_min, r_max

    # FIX 1: no architecture-specific data available — fall back to the
    # generic range and warn loudly, once per model per process, instead
    # of silently substituting a narrower hardcoded [1, 50].
    if model_name not in _WARNED_MODELS:
        print(
            f"\n  [WARNING] No H2 sweep data found for model '{model_name}' in "
            f"{H2_SUMMARY_PATH}.\n"
            f"            Falling back to the generic range "
            f"{GENERIC_FALLBACK_RANGE} — this prediction is NOT "
            f"architecture-calibrated.\n"
            f"            Run h2_experiment.py at least once for this "
            f"architecture to get a real range.\n"
        )
        _WARNED_MODELS.add(model_name)

    return GENERIC_FALLBACK_RANGE


def demand_to_r_star(quality_demand: float, model_name: str = None,
                      arch_ratio_range: dict = None) -> int:
    """
    Map quality demand score to the nearest tested ratio.

    If model_name is provided and has data in h2_per_arch_summary.csv,
    uses that architecture-specific range (widened if necessary — FIX 3).
    Otherwise uses the generic [1, 100] range with a one-time warning
    (FIX 1).

    arch_ratio_range : optional override, skipping the on-disk lookup.
        Used by ratio_predictor_verifier.py to inject a leave-one-out
        range (recomputed excluding the cell under test) instead of the
        cached range from h2_per_arch_summary.csv, which would otherwise
        leak that cell's own contribution back into its own prediction.

    Returns integer percentage from RATIO_GRID.
    """
    if arch_ratio_range is None:
        arch_ratio_range = load_arch_ratio_range()
    r_min, r_max = _resolve_range(model_name, arch_ratio_range)

    # Map demand [0, 1] → [r_min, r_max]
    r_continuous = r_min + quality_demand * (r_max - r_min)

    # Snap to nearest grid point within the resolved range
    valid_grid = [g for g in RATIO_GRID if r_min <= g <= r_max]
    if not valid_grid:
        valid_grid = RATIO_GRID

    r_star = min(valid_grid, key=lambda g: abs(g - r_continuous))
    return r_star


# ─────────────────────────────────────────────────────────────────
# MAIN — predict_r_star
# ─────────────────────────────────────────────────────────────────

def predict_r_star(
    synth_mse:   float,
    real_mse:    float,
    model_name:  str,
    real_train:  np.ndarray,
    synthetic:   np.ndarray,
    weights:     tuple = DEFAULT_WEIGHTS,
    verbose:     bool  = True,
    channel_aware: bool = False,
    arch_ratio_range_override: dict = None,
) -> int:
    """
    Predict the optimal mixing ratio r* for a new dataset-model
    combination without running a full sweep.

    Parameters
    ----------
    synth_mse   : MSE of student trained on pure synthetic data
    real_mse    : MSE of student trained on pure real data
    model_name  : model architecture name, e.g. 'DLinear', 'LSTM', 'MLP', 'CNN'
    real_train  : numpy array of real training data (N, features)
    synthetic   : numpy array of synthetic sequence (M, features)
    weights     : (w_error, w_divergence, w_structure, w_arch)
    verbose     : print breakdown of scores
    channel_aware : if True, use the per-channel divergence/structure
                    variants (recommended for multivariate datasets with
                    more than a couple of channels, e.g. 'weather')
    arch_ratio_range_override : optional {model: (r_min, r_max)} dict,
                    bypassing the on-disk h2_per_arch_summary.csv lookup.
                    See demand_to_r_star() — used for leave-one-out backtesting.

    Returns
    -------
    int : recommended mixing ratio percentage (from RATIO_GRID, 1-100)
    """
    # Step 1: Compute four scores
    s_error = measure_error_ratio(synth_mse, real_mse)
    if channel_aware:
        s_div = measure_divergence_per_channel(real_train, synthetic)
        s_str = measure_structure_loss_all_channels(real_train, synthetic)
    else:
        s_div = measure_divergence(real_train, synthetic)
        s_str = measure_structure_loss(real_train, synthetic)
    s_arch = measure_arch_sensitivity(model_name)

    # Step 2: Combine into quality demand
    demand = compute_quality_demand(s_error, s_div, s_str, s_arch, weights)

    # Step 3: Map to r*
    r_star = demand_to_r_star(demand, model_name=model_name,
                               arch_ratio_range=arch_ratio_range_override)

    if verbose:
        arch_ratio_range = arch_ratio_range_override or load_arch_ratio_range()
        r_min, r_max = _resolve_range(model_name, arch_ratio_range)
        print(f"\n  r* Prediction for {model_name}")
        print(f"  {'─'*40}")
        print(f"  error_ratio score : {s_error:.4f}  (weight {weights[0]})")
        print(f"  divergence score  : {s_div:.4f}  (weight {weights[1]})"
              + ("  [channel-aware: worst channel]" if channel_aware else ""))
        print(f"  structure loss    : {s_str:.4f}  (weight {weights[2]})"
              + ("  [channel-aware: mean of channels]" if channel_aware else ""))
        print(f"  arch sensitivity  : {s_arch:.4f}  (weight {weights[3]})")
        print(f"  {'─'*40}")
        print(f"  quality demand    : {demand:.4f}")
        print(f"  ratio range used  : [{r_min}%, {r_max}%]")
        print(f"  predicted r*      : {r_star}%")
        print(f"  compression kept  : {100 - r_star}%")

    return r_star


# ─────────────────────────────────────────────────────────────────
# DEMO — mark predicted r* on existing sweep curves
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    import matplotlib.pyplot as plt

    # ── FIX 7: paths anchored to this file, not the current working
    # directory. Inputs are read from results/ (where h2_experiment.py
    # writes h2_best_ratios.csv and the predictor_inputs/*.npz files).
    # Annotated outputs are written to a SEPARATE folder, ratio_predictor_
    # results/, instead of overwriting the original sweep's curve PNGs.
    INPUT_DIR  = _EXPERIMENTS_DIR / 'results'
    OUTPUT_DIR = _EXPERIMENTS_DIR / 'ratio_predictor_results'

    csv_path   = INPUT_DIR / 'h2_results' / 'h2_best_ratios.csv'
    sweep_path = INPUT_DIR / 'h2_results' / 'h2_full_sweep.csv'

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run h2_experiment.py first.")
        raise SystemExit(1)

    try:
        sweep_df = pd.read_csv(sweep_path)
    except FileNotFoundError:
        print(f"ERROR: CSV not found at {sweep_path}")
        print("Run h2_experiment.py first.")
        raise SystemExit(1)

    # Per-ratio MSE columns live in h2_full_sweep.csv as 'ratio_XXX_pct',
    # not in h2_best_ratios.csv — merge them in on (dataset, model).
    df = df.merge(sweep_df, on=['dataset', 'model'], how='left')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    print("=" * 65)
    print("predict_r_star — marking predicted r* on existing curves")
    print(f"  reading inputs from : {INPUT_DIR}")
    print(f"  writing outputs to  : {OUTPUT_DIR}")
    print("=" * 65)

    summary_rows = []

    for _, row in df.iterrows():
        dataset   = row['dataset']
        model     = row['model']
        synth_mse = float(row['synth_mse'])
        real_mse  = float(row['real_mse'])

        # ── Load real arrays — REQUIRED, no silent fallback (FIX 5) ───────
        inputs_path = INPUT_DIR / 'predictor_inputs' / f'{dataset}_{model}_inputs.npz'
        if not inputs_path.exists():
            print(f"\n  {'!'*65}")
            print(f"  [SKIPPED] {dataset} x {model}")
            print(f"  Missing input file: {inputs_path}")
            print(f"  Refusing to substitute random noise for real data — "
                  f"that would produce a meaningless prediction with no "
                  f"indication it wasn't real.")
            print(f"  Save real_train / synthetic arrays to this path to "
                  f"include this cell.")
            print(f"  {'!'*65}")
            continue

        data       = np.load(inputs_path, allow_pickle=True)
        real_train = data['real_train']
        synthetic  = data['synthetic']

        # ── Run predictor ─────────────────────────────────────────────────
        predicted = predict_r_star(
            synth_mse  = synth_mse,
            real_mse   = real_mse,
            model_name = model,
            real_train = real_train,
            synthetic  = synthetic,
            verbose    = False,
        )

        # ── Locate the existing (source) mixing curve PNG ─────────────────
        # Only used as a gate — confirms a curve was actually produced for
        # this cell. The plot itself is redrawn fresh from the CSV's ratio
        # columns below, then saved as a NEW file in OUTPUT_DIR (FIX 7),
        # leaving the original untouched.
        source_curve_path = INPUT_DIR / 'h2_strategy_selection' / 'S1_Random' / f"{dataset}_{model}_mixing_curve.png"
        if not source_curve_path.exists():
            print(f"  SKIP {dataset} × {model} — no mixing curve found at {source_curve_path}")
            continue

        out_curve_path = OUTPUT_DIR / f"{dataset}_{model}_mixing_curve_predicted.png"

        # ── Extract the ratio MSE columns from CSV to find x/y position ───
        ratio_cols = sorted(
            [c for c in row.index
             if c.startswith('ratio_') and c.endswith('_pct')],
            key=lambda c: int(c.replace('ratio_', '').replace('_pct', ''))
        )
        ratios = [int(c.replace('ratio_', '').replace('_pct', ''))
                  for c in ratio_cols]
        mses   = [float(row[c]) for c in ratio_cols
                  if not pd.isna(row.get(c))]
        ratios  = ratios[:len(mses)]

        if not ratios:
            print(f"  SKIP {dataset} × {model} — no ratio data in CSV")
            continue

        # ── Find MSE at predicted ratio if it was tested ──────────────────
        pred_col      = f'ratio_{predicted:03d}_pct'
        pred_in_sweep = pred_col in row.index and not pd.isna(row.get(pred_col))
        pred_mse_val  = float(row[pred_col]) if pred_in_sweep else None

        # Predicted r* usually falls between tested sweep points (which
        # only cover 0/1/2/5/10/15/20/30/50/100%), so interpolate along
        # the mixing curve to report an MSE estimate even when untested.
        pred_mse_interp = float(np.interp(predicted, ratios, mses))

        # ── Redraw the curve fresh with the predicted r* marker added ─────
        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(ratios, mses, 'o-',
                color='steelblue', linewidth=2, markersize=5,
                label='Hybrid MSE', zorder=3)

        ax.axhline(synth_mse, color='#CC0000', linestyle='--', linewidth=1.3,
                   label=f'Pure synthetic (0%): {synth_mse:.4f}')
        ax.axhline(real_mse, color='#217346', linestyle='--', linewidth=1.3,
                   label=f'Pure real (100%): {real_mse:.4f}')

        # FIX 6: correct column name for the actual best ratio
        best_ratio_str = str(row.get('best_ratio_key', ''))
        if best_ratio_str and best_ratio_str != 'nan':
            try:
                best_r   = int(best_ratio_str.replace('hybrid_', ''))
                best_col = f'ratio_{best_r:03d}_pct'
                if best_col in row.index and not pd.isna(row.get(best_col)):
                    best_mse_val = float(row[best_col])
                    ax.scatter([best_r], [best_mse_val],
                               color='#217346', s=200, zorder=6, marker='*',
                               label=f'Actual best r*={best_r}%  '
                                     f'MSE={best_mse_val:.4f}')
            except (ValueError, KeyError):
                pass

        ax.axvline(predicted, color='#FF8C00', linestyle=':',
                   linewidth=2, zorder=4,
                   label=f'Predicted r*={predicted}%')

        if pred_in_sweep and pred_mse_val is not None:
            ax.scatter([predicted], [pred_mse_val],
                       color='#FF8C00', s=220, zorder=7, marker='o',
                       label=f'Predicted MSE={pred_mse_val:.4f}')
            ax.annotate(
                f'  r*={predicted}%\n  MSE={pred_mse_val:.4f}',
                xy=(predicted, pred_mse_val),
                xytext=(predicted + max(1, max(ratios) * 0.03),
                        pred_mse_val + 0.008 * abs(pred_mse_val)),
                fontsize=8.5, color='#FF8C00', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.2)
            )
        else:
            ax.scatter([predicted], [pred_mse_interp],
                       color='#FF8C00', s=220, zorder=7, marker='o',
                       label=f'Predicted MSE (interp)={pred_mse_interp:.4f}')
            ax.annotate(
                f'  r*={predicted}%\n  MSE={pred_mse_interp:.4f} (interp)',
                xy=(predicted, pred_mse_interp),
                xytext=(predicted + max(1, max(ratios) * 0.03),
                        pred_mse_interp + 0.008 * abs(pred_mse_interp)),
                fontsize=8.5, color='#FF8C00', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.2)
            )

        ax.set_xlabel("Real Data Percentage (%)", fontsize=11)
        ax.set_ylabel("Test MSE", fontsize=11)
        ax.set_title(
            f"{dataset} × {model} — Mixing Curve  "
            f"[Predicted r*={predicted}%]",
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=8.5, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_curve_path, dpi=130, bbox_inches='tight')
        plt.close()

        actual = str(row.get('best_ratio_key', 'unknown'))
        match  = "✓" if str(predicted) == actual.replace('hybrid_', '') else "—"
        mse_note = "(tested)" if pred_in_sweep else "(interp)"
        print(f"  {dataset:<18} {model:<10}  "
              f"predicted={predicted:>3}%  MSE={pred_mse_interp:.6f} {mse_note}  "
              f"actual={actual:>12}  {match}  "
              f"→ {out_curve_path.name}")

        summary_rows.append({
            'dataset':                dataset,
            'model':                  model,
            'synthetic_only_mse':     synth_mse,
            'real_only_mse':          real_mse,
            'predicted_ratio_pct':    predicted,
            'compression_pct':       100 - predicted,
            'predicted_ratio_mse':    round(pred_mse_interp, 6),
            'predicted_ratio_mse_source': 'tested' if pred_in_sweep else 'interpolated',
        })

    summary_csv_path = OUTPUT_DIR / 'predicted_ratio_summary.csv'
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    print(f"\nAll curves saved to {OUTPUT_DIR}/ with predicted r* markers.")
    print(f"Summary CSV saved to {summary_csv_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
