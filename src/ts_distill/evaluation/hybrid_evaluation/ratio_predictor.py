"""
ratio_predictor.py
============================
Predicts the optimal mixing ratio r* for any new dataset-architecture.

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
"""

import numpy as np
from statsmodels.tsa.stattools import acf as compute_acf

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
    real_std  = real_train.std() + 1e-8

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

# Default weights — calibrated from a 24-cell study.
# NOTE: these are a reasonable manually-chosen prior, not a statistically
# fit result.
# (error_ratio, divergence, structure_loss, arch_sensitivity)
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

# Grid spans the full 1-100 range, so a catastrophic-failure
# quality_demand can recommend pure real data.
RATIO_GRID = list(range(1, 101))

# Minimum spread enforced on any architecture-specific range so
# quality_demand always has room to move the prediction. If a caller-
# supplied [r_min, r_max] is narrower than this, it gets symmetrically
# widened.
MIN_RANGE_SPREAD = 20

# Used whenever no architecture-specific range is supplied for a model.
GENERIC_FALLBACK_RANGE = (1, 100)


def _resolve_range(model_name: str, arch_ratio_range: dict) -> tuple:
    """
    Resolve the (r_min, r_max) range to use for a given model, applying
    minimum spread enforcement, or falling back to the generic range.
    """
    if model_name and arch_ratio_range and model_name in arch_ratio_range:
        r_min, r_max = arch_ratio_range[model_name]

        # Widen degenerate or overly narrow ranges so quality_demand
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

    return GENERIC_FALLBACK_RANGE


def demand_to_r_star(quality_demand: float, model_name: str = None,
                      arch_ratio_range: dict = None) -> int:
    """
    Map quality demand score to the nearest tested ratio.

    arch_ratio_range : optional {model: (r_min, r_max)} dict. If the
        model isn't present (or no dict is given), falls back to
        GENERIC_FALLBACK_RANGE. Used by ratio_predictor_verifier.py to
        inject a leave-one-out range recomputed excluding the cell
        under test.

    Returns integer percentage from RATIO_GRID.
    """
    r_min, r_max = _resolve_range(model_name, arch_ratio_range or {})

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
    arch_ratio_range_override : optional {model: (r_min, r_max)} dict.
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
        r_min, r_max = _resolve_range(model_name, arch_ratio_range_override or {})
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


