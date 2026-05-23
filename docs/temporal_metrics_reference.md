# Temporal Fidelity Metrics — Reference Document

**Framework**: `ts-distill` | **Module**: `src/ts_distill/metrics/`

This document describes the exact computation of every temporal fidelity metric
used to evaluate MTT dataset distillation quality. Each section includes the
mathematical formula, implementation details, how to interpret the output, and
which hypothesis the metric tests.

---

## Background — What Is Being Compared

Every metric compares two sequences:

| Symbol | Meaning | Shape |
|--------|---------|-------|
| **real** | Continuous real training segment, StandardScaler-normalised | `(T, C)` |
| **synthetic** | Distilled synthetic sequence from MTT | `(M, C)` |

Where `T` = length of the real training set (varies by dataset),
`M` = 384 (fixed by the distillation config), and `C` = number of channels
(7 for all ETT datasets).

**Truncation rule** — three out of four metrics compare equal-length windows:

```
N = min(T, M)     →  real[:N]  vs  synthetic[:M]
```

Since `M=384 < T` in all ETT splits, `N = 384` in practice — the first 384
time steps of the real training data are compared against all 384 synthetic
steps.

**Exception — VarianceMetric** uses the full real sequence (`T` steps) because
variance is a global property that improves in accuracy with more data.

---

## 1. ACF Metric (`ACFMetric`)

**File**: [src/ts_distill/metrics/acf.py](../src/ts_distill/metrics/acf.py)

### 1.1 Purpose

Measures how well the synthetic sequence reproduces the autocorrelation
structure of the real data. The ACF captures how strongly a time series is
correlated with its own past values at each lag `k`.

A perfect synthetic series would have an identical ACF to the real series.

### 1.2 Lag-Band Split

The computation is split into **two separate scores** based on lag distance:

| Score | Lag range | What it captures |
|-------|-----------|-----------------|
| `acf_short` | lags 1 .. `period` | Dominant seasonal / periodic cycle |
| `acf_long` | lags `period+1` .. `K` | Slow decay / long-range dependence |

The split directly operationalises **Hypothesis H1**:
> *"Current distillation methods preserve dominant periodic structures
> (short-range ACF) but fail to capture long-range temporal dependencies
> (long-range ACF)."*

H1 predicts `acf_short < acf_long` across all datasets and models.

### 1.3 Dataset Periods

| Dataset | Frequency | Period (`p`) |
|---------|-----------|-------------|
| ETTh1, ETTh2 | Hourly | 24 (one day) |
| ETTm1, ETTm2 | 15-minute | 96 (one day) |

### 1.4 Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `period` | 24 | Dominant seasonal period (set per dataset) |
| `n_lags` (K) | 100 | Total ACF lags computed. Must satisfy `period < K < N` |

### 1.5 Formula

For each channel `c`:

**Step 1 — Truncate** both sequences to the same length:
```
N = min(T, M)
real_c = real[:N, c]
syn_c  = synthetic[:N, c]
```

**Step 2 — Compute ACF** vectors (lags 1 through K, lag-0 dropped):
```
ρ_real[k]  =  ACF(real_c,  lag=k)    for k = 1, 2, ..., K
ρ_syn[k]   =  ACF(syn_c,   lag=k)    for k = 1, 2, ..., K
```

Computed via `statsmodels.tsa.stattools.acf(x, nlags=K, fft=True)[1:]`
(FFT-accelerated Yule-Walker estimator; `[1:]` discards the always-1 lag-0 value).

**Step 3 — Per-lag absolute error**:
```
δ[k] = |ρ_real[k] − ρ_syn[k]|
```

**Step 4 — Band-wise mean**:
```
acf_short[c] = mean( δ[k]  for k in {1, ..., p}    )
acf_long[c]  = mean( δ[k]  for k in {p+1, ..., K}  )
```

**Step 5 — Aggregate across channels** (main table only):
```
acf_short = mean over c of acf_short[c]
acf_long  = mean over c of acf_long[c]
```

### 1.6 Output

| Scope | Variable | Shape | Unit |
|-------|----------|-------|------|
| Per channel | `acf_results` | `(C, 2)` | dimensionless [0, 2] |
| Main table | `acf_short`, `acf_long` | scalar | dimensionless [0, 2] |

ACF values lie in [−1, 1], so the absolute difference lies in [0, 2]. In
normalised data, real-world values are almost always within [0, 0.5].

### 1.7 Interpretation

- **`acf_short → 0`**: The synthetic data reproduces the seasonal cycle well.
- **`acf_long → 0`**: The synthetic data also reproduces long-range dependencies.
- **`acf_long > acf_short`**: The distillation compressed away long-range memory
  (supports H1).
- **`acf_long ≈ acf_short`**: The method treats all lags equally, regardless of
  temporal structure.

---

## 2. Frequency Metric (`FrequencyMetric`)

**File**: [src/ts_distill/metrics/frequency.py](../src/ts_distill/metrics/frequency.py)

### 2.1 Purpose

Measures how well the synthetic sequence reproduces the **frequency content**
of the real data. The FFT magnitude spectrum shows which frequencies carry the
most energy. A large `fft_distance` means the synthetic data emphasises
different frequencies than the real data.

### 2.2 Why Magnitude Only

Phase is not compared because the synthetic sequence is generated independently
of the real sequence — their phases are arbitrarily shifted. Only the
distribution of energy across frequencies (magnitude) is a meaningful property
to compare.

### 2.3 Formula

For each channel `c`:

**Step 1 — Truncate** to equal length:
```
N = min(T, M)
```

**Step 2 — Compute one-sided FFT magnitude spectra**:
```
F_real[f] = |FFT(real[:N, c])|    for f = 0, 1, ..., N//2
F_syn[f]  = |FFT(syn[:N,  c])|
```

Computed via `numpy.fft.rfft` (real-valued FFT), which returns `N//2 + 1`
frequency bins. Both calls receive the same `N` samples, so outputs have
identical length.

**Step 3 — L2 distance, normalised by N**:
```
fft_distance[c] = ||F_real − F_syn||₂ / N
```

The division by `N` makes the metric comparable across datasets with different
sequence lengths (normalises for FFT magnitude scale, which grows with N).

**Step 4 — Aggregate** (main table):
```
fft_distance = mean over c of fft_distance[c]
```

### 2.4 Output

| Scope | Variable | Shape | Unit |
|-------|----------|-------|------|
| Per channel | `fft_results` | `(C,)` | normalised amplitude |
| Main table | `fft_distance` | scalar | normalised amplitude |

### 2.5 Interpretation

- **Small `fft_distance`**: The synthetic data contains the same frequency
  components as the real data — dominant periodicities are preserved.
- **Large `fft_distance`**: The synthetic data has a different spectral profile
  — certain frequencies are over- or under-represented.
- Comparing `fft_distance` with `acf_short` provides redundant confirmation:
  a method that preserves the dominant frequency peak (low `fft_distance`)
  should also score well on `acf_short`.

---

## 3. Trend Metric (`TrendMetric`)

**File**: [src/ts_distill/metrics/trend.py](../src/ts_distill/metrics/trend.py)

### 3.1 Purpose

Measures how well the synthetic sequence reproduces the **slow-moving trend**
of the real data. A trend error separates trend preservation from seasonal
preservation — a method could score well on ACF (seasonal) while completely
missing the trend direction.

### 3.2 Method — STL Decomposition

STL (Seasonal-Trend decomposition via LOESS) additively decomposes a time series:

```
x[t] = trend[t] + seasonal[t] + residual[t]
```

The `trend[t]` component is a smoothed signal capturing the direction the series
moves over many periods, independent of seasonal fluctuations.

**Parameters**:
- `period`: Same per-dataset value used in ACFMetric (24 for ETTh, 96 for ETTm)
- `robust=True`: Uses iteratively reweighted LOESS, reducing sensitivity to
  outliers in StandardScaler-normalised data
- **Safety requirement**: sequence length `N >= 2 × period`. With `M=384` and
  max `period=96`: `384 >= 192`. Always satisfied with default config.

### 3.3 Formula

For each channel `c`:

**Step 1 — Truncate** to equal length:
```
N = min(T, M)
```

**Step 2 — STL decomposition** on both truncated sequences:
```
trend_real[t] = STL(real[:N, c],  period=p, robust=True).trend
trend_syn[t]  = STL(syn[:N,  c],  period=p, robust=True).trend
```

Both trend vectors have shape `(N,)`.

**Step 3 — L2 distance, normalised by N**:
```
trend_error[c] = ||trend_real − trend_syn||₂ / N
```

The division by `N` normalises for sequence length so that values are
comparable across different dataset splits.

**Step 4 — Aggregate** (main table):
```
trend_error = mean over c of trend_error[c]
```

### 3.4 Output

| Scope | Variable | Shape | Unit |
|-------|----------|-------|------|
| Per channel | `trend_results` | `(C,)` | normalised units |
| Main table | `trend_error` | scalar | normalised units |

Units are in the same scale as the input data (StandardScaler output, so
approximately standard deviations).

### 3.5 Interpretation

- **Small `trend_error`**: The synthetic data follows the same long-term
  direction as the real training window.
- **Large `trend_error`**: The synthetic data drifts in a different direction
  than the real data — the distillation captured local patterns but not the
  global trajectory.
- Comparing `trend_error` with `transfer_mse` tests whether trend preservation
  is necessary for forecasting accuracy (H2).

---

## 4. Variance Metric (`VarianceMetric`)

**File**: [src/ts_distill/metrics/variance.py](../src/ts_distill/metrics/variance.py)

### 4.1 Purpose

Measures how well the synthetic sequence preserves the **signal amplitude**
relative to the real data. A synthetic series with much lower variance looks
"flatter" than the real data; one with much higher variance is more volatile.
Models trained on a mismatched variance may learn to predict at the wrong
scale.

### 4.2 Why Full Real Sequence

Unlike the other three metrics, variance uses the **full real training sequence**
(all `T` time steps), not the truncated `N=384` window. Variance is a global
statistical property — estimated accuracy improves with sample size. Using the
full sequence gives the most reliable baseline estimate of the real data's
amplitude.

### 4.3 Formula

For each channel `c`:

**Step 1 — Population variances** (no truncation of real):
```
Var_real[c] = Var(real[:, c])      — full T time steps
Var_syn[c]  = Var(synthetic[:, c]) — all M=384 steps
```

Computed via `numpy.var(x, axis=0)` (population variance, `ddof=0`).

**Step 2 — Relative absolute difference**:
```
variance_diff[c] = |Var_real[c] − Var_syn[c]| / (Var_real[c] + ε)
```

Where `ε = 1e-10` guards against division by zero on perfectly constant
channels (cannot occur after StandardScaler unless a feature has zero variance
in the entire training set).

The division by `Var_real` makes the metric **scale-free**: a
`variance_diff = 0.1` means the synthetic variance deviates by 10 % from the
real variance, regardless of the original signal amplitude.

**Step 3 — Aggregate** (main table):
```
variance_diff = mean over c of variance_diff[c]
```

### 4.4 Output

| Scope | Variable | Shape | Unit |
|-------|----------|-------|------|
| Per channel | `var_results` | `(C,)` | dimensionless ratio [0, ∞) |
| Main table | `variance_diff` | scalar | dimensionless ratio |

- `0` = perfect variance preservation
- `0.1` = 10 % relative deviation
- `> 1.0` = variance more than doubled or halved (poor preservation)

### 4.5 Interpretation

- **`variance_diff ≈ 0`**: The synthetic data matches the real data in amplitude;
  a model trained on synthetic data encounters the same signal scale as in
  the real test set.
- **`variance_diff > 0.5`**: Significant amplitude mismatch — the synthetic data
  is either over-smoothed (underestimated variance) or artificially noisy
  (overestimated variance).
- After StandardScaler normalisation all channels have unit variance by
  construction, so `Var_real[c] ≈ 1` and `variance_diff ≈ |1 − Var_syn[c]|`.
  This means `variance_diff` directly reflects how far the synthetic variance
  deviates from 1.0.

---

## 5. Result Tables

### 5.1 Feature-Wise Table (`metrics_feature_wise.csv`)

One row per channel per experiment. Used for per-feature analysis and appendix
figures.

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | str | Dataset name (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`) |
| `feature` | int | 0-indexed channel number (0 – 6 for ETT) |
| `model` | str | Forecasting model (`DLinear`, `LSTM`, `MLP`, `CNN`) |
| `distillation_method` | str | Algorithm (`MTT`) |
| `real_mse` | float | Test MSE, model trained on real data |
| `transfer_mse` | float | Test MSE, model trained on synthetic data |
| `acf_short` | float | ACF MAE over lags 1..`period` |
| `acf_long` | float | ACF MAE over lags `period+1`..K |
| `fft_distance` | float | FFT magnitude L2 / N |
| `trend_error` | float | STL trend L2 / N |
| `variance_diff` | float | `|Var_real − Var_syn| / Var_real` |

### 5.2 Main Table (`metrics_main.csv`)

One row per experiment (metrics averaged across all C channels). Used for
hypothesis testing and the main thesis body.

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | str | Dataset name |
| `model` | str | Forecasting model |
| `distillation_method` | str | Algorithm |
| `real_mse` | float | Baseline MSE |
| `transfer_mse` | float | Distillation MSE |
| `acf_short` | float | Mean `acf_short` across channels |
| `acf_long` | float | Mean `acf_long` across channels |
| `fft_distance` | float | Mean `fft_distance` across channels |
| `trend_error` | float | Mean `trend_error` across channels |
| `variance_diff` | float | Mean `variance_diff` across channels |

---

## 6. Hypothesis Mapping

| Hypothesis | Primary metrics | Expected pattern |
|------------|----------------|-----------------|
| **H1** — Periodic OK, long-range fails | `acf_short`, `acf_long` | `acf_long > acf_short` consistently |
| **H2** — Temporal fidelity correlates with forecasting | `acf_long`, `transfer_mse` | Pearson/Spearman r > 0.5 |
| **H3** — Explicit temporal losses improve quality | All metrics (Stage 4 vs Stage 3) | Lower scores under temporal-loss MTT |

---

## 7. Metric Quick Reference

| Metric | Formula | Lower = | Range |
|--------|---------|---------|-------|
| `acf_short` | `mean |ρ_real[k] − ρ_syn[k]|`, k=1..p | Better seasonal match | [0, 2] |
| `acf_long` | `mean |ρ_real[k] − ρ_syn[k]|`, k=p+1..K | Better long-range match | [0, 2] |
| `fft_distance` | `‖F_real − F_syn‖₂ / N` | Better frequency match | [0, ∞) |
| `trend_error` | `‖trend_real − trend_syn‖₂ / N` | Better trend match | [0, ∞) |
| `variance_diff` | `|Var_real − Var_syn| / Var_real` | Better amplitude match | [0, ∞) |
| `real_mse` | Forecasting MSE on real-trained model | — | [0, ∞) |
| `transfer_mse` | Forecasting MSE on synthetic-trained model | Better distillation | [0, ∞) |

---

## 8. Implementation Notes

### Normalisation consistency
All metric computations receive data that has already been normalised by
StandardScaler fitted on the training split. Metrics are therefore scale-free
within a dataset. Cross-dataset comparisons are valid for `acf_short`,
`acf_long`, and `variance_diff` (fully dimensionless). `fft_distance` and
`trend_error` are normalised by N but retain units of the input scale
(approximately standard deviations), making cross-dataset comparisons
directionally valid but not strictly comparable in magnitude.

### Reproducibility
- ACF: `statsmodels.tsa.stattools.acf` with `fft=True` (Blackman-Tukey
  estimator). Results are deterministic.
- FFT: `numpy.fft.rfft` — deterministic.
- STL: `statsmodels.tsa.seasonal.STL` with `robust=True` — deterministic
  (iterative LOESS converges to a fixed point).
- Variance: `numpy.var` — deterministic.

No randomness is introduced in any metric computation.

### Dependencies

```
numpy       ≥ 1.21
statsmodels ≥ 0.13   (for acf() and STL)
```
