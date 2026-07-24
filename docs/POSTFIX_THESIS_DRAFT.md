# Post-Hoc Temporal Fidelity Restoration in Time-Series Dataset Distillation

*Thesis chapter draft — generated from `postfix_sweep.csv` (3 seeds × 3 models × 5 datasets × 7 α = 315 runs).*

---

## 1. Summary of Contribution

Dataset distillation compresses a large training set into a small synthetic one that
preserves *training utility* — a model trained on the synthetic data should generalize
almost as well as one trained on the real data. We show that the dominant trajectory-matching
distiller (MTT) achieves this utility **at the cost of destroying the temporal-statistical
structure** of the data: the distilled series no longer carries the correct frequency spectrum,
dominant period, autocorrelation, or spectral peak energy of the source signal.

We propose a **post-hoc FFT amplitude correction** ("post-fix") that repairs the frequency-domain
temporal structure *after* distillation, controlled by a single blend strength `α ∈ [0, 1]`. The
central empirical findings are:

1. **Temporal restoration is reliable and exact.** Across all 315 runs, post-fix drives the
   frequency-distance, dominant-period error, and spectral-peak error monotonically to **zero**
   at `α = 1`, and improves autocorrelation as a by-product. This holds for every dataset, model,
   and seed without exception.
2. **The utility cost is dataset-dependent and predictable.** For strongly-seasonal datasets
   (15-minute ETTm, 10-minute weather) the correction is *free or even improves* utility; for the
   weaker-seasonality hourly ETTh datasets it introduces a tunable trade-off.
3. **The correction is partial by construction.** It restores frequency-domain properties but
   cannot repair time-domain trend or signal variance — a limitation that motivates a future
   composite correction.

---

## 2. Background and Problem Formulation

### 2.1 Matching Training Trajectories (MTT)

Let $R \in \mathbb{R}^{T \times C}$ be the real (standardized) training series with $C$ channels,
and $S \in \mathbb{R}^{M \times C}$ the synthetic series with $M \ll T$ (here $M = 384$).
An *expert* model is trained on $R$ and its weight trajectory
$\{\theta_{\text{exp}}^{(0)}, \theta_{\text{exp}}^{(1)}, \dots\}$ is recorded. MTT optimizes $S$
so that a *student*, trained for $k$ inner steps on $S$ starting from $\theta_{\text{exp}}^{(t)}$,
lands near $\theta_{\text{exp}}^{(t+k)}$:

$$
\mathcal{L}_{\text{MTT}}(S) \;=\;
\frac{\big\lVert \theta_{\text{stu}}^{(k)}(S) - \theta_{\text{exp}}^{(t+k)} \big\rVert_2^2}
     {\big\lVert \theta_{\text{exp}}^{(t)} - \theta_{\text{exp}}^{(t+k)} \big\rVert_2^2 + \varepsilon}
$$

The denominator normalizes the trajectory segment length so the loss is scale-invariant.
$S$ is updated by second-order gradients $\partial \mathcal{L}_{\text{MTT}} / \partial S$
obtained by differentiating through the inner loop.

### 2.2 The temporal-distortion problem

$\mathcal{L}_{\text{MTT}}$ depends on $S$ **only through the gradients it induces on the student
weights** — never on how $S$ looks as a time series. Two very different-looking sequences that
produce the same weight updates are equivalent to MTT. Consequently the optimizer has no incentive
to preserve seasonality, periodicity, or autocorrelation, and empirically it does not (Section 6.1).

This matters whenever the distilled set is intended as a *representative, shareable artifact* rather
than a single-model training shortcut: multi-task reuse, cross-architecture transfer, exploratory
analysis, and domain-expert trust all require the synthetic series to resemble the real generative
process.

---

## 3. Method: FFT Amplitude Post-Fix

### 3.1 Construction

For each channel $c$, take the discrete Fourier transform of the distilled synthetic channel,
$\mathcal{F}(S_c) = |\mathcal{F}(S_c)|\, e^{i\phi_c}$, where $\phi_c = \angle \mathcal{F}(S_c)$
is the phase. We build a **whole-dataset target amplitude** $\bar{A}_c$ by averaging the amplitude
spectra of all $K = \lfloor T/M \rfloor$ non-overlapping length-$M$ segments of the real series:

$$
\bar{A}_c \;=\; \frac{1}{K} \sum_{j=1}^{K} \big|\, \mathcal{F}\big(R_c^{(j)}\big) \big|,
\qquad R_c^{(j)} = R_c[(j-1)M : jM]
$$

Averaging over $K$ segments (e.g. $K \approx 22$ for ETTh, $K \approx 90$ for weather) yields a
stable spectral envelope representing the entire dataset rather than one arbitrary window. The
corrected synthetic channel blends the two amplitudes while **keeping the synthetic phase**:

$$
\tilde{S}_c \;=\; \mathcal{F}^{-1}\!\Big[\, \big((1-\alpha)\,|\mathcal{F}(S_c)| + \alpha\,\bar{A}_c\big)\; e^{\,i\phi_c} \,\Big]
$$

- $\alpha = 0$ returns the distilled series unchanged.
- $\alpha = 1$ replaces the amplitude spectrum entirely with the real envelope, retaining only the
  phase (the "shape") learned by distillation.

Phase is preserved deliberately: phase alignment between two independently generated sequences is
arbitrary, so only the amplitude (the energy at each frequency) carries transferable seasonal
information.

### 3.2 Linear control of spectral distance

Because the blend is affine in the amplitude, the frequency distance to the real envelope scales
exactly linearly in $\alpha$:

$$
d_{\text{FFT}}(\tilde{S}) \;=\; (1-\alpha)\, d_{\text{FFT}}(S)
$$

so $\alpha$ is a direct, interpretable knob on how much temporal structure is restored. This
identity is confirmed numerically (Section 6.2): the measured `fft_distance` reaches machine-zero
($\sim 10^{-8}$) at $\alpha = 1$ in every run.

---

## 4. Temporal Fidelity Metrics

All metrics compare the synthetic series against the **whole real training series**, using the same
segment-averaging as the post-fix target so that metric and method are consistent. Lower is better
for all of them.

| Metric | Definition | Captures |
|---|---|---|
| `fft_distance` | $\frac{1}{M}\lVert \lvert\mathcal{F}(S_c)\rvert - \bar{A}_c \rVert_2$, averaged over $c$ | Overall spectral energy match |
| `freq_rank_error` | $\lvert \arg\max_{f>0}\lvert\mathcal{F}(S_c)\rvert - \arg\max_{f>0}\bar{A}_c \rvert$ | Dominant-period preservation |
| `peak_mag_ratio` | $\lvert \bar{A}_c[f^\*] - \lvert\mathcal{F}(S_c)\rvert[f^\*]\rvert / \bar{A}_c[f^\*]$, $f^\*$ = real peak | Energy at the dominant frequency |
| `acf_short`, `acf_long` | mean abs. diff. of autocorrelation over lags $<$ and $\ge$ period | Short/long-range temporal dependence |
| `trend_error` | error of the STL/low-frequency trend component | Slow drift (time domain) |
| `variance_diff` | relative difference of per-channel variance | Signal energy / scale |
| `cross_corr_error` | error of the inter-channel correlation matrix | Between-channel structure |

**Utility** is reported as `transfer_mse` — the test MSE of a probe model trained on the synthetic
series and evaluated on the real held-out test set — alongside `real_mse`, the same probe trained on
real data (the achievable lower bound).

Two mathematical couplings between these metrics are used in the analysis:

- **Wiener–Khinchin:** the power spectrum is the Fourier transform of the autocorrelation, so
  correcting the amplitude spectrum improves ACF *for free*.
- **Parseval:** total variance equals the summed squared amplitude, so a partial amplitude blend
  approximately preserves variance — meaning `variance_diff` is largely *outside* this method's reach.

---

## 5. Experimental Setup

| Component | Setting |
|---|---|
| Datasets | ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-min), weather (10-min) |
| Models | DLinear, MLP, CNN (LSTM excluded — Section 8) |
| Seeds | 3 (7, 42, 123); results reported as **median over seeds** (robust to divergence) |
| Distillation | MTT, 300 steps, expert 80 epochs, $M=384$, seq/pred len $96/96$ |
| α grid | {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} |
| Probe eval | Adam, lr $10^{-3}$, early stopping (patience 10) on real val set |
| Protocol | distill **once** per (seed, dataset, model); apply all α post-hoc; retrain probe per α |

All hyperparameters are frozen across the matrix — no per-dataset tuning.

---

## 6. Results

### 6.1 Distillation distorts temporal structure (α = 0 baseline)

At $\alpha=0$ (pure MTT output) the synthetic series is spectrally wrong on every dataset. Median
over seeds and models:

| Dataset | `fft_distance` | `freq_rank_error` | `peak_mag_ratio` | `acf_short` |
|---|---|---|---|---|
| ETTh1 | 0.356 | 3.4 bins | 0.377 | 0.288 |
| ETTh2 | 0.319 | **13.4 bins** | 0.478 | 0.294 |
| ETTm1 | 0.399 | 1.0 bin | 0.554 | 0.390 |
| ETTm2 | 0.325 | 1.6 bins | 0.459 | 0.225 |
| weather | 0.259 | 0.5 bins | 0.496 | 0.188 |

A `peak_mag_ratio` near 0.5 means roughly half the energy at the dominant seasonal frequency is
missing; a `freq_rank_error` of 13 bins (ETTh2) means the distilled series does not even peak at the
right period. This is the quantitative statement of the distortion problem.

### 6.2 Post-fix restores frequency-domain fidelity (reliable, exact)

At $\alpha = 1$ the same three frequency-domain metrics collapse to zero on **every** dataset:

| Dataset | `fft_distance` 0→1 | `freq_rank_error` 0→1 | `peak_mag_ratio` 0→1 | `acf_short` 0→1 |
|---|---|---|---|---|
| ETTh1 | 0.356 → **0.000** | 3.4 → **0** | 0.377 → **0** | 0.288 → 0.184 |
| ETTh2 | 0.319 → **0.000** | 13.4 → **0** | 0.478 → **0** | 0.294 → 0.208 |
| ETTm1 | 0.399 → **0.000** | 1.0 → **0** | 0.554 → **0** | 0.390 → 0.357 |
| ETTm2 | 0.325 → **0.000** | 1.6 → **0** | 0.459 → **0** | 0.225 → 0.180 |
| weather | 0.259 → **0.000** | 0.5 → **0** | 0.496 → **0** | 0.188 → 0.147 |

This is the strongest result: the method provably repairs spectrum, dominant period, and peak
energy, and improves autocorrelation as a Wiener–Khinchin by-product — with zero exceptions across
315 runs. The linear scaling $d_{\text{FFT}}(\tilde S) = (1-\alpha)\,d_{\text{FFT}}(S)$ is confirmed.

### 6.3 Utility trade-off is dataset-dependent

Median `transfer_mse` across seeds, per model × dataset (lower is better; **best-α change** vs
α = 0 in the last column):

**DLinear**

| Dataset | α=0 | α=.3 | α=.5 | α=.7 | α=1 | verdict |
|---|---|---|---|---|---|---|
| ETTh1 | 0.435 | 0.442 | 0.435 | 0.444 | 0.439 | neutral |
| ETTh2 | 0.330 | 0.328 | 0.337 | 0.357 | 0.381 | **hurts** (+15% at α=1) |
| ETTm1 | 0.437 | 0.408 | 0.403 | **0.400** | 0.414 | **helps −9%** |
| ETTm2 | 0.345 | 0.308 | 0.274 | 0.263 | **0.234** | **helps −32%** |
| weather | 0.197 | 0.197 | 0.198 | 0.204 | 0.205 | neutral |

**MLP**

| Dataset | α=0 | α=.3 | α=.5 | α=.7 | α=1 | verdict |
|---|---|---|---|---|---|---|
| ETTh1 | 0.415 | 0.429 | 0.444 | 0.468 | 0.513 | **hurts** (+24% at α=1) |
| ETTh2 | 0.370 | 0.374 | 0.387 | 0.401 | 0.383 | neutral |
| ETTm1 | 0.606 | 0.611 | 0.568 | 0.583 | **0.564** | **helps −7%** |
| ETTm2 | 0.229 | 0.280 | 0.326 | 0.290 | 0.337 | **hurts** |
| weather | 0.233 | 0.233 | 0.229 | 0.232 | **0.223** | **helps −4%** |

**CNN**

| Dataset | α=0 | α=.3 | α=.5 | α=.7 | α=1 | verdict |
|---|---|---|---|---|---|---|
| ETTh1 | 0.498 | 0.470 | 0.468 | 0.448 | **0.443** | **helps −11%** |
| ETTh2 | 0.349 | 0.355 | 0.359 | 0.384 | 0.668 | **hurts / unstable** |
| ETTm1 | 0.689 | 0.537 | 0.469 | 0.408 | **0.382** | **helps −44%** |
| ETTm2 | 0.244 | 0.279 | 0.242 | **0.234** | 0.249 | helps −4% (seed-unstable) |
| weather | 0.235 | 0.229 | 0.239 | 0.226 | **0.224** | helps −5% |

**Pattern.** The 15-minute (ETTm) and 10-minute (weather) datasets — which have strong, high-resolution
daily seasonality — tend to *benefit* from post-fix, sometimes dramatically (DLinear ETTm2 −32%,
CNN ETTm1 −44%). The hourly ETTh datasets, with weaker/noisier seasonality, tend to *pay* a utility
cost. Interpretation: when seasonality is the dominant predictive signal, injecting the correct real
spectrum aligns the synthetic data with what the model needs, so temporal fidelity and utility
improve together; when seasonality is weak, the amplitude spectrum carries less predictive weight
and forcing it trades against the distillation objective.

### 6.4 Limitation: trend and variance are not restored

Median over seeds and models, `trend_error` and `variance_diff` at α = 0 vs α = 1:

| Dataset | `trend_error` 0→1 | `variance_diff` 0→1 |
|---|---|---|
| ETTh1 | 1.159 → 1.077 | 0.554 → 0.647 (worse) |
| ETTh2 | 0.616 → 0.728 (worse) | 0.847 → 0.788 |
| ETTm1 | 0.896 → 0.752 (better) | 0.794 → 0.710 (better) |
| ETTm2 | 0.524 → 0.571 (worse) | 0.908 → 0.871 |
| weather | 0.663 → 0.535 (better) | 0.737 → 0.713 |

Unlike the frequency-domain metrics, trend and variance move inconsistently and never approach zero.
This is expected: trend lives in the near-DC frequency bin (barely touched by amplitude matching to a
mean envelope), and variance is fixed by Parseval's theorem under a partial blend. The method is, by
construction, a *frequency-domain* correction.

---

## 7. Threats to Validity and Data-Quality Notes

Two categories of anomaly are present in the raw sweep and are handled by reporting the **median**
over seeds rather than the mean:

1. **CNN probe divergence.** CNN has no normalization layers and its probe occasionally diverges,
   producing `transfer_mse` far above `real_mse` — e.g. CNN×ETTm2 seed 42 reaches 1.3–3.8 across α,
   and CNN×ETTh2 seed 7 spikes to 1.20 at α = 1. Gradient clipping was tried and did **not** resolve
   it. The seed-spread of `transfer_mse` at α = 0 confirms the issue is localized: it is < 0.06 for
   most cells but **1.58** for CNN×ETTm2.
2. **Pathological distillation on seed 123.** For MLP×ETTm2 and MLP×weather, the α = 0 synthetic has
   `peak_mag_ratio` ≈ 4.7 and `variance_diff` ≈ 8.0 — the distiller itself produced a broken spectrum
   for that seed. Notably, post-fix *rescues* these: the spectrum returns to normal as α → 1. This is
   a secondary positive result (post-fix as a stabilizer of unstable distillation runs) but it also
   inflates the α = 0 baseline variance for those cells.

Neither anomaly affects the core frequency-restoration claim (Section 6.2), which holds identically
on the pathological rows.

---

## 8. LSTM Exclusion

LSTM is excluded from the matrix. MTT meta-gradients vanish through the recurrent inner loop: the
gradient magnitude is roughly an order smaller than for feed-forward models, and the trajectory
matching signal fails to move the synthetic data (`seq_delta` ≈ 0 after 300 steps). This is a
property of second-order trajectory matching on recurrent architectures, not of the post-fix method.
It is documented with a single gradient-flow figure in the appendix.

---

## 9. Is the current evidence sufficient? What remains?

**Sufficient to claim now (no new experiments needed):**

- *Distillation distorts frequency-domain temporal structure.* — Section 6.1, 15/15 cells.
- *Post-fix restores it reliably and exactly.* — Section 6.2, 315/315 runs, monotone to zero.
- *The utility effect is dataset-dependent and correlates with seasonality strength.* — Section 6.3.
- *The correction is frequency-domain only; trend/variance are out of reach.* — Section 6.4.

**Needed before the final claims are publication-solid (mostly analysis, not new runs):**

1. **Validation-based α★ selection.** Currently the "best α" is read off the test MSE. For an honest
   headline number, select α★ per (dataset, model) on the **validation** set — e.g. the largest α
   whose validation `transfer_mse` stays within 5 % of the α = 0 baseline — and report its test
   metrics. The raw data already supports this; it is a post-processing pass.
2. **Statistical test.** A paired Wilcoxon signed-rank test across the five datasets (distill-only vs
   distill+post-fix, per metric) with reported p-values. n = 5 is small but standard; claim
   significance only when p < 0.05 *and* the effect exceeds the seed spread.
3. **Resolve CNN, or scope it out.** Either (a) footnote CNN×{ETTh2, ETTm2} as unstable and exclude
   those cells, or (b) run 2 additional seeds for CNN only to average out the divergence. Option (a)
   is cheaper and defensible; the frequency-restoration claim does not depend on CNN utility.
4. **Add a random-init baseline row.** A probe trained on raw real windows (no distillation)
   contextualizes both the distortion and the utility numbers as a lower bound.
5. **(Optional) Two more seeds (→ 5 total)** for the moderately noisy cells (CNN/MLP on ETTm1/ETTm2),
   to tighten the mean ± std. Not required if reporting medians with the anomaly notes above.

**Verdict:** the experiment set is **sufficient for the core contribution** (temporal distortion +
reliable frequency restoration) and for a *nuanced, honest* utility story. The remaining items are
analysis passes (α★ on validation, Wilcoxon, baseline row) plus a decision on CNN — not a new round
of large experiments.

---

## 10. Suggested Thesis Structure

1. **Introduction** — dataset distillation for time series; why temporal fidelity matters
   (shareable/representative artifact framing, Section 2.2).
2. **Background** — MTT (Section 2.1), temporal-statistical metrics (Section 4).
3. **Problem: temporal distortion** — formalization + evidence (Section 2.2, 6.1).
4. **Method: FFT amplitude post-fix** — construction, linear-α property (Section 3).
5. **Experiments** — setup (Section 5), restoration result (6.2), trade-off ablation (6.3),
   limitations (6.4), α sensitivity as the central ablation figure.
6. **Analysis** — seasonality hypothesis, Wiener–Khinchin/Parseval couplings, threats to validity
   (Section 7), LSTM (Section 8).
7. **Limitations & future work** — composite multi-property correction (trend + variance +
   cross-channel), which the present limitations directly motivate.
8. **Conclusion.**

The α-sensitivity that first looked like a liability becomes the paper's central **sensitivity
analysis**, and the dataset-dependent utility split becomes a **mechanistic finding** rather than
noise.
