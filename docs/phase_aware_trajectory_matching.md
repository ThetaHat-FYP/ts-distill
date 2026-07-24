# Phase-Aware Trajectory Matching — Method Reference

**Framework**: `ts-distill` | **Module**: `src/ts_distill/distillation_core/distillation_algorithm/`

This document describes the design, motivation, and exact mathematical
formulation of Phase-Aware Trajectory Matching, the core contribution of
Thread 3 (Phase-Aware Trajectory Matching) of the FYP. It is written to be
dropped directly into the research report: each section states *what* the
method does, *why* it does it, and the *formula* that implements it, with
pointers to the exact source file.

---

## 1. Background — Standard Matching Training Trajectories (MTT)

MTT distillation does not try to match the real data distribution directly.
Instead, it optimises a small synthetic sequence $S$ so that a student model
trained on $S$ follows the *same path through parameter space* as an expert
model trained on real data.

**File**: [`mtt.py`](../src/ts_distill/distillation_core/distillation_algorithm/mtt.py)

### 1.1 Expert trajectory

An expert model is trained on real data for `expert_epochs` epochs. Its
weights are checkpointed every epoch by `SimpleRecorder`, producing an
ordered trajectory:

$$
\mathcal{T} = \{\theta_0, \theta_1, \dots, \theta_E\}
$$

where $\theta_e$ is the expert's parameter vector after epoch $e$.

### 1.2 Checkpoint-pair sampling

At each outer-loop distillation step, a pair of checkpoints separated by a
fixed **trajectory gap** $g$ (`trajectory_gap` in config) is drawn at random:

$$
(\theta_{\text{start}},\ \theta_{\text{target}}) = (\theta_t,\ \theta_{t+g}), \qquad t \sim \mathrm{Uniform}\{0, \dots, E-g\}
$$

$\theta_{\text{start}}$ is where the student is initialised; $\theta_{\text{target}}$
is where the expert ended up $g$ real-data epochs later.

### 1.3 Inner loop — differentiable student unrolling

A fresh student (same architecture as the expert) is initialised at
$\theta_{\text{start}}$ and trained on the synthetic sequence $S$ for $K$
gradient-descent steps (`student_steps`), **with the computation graph kept
alive** (`torch.autograd.grad(..., create_graph=True)`), so gradients can
later flow all the way back into $S$:

$$
\theta^{(0)} = \theta_{\text{start}}, \qquad
\theta^{(k+1)} = \theta^{(k)} - \eta_s \, \nabla_{\theta^{(k)}}\, \mathcal{L}_{\text{MSE}}\!\left(f(x_k;\theta^{(k)}),\, y_k\right), \quad k = 0,\dots,K-1
$$

where $(x_k, y_k)$ is a mini-batch of (look-back, forecast) windows sliced
from $S$, and $\eta_s$ is the inner-loop learning rate (`student_lr`). The
final student parameters are $\theta_{\text{final}} = \theta^{(K)}$.

### 1.4 Outer loss — parameter matching

The outer loss compares $\theta_{\text{final}}$ to $\theta_{\text{target}}$,
normalised by how far the expert itself moved over the same $g$ epochs:

$$
\mathcal{L}_{\text{param}} = \sum_{p} \left\lVert \theta_{\text{final}}^{(p)} - \theta_{\text{target}}^{(p)} \right\rVert_F^2,
\qquad
D_{\text{param}} = \sum_{p} \left\lVert \theta_{\text{start}}^{(p)} - \theta_{\text{target}}^{(p)} \right\rVert_F^2
$$

$$
\boxed{\ \mathcal{L}_{\text{grand}} = \dfrac{\mathcal{L}_{\text{param}}}{D_{\text{param}} + \varepsilon}\ } \qquad \varepsilon = 10^{-12}
$$

summed over every named parameter tensor $p$ shared between student and
expert (`MSEMatcher.calculate_loss`, [`mse_matcher.py`](../src/ts_distill/trajectory/matcher/mse_matcher.py)).
If the student closes the parameter gap as effectively as the expert does on
real data, $\mathcal{L}_{\text{grand}} \to 1$.

### 1.5 Outer update

$\mathcal{L}_{\text{grand}}$.backward() propagates through every inner step
(second-order gradients, enabled by `create_graph=True`) back into $S$:

$$
S \leftarrow S - \eta_o \, \nabla_S\, \mathcal{L}_{\text{grand}} \qquad \text{(SGD, momentum = 0.5)}
$$

This is repeated for `n_distill_steps` outer iterations.

---

## 2. Motivation for Phase-Awareness

Standard MTT applies **parameter matching** ($\S1.4$) uniformly across the
*entire* expert trajectory, from random initialisation to convergence. Two
empirical/qualitative observations motivate splitting this into phases:

1. **Early phase** — the expert is moving rapidly through parameter space.
   Parameter-space deltas are large and carry a strong, directional training
   signal. Matching them drives fast, informative updates to the synthetic
   data.
2. **Late phase** — the expert has largely converged and is fine-tuning.
   Remaining parameter-space movement is small and increasingly encodes
   **architecture-specific noise** — idiosyncratic weight adjustments tied to
   the expert's own network topology (e.g. LSTM gate weights, CNN kernel
   layout) rather than to what the model has learned to predict. Matching
   parameters in this regime risks distilling architecture-specific bias
   into the synthetic data rather than transferable knowledge.

**Hypothesis (H3 of the thread):** switching the matching objective from
parameter-space to **prediction-space** once the expert enters its late
(fine-tuning) phase anchors the synthetic data to *functional* behaviour —
what the model predicts, not how it is parameterised — which is
architecture-agnostic by construction and should reduce cross-architecture
bias in the resulting synthetic dataset while preserving same-architecture
distillation quality (H5).

This is implemented by `PhaseAwareMTTDistiller`.

---

## 3. Phase Boundary Detection — $T^{+}$

**File**: [`valloss_detector.py`](../src/ts_distill/trajectory/phase_detector/valloss_detector.py)

Before phase-aware matching can switch objectives, it needs a **boundary
epoch** $T^{+}$ that separates "early" from "late". $T^{+}$ is detected
automatically, on the fly, from the expert's own per-epoch validation-loss
curve — no manual tuning per architecture is required.

**Definition:** $T^{+}$ is the epoch of the best (lowest) *smoothed*
validation loss observed **before** a sustained plateau begins, where a
plateau is `patience` consecutive epochs that each fail to improve on the
best-so-far smoothed loss by at least `min_delta_frac` (relative).

### 3.1 Smoothing

The raw per-epoch validation loss sequence $\{v_1, \dots, v_E\}$ is
smoothed with a centred rolling average of window $w$ (`smoothing_window`):

$$
\tilde v_i = \frac{1}{|\,[\,\ell_i,\,h_i)\,|}\sum_{j=\ell_i}^{h_i-1} v_j,
\qquad \ell_i = \max(0,\, i - \lfloor w/2 \rfloor),\ \ h_i = \min(E,\, i + \lfloor w/2 \rfloor + 1)
$$

### 3.2 Plateau search

$$
v^{\*}_1 = \tilde v_1,\quad n_{\text{fail}} = 0
$$

For $i = 2, \dots, E$:

$$
\text{if } \tilde v_i < v^{\*}_{\text{best}} \cdot (1 - \delta):\quad v^{\*}_{\text{best}} \leftarrow \tilde v_i,\ \ i_{\text{best}} \leftarrow i,\ \ n_{\text{fail}} \leftarrow 0
$$
$$
\text{else:}\quad n_{\text{fail}} \leftarrow n_{\text{fail}} + 1;\quad \text{if } n_{\text{fail}} \ge \text{patience}:\ \ T^{+} \leftarrow i_{\text{best}}\ ;\ \textbf{break}
$$

where $\delta =$ `min_delta_frac`. If no plateau of length `patience` is
ever found, $T^{+} = \text{None}$ and phase-aware matching degenerates to
standard parameter matching for the whole run (identical to
`MTTDistiller`).

### 3.3 Default configuration (`PHASE_BOUNDARY_CONFIG`)

| Parameter | Value | Meaning |
|---|---|---|
| `smoothing_window` | 5 | Rolling-average window applied before plateau search |
| `patience` | 5 | Consecutive non-improving epochs that define a plateau |
| `min_delta_frac` | 0.01 | Minimum relative improvement (1%) to reset the plateau counter |

Reference values found earlier via offline analysis (interim report):
LSTM $T^{+}=48$, MLP $T^{+}=30$, CNN $T^{+}=58$, DLinear — no clear
plateau (matches the "$T^{+}=\text{None}$" fallback above). In the current
pipeline this detection runs **on the fly** for every (dataset, architecture,
seed) combination rather than using these fixed values, so it adapts per run.

---

## 4. Phase-Aware Matching Loss

**File**: [`phase_aware_mtt.py`](../src/ts_distill/distillation_core/distillation_algorithm/phase_aware_mtt.py)

`PhaseAwareMTTDistiller` extends `MTTDistiller`, overriding only the
checkpoint-pair sampling and the outer-loop loss. The inner loop ($\S1.3$)
is inherited unchanged.

### 4.1 Same-phase checkpoint-pair constraint

A checkpoint pair $(\theta_{\text{start}}, \theta_{\text{target}})$ at steps
$(t, t+g)$ is only valid if **both** endpoints lie in the same phase
relative to $T^{+}$:

$$
\text{is\_late}(t) = \mathbb{1}[t \ge T^{+}]
$$

$$
\text{valid pair} \iff \text{is\_late}(t) = \text{is\_late}(t+g)
$$

A pair that *straddles* the boundary (one endpoint early, one late) is
ambiguous and is **discarded**; a new pair is resampled (up to 1000 attempts,
falling back to the start checkpoint's phase if the trajectory is too short
to find a clean pair — this is a defensive fallback, not expected to trigger
in practice given $E \gg g$).

### 4.2 Early phase → parameter matching

Identical to standard MTT ($\S1.4$):

$$
\mathcal{L}_{\text{early}} = \frac{\displaystyle\sum_{p}\left\lVert\theta_{\text{final}}^{(p)}-\theta_{\text{target}}^{(p)}\right\rVert_F^2}{\displaystyle\sum_{p}\left\lVert\theta_{\text{start}}^{(p)}-\theta_{\text{target}}^{(p)}\right\rVert_F^2 + \varepsilon}
$$

### 4.3 Late phase → prediction matching

Let $X$ be a batch of $B$ look-back windows sampled from the synthetic
sequence $S$ (inputs are **detached** — gradient must reach $S$ only through
$\theta_{\text{final}}$, i.e. through the inner loop, not directly through
the input values). Let $f(X;\theta)$ denote the model's forecast under
parameters $\theta$, evaluated via a functional (stateless) forward pass:

$$
\mathcal{L}_{\text{pred}} = \mathrm{MSE}\big(f(X;\theta_{\text{final}}),\ f(X;\theta_{\text{target}})\big)
$$

$$
D_{\text{pred}} = \mathrm{MSE}\big(f(X;\theta_{\text{start}}),\ f(X;\theta_{\text{target}})\big)
$$

$$
\boxed{\ \mathcal{L}_{\text{late}} = \dfrac{\mathcal{L}_{\text{pred}}}{\max\!\big(D_{\text{pred}},\ \delta\big)}\ } \qquad \delta = 10^{-4}
$$

$f(X;\theta_{\text{target}})$ and $f(X;\theta_{\text{start}})$ are computed
under `torch.no_grad()` — they are fixed reference points, not
differentiated. $D_{\text{pred}}$ is additionally **detached** before the
clamp, so it acts purely as a (stop-gradient) normalisation constant.

> **Note on normalisation semantics.** $\mathcal{L}_{\text{param}}$ ($\S4.2$)
> is a **sum** of squared differences over all parameters (`MSEMatcher`
> uses `torch.sum`), whereas $\mathcal{L}_{\text{pred}}$ ($\S4.3$) is a
> **mean** squared error over the batch (`F.mse_loss`, default
> `reduction='mean'`). The two losses are therefore on different absolute
> scales; this is immaterial to optimisation because each is independently
> normalised by its own baseline distance ($D_{\text{param}}$ or
> $D_{\text{pred}}$) before being backpropagated, but it means the two
> phases' loss curves are **not directly comparable** in raw magnitude when
> plotted together — only the normalised ratio is meaningful.

### 4.4 Why clamp instead of the $\varepsilon$ used in parameter matching

In the late-phase plateau, consecutive checkpoints make *nearly identical*
predictions (the expert has converged), so $D_{\text{pred}} \to 0$. A small
additive $\varepsilon$ (as used for $D_{\text{param}}$) is insufficient to
stabilise a ratio whose denominator can be several orders of magnitude
smaller than $\varepsilon^{1/2}$ in this regime; a hard floor
$\delta = 10^{-4}$ is used instead to prevent $\mathcal{L}_{\text{late}}$
from exploding and driving $S$ to `NaN`.

### 4.5 Additional stabilisation — gradient clipping

Because the late-phase loss can still carry a large meta-gradient even after
denominator clamping, `PhaseAwareMTTDistiller` (and `PredictiveMTTDistiller`,
$\S5$) clip the gradient on the synthetic data **after** `.backward()` and
**before** the optimiser step — a stabilisation not present in the
vanilla `MTTDistiller`:

$$
\nabla_S \mathcal{L} \leftarrow \nabla_S \mathcal{L} \cdot \min\!\left(1,\ \frac{1}{\lVert \nabla_S \mathcal{L} \rVert_2}\right)
$$

i.e. `torch.nn.utils.clip_grad_norm_([synthetic_data], max_norm=1.0)`.

---

## 5. Relationship to `PredictiveMTTDistiller`

**File**: [`predictive_mtt.py`](../src/ts_distill/distillation_core/distillation_algorithm/predictive_mtt.py)

`PredictiveMTTDistiller` implements exactly $\S4.3$'s prediction-matching
loss, but applies it to **every** outer step regardless of trajectory phase
(no boundary, no resampling). It exists as an ablation / control condition:
comparing `param_mtt` (`MTTDistiller`), `pred_mtt`
(`PredictiveMTTDistiller`, prediction matching throughout), and
`phase_aware_mtt` (`PhaseAwareMTTDistiller`, switches at $T^{+}$) isolates
*when* prediction matching helps, versus *whether* it helps at all.

| Distiller | Early-phase loss | Late-phase loss | Phase boundary used? |
|---|---|---|---|
| `MTTDistiller` | $\mathcal{L}_{\text{param}}$ | $\mathcal{L}_{\text{param}}$ | No |
| `PredictiveMTTDistiller` | $\mathcal{L}_{\text{pred}}$ | $\mathcal{L}_{\text{pred}}$ | No |
| `PhaseAwareMTTDistiller` | $\mathcal{L}_{\text{param}}$ | $\mathcal{L}_{\text{pred}}$ | Yes — $T^{+}$ from $\S3$ |

When `phase_boundary = None`, `PhaseAwareMTTDistiller` collapses to
identical behaviour to `MTTDistiller` (no resampling, no phase switch) —
this is the built-in fallback used whenever $T^{+}$ detection fails to find
a plateau (e.g. DLinear).

---

## 6. End-to-End Algorithm (Phase-Aware Outer Loop)

```
Input: expert trajectory 𝒯 = {θ_0, ..., θ_E}, phase boundary T⁺ (or None),
       synthetic init S₀, n_distill_steps, trajectory_gap g

S ← S₀   (requires_grad = True)
optimizer ← SGD([S], lr=η_o, momentum=0.5)

for step in 1..n_distill_steps:
    # 1. Sample a same-phase checkpoint pair
    repeat up to 1000×:
        (θ_start, θ_target) ← sample_checkpoint_pair(𝒯, gap=g)
        is_late ← (θ_start.step ≥ T⁺) == (θ_target.step ≥ T⁺)
    until is_late is well-defined (both endpoints agree)

    # 2. Fresh student, same architecture as expert
    student ← model_factory()

    # 3. Differentiable inner loop (K steps, create_graph=True)
    θ_final, θ_start ← unroll(student, S, θ_start)

    # 4. Phase-conditioned loss
    if is_late(θ_start.step):
        L ← ‖f(X;θ_final) − f(X;θ_target)‖²_MSE / max(‖f(X;θ_start) − f(X;θ_target)‖²_MSE, 1e-4)
    else:
        L ← Σ‖θ_final − θ_target‖² / (Σ‖θ_start − θ_target‖² + 1e-12)

    # 5. Meta-gradient step
    L.backward()                                   # through the whole inner loop into S
    clip_grad_norm_([S], max_norm=1.0)
    optimizer.step()

    # 6. (optional) validation snapshot every val_snapshot_every steps
    #    — train a probe student on S, score on real val set, keep best S

return best S (by validation MSE) or final S
```

---

## 7. Pipeline Integration

**File**: [`example/experiments/experiment_matrix.py`](../example/experiments/experiment_matrix.py)

Phase-aware matching is opt-in via `USE_PHASE_AWARE_MATCHING = True`. When
enabled, `run_single_experiment()`:

1. Attaches a `ValLossRecorderCallback` to the expert `Trainer`, which calls
   `eval_epoch(val_loader)` at the end of every epoch to capture the raw
   validation-loss curve $\{v_1,\dots,v_E\}$ (one extra forward pass per
   epoch over the val set — only paid when phase-aware matching is on).
2. Runs `ValLossPlateauDetector(**PHASE_BOUNDARY_CONFIG).detect(...)` on
   that curve to obtain $T^{+}$ ($\S3$).
3. Constructs `PhaseAwareMTTDistiller(..., phase_boundary=T⁺)` in place of
   `MTTDistiller` and runs `distill()` exactly as in the baseline pipeline.

No offline phase-boundary sweep is required — detection happens on the fly,
per (dataset, architecture, seed), from that run's own expert trajectory.

### Key hyperparameters (shared with the baseline MTT config)

| Parameter | Value |
|---|---|
| `expert_epochs` | 80 |
| `trajectory_gap` ($g$) | 5 |
| `n_distill_steps` | 1000 |
| `n_synthetic` ($\lvert S\rvert$) | 384 |
| `student_steps` ($K$) | 20 |
| `student_lr` ($\eta_s$) | 0.01 |
| `synthetic_lr` ($\eta_o$) | 5.0 |
| `seq_len` / `pred_len` | 96 / 96 |

---

## 8. Summary — Design Rationale

| Design choice | Reason |
|---|---|
| Detect $T^{+}$ from validation loss, not training loss | Training loss keeps falling under overfitting; validation-loss plateau marks where the expert stops generalising further, which is the more meaningful "phase change" for matching purposes. |
| Discard/resample straddling checkpoint pairs | A pair that spans both phases has no well-defined "which loss applies" answer; forcing same-phase pairs keeps each outer step's objective unambiguous. |
| Prediction matching only in late phase | Early-phase parameter deltas are large and informative; late-phase parameter deltas are small and architecture-specific — switching preserves the early signal while avoiding late-phase architectural noise (H3). |
| Hard floor ($10^{-4}$) instead of additive $\varepsilon$ for $D_{\text{pred}}$ | Late-phase predictions from adjacent checkpoints can be near-identical, making $D_{\text{pred}}$ collapse toward zero; an additive $\varepsilon$ is too small to prevent ratio blow-up in that regime. |
| Gradient-norm clipping on $S$ (phase-aware / predictive variants only) | Extra safeguard against the same near-zero-denominator instability propagating into an exploding meta-gradient. |
| `phase_boundary=None` ⇒ identical to `MTTDistiller` | Guarantees a safe, well-tested fallback for architectures (e.g. DLinear) where no plateau is detected, without special-casing them elsewhere in the pipeline. |
