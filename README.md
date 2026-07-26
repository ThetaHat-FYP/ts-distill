# ts-distill

Modular **time-series dataset distillation** with post-hoc temporal fidelity restoration.

Compress a large time-series training set into a small synthetic one that still trains a
forecaster to comparable accuracy — then repair the temporal properties that distillation
destroys.

---

## Why

Trajectory-matching distillation (MTT) optimises synthetic data purely for **training
utility**. The result trains a good model, but its spectrum, periodicity, and
autocorrelation no longer resemble the real series. That matters whenever the synthetic
data is inspected, shared, or reused for anything other than fitting one model.

`ts-distill` separates the two concerns:

1. **Distil** for utility (MTT / phase-aware MTT).
2. **Post-fix** the frequency content afterwards (`FFTAmplitudePostFix`), trading a
   controllable amount of utility for temporal realism.
3. **Mix** a small fraction of real data back in and measure where the accuracy /
   compression trade-off is best.

---

## Install

```bash
pip install -e .              # library only
pip install -e ".[examples]"  # also installs scikit-learn, used by example/ scripts
```

Requires Python >= 3.9.

---

## Quick start

```python
import torch
from ts_distill import (
    configure_logging, CSVDataLoader, create_model, Trainer, MiniBatchLoader,
    SimpleRecorder, SimpleCallback, MSEMatcher, MTTDistiller,
    RandomSampleInitializer, FFTAmplitudePostFix,
)

configure_logging()          # library is silent until you ask for progress
torch.manual_seed(42)

# 1. Train an expert and record its trajectory
recorder = SimpleRecorder(record_every=1)
expert   = create_model('MLP', seq_len=96, pred_len=96, in_features=7, model_kwargs={})
Trainer(
    model=expert,
    optimizer=torch.optim.SGD(expert.parameters(), lr=0.01, momentum=0.9),
    criterion=torch.nn.MSELoss(), device='cpu', seq_len=96,
).fit(
    dataloader=MiniBatchLoader(train_windows, batch_size=64),
    epochs=80, callbacks=[recorder, SimpleCallback()],
)

# 2. Initialise + distil
initializer = RandomSampleInitializer()
distiller   = MTTDistiller(
    initializer=initializer, matcher=MSEMatcher(),
    model_factory=lambda: create_model('MLP', 96, 96, 7, {}),
    expert_recorder=recorder,
    synthetic_lr=5.0, student_lr=0.01, student_steps=20,
)
synthetic = distiller.distill(
    synthetic_init=initializer.initialize_sequence(raw_train_data, 384),
    n_steps=300, val_data=val_windows,
)

# 3. Restore the amplitude spectrum (alpha = blend strength in [0, 1])
synthetic = FFTAmplitudePostFix(alpha=0.3).apply(synthetic, raw_train_data)
```

A complete, runnable version — including data loading, hybrid mixing, and temporal
metrics — lives in **[`example/demo_lib.py`](example/demo_lib.py)**:

```bash
python example/demo_lib.py
```

---

## Components

| Package | What it provides |
|---|---|
| `ts_distill.data_pipeline` | `CSVDataLoader`, `get_data_splits`, `make_windows`, `MiniBatchLoader` |
| `ts_distill.models` | `create_model` -> `MLP`, `CNN`, `LSTM`, `DLinear` (interchangeable shapes) |
| `ts_distill.trainer` | `Trainer` (early stopping), callbacks |
| `ts_distill.trajectory` | `SimpleRecorder`, `MSEMatcher`, `ValLossPlateauDetector` |
| `ts_distill.distillation_core` | `MTTDistiller`, `PhaseAwareMTTDistiller`, 3 initializers |
| `ts_distill.post_processing` | `FFTAmplitudePostFix` |
| `ts_distill.evaluation` | `Evaluator`, `BaseHybridEvaluator`, `find_optimal_ratio` |
| `ts_distill.metrics` | `MetricAggregator` — 9 temporal fidelity metrics |
| `ts_distill.visualization` | `TimeSeriesVisualizer` |

### Initializers

All three pick **one contiguous block** of `n_synthetic` timesteps; they differ only in
the selection rule.

| Initializer | Rule | Character |
|---|---|---|
| `RandomSampleInitializer` | uniformly random start | high seed-to-seed variance |
| `GeometrySequenceInitializer` | medoid (closest to centroid) | deterministic, flattest block |
| `UncertaintySampleInitializer` | highest volatility | deterministic, most extreme block |

Because the block only sets the starting point of a non-convex optimisation, compare
strategies on **seed variance** and **downstream hybrid gain**, not on a single-seed
transfer MSE.

### Post-fix

For each channel `c`, keep the synthetic phase and blend the amplitude toward the real
data's segment-averaged envelope `A_c`:

```
S_c = F^-1[ ((1 - alpha)*|F(S_c)| + alpha*A_c) * e^(i*phi_c) ]
```

`alpha = 0` leaves the distilled data untouched; `alpha = 1` fully adopts the real
amplitude spectrum. Frequency distance falls linearly:
`d_FFT(S_fixed) = (1 - alpha) * d_FFT(S)`.

Because spectrum and autocorrelation are a Fourier pair (Wiener–Khinchin), the ACF
metrics improve for free. Total variance is fixed by Parseval's theorem, so a partial
blend cannot correct `variance_diff` or the DC/trend term — an honest limitation.

---

## Reproducibility

Every random draw comes from the global torch RNG. `torch.manual_seed(...)` makes runs
reproducible **only if the order of random calls is unchanged** — anything that iterates
a `DataLoader` consumes RNG and shifts every later draw. When inserting such a step,
preserve the stream:

```python
state = torch.get_rng_state()
...                                # step that consumes RNG
torch.set_rng_state(state)
```

This is why the demos evaluate the real baseline *after* distillation, and why the
phase-aware validation pass is RNG-guarded.

---

## Logging

The library never writes to stdout on its own. Progress goes through the standard
`logging` module under the `ts_distill` logger:

```python
from ts_distill import configure_logging
configure_logging('INFO')      # or 'DEBUG' / 'WARNING'
```

---

## Repository layout

```
src/ts_distill/     the library
example/            runnable demos and experiment scripts
docs/               method write-ups, metric reference, result figures
```

---

## License

MIT — see [LICENSE](LICENSE).
