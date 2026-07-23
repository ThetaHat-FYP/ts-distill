# Phase-Aware Trajectory Matching to Mitigate Cross-Architecture Overfitting in MTT-Based Time-Series Distillation

## 1. Motivation

The standard Matching Training Trajectories (MTT) algorithm distills a synthetic dataset by matching the **parameters** of a student network to checkpoints along an expert's training trajectory at every step of distillation (`loss = ||θ_student − θ_target||²`). Because the synthetic data is optimized to reproduce a *specific architecture's* weight trajectory, the resulting synthetic set can encode architecture-specific inductive biases rather than genuine, transferable properties of the underlying time series. When a *different* student architecture is later trained on this synthetic data, performance degrades relative to training on real data — i.e., the synthetic data **overfits to the expert architecture used to generate it**, hurting cross-architecture generalization.

This motivated an investigation into whether the *type* of matching signal used during distillation — parameter space vs. output/prediction space — is responsible for this cross-architecture degradation, and whether the two matching styles could be combined to get the benefits of both.

## 2. Key Insight: Parameter Matching vs. Prediction Matching

- **Parameter matching** (original MTT) forces the synthetic data's induced gradient updates to reproduce the exact weight trajectory of the expert. This is a strong, architecture-specific constraint — it is sensitive to how a particular architecture parameterizes its function, not just what function it computes.
- **Prediction matching** instead matches the *input-output behaviour* of the expert (`loss = ||f_student(X) − f_expert(X)||²`), which is architecture-agnostic — any student capable of approximating the expert's predictions can be matched, regardless of how its weights are structured.

The hypothesis: **early in expert training**, the trajectory is dominated by large, architecture-general moves (coarse feature learning), so parameter matching is safe and informative. **Late in training**, as the expert approaches convergence, its weight trajectory becomes increasingly shaped by architecture-specific curvature/fine-tuning — continuing to match parameters here is what injects the architecture-specific overfitting. Switching to prediction matching in this late phase should preserve the *functional* behaviour being distilled without forcing an incompatible weight geometry onto other architectures.

This gives rise to **Phase-Aware Trajectory Matching**: a hybrid distillation objective that uses parameter matching in the *early phase* of the expert trajectory and prediction matching in the *late phase*, split at a detected **phase boundary** T⁺.

## 3. Detecting the Phase Boundary (T⁺)

To find T⁺ in a principled, per-(dataset, architecture, seed) way rather than picking an arbitrary epoch, the boundary is detected from the expert's **validation loss curve** during its training run (`h_detect_phase_boundary_valloss.py`):

1. Train the expert for `expert_epochs` epochs, recording a weight checkpoint (`SimpleRecorder`) **and** the validation MSE at the end of every epoch.
2. Smooth the raw per-epoch validation loss with a centred rolling average (`smoothing_window = 5`).
3. Track the best (lowest) smoothed validation loss seen so far. An epoch counts as an "improvement" only if it is at least `min_delta_frac = 1%` (relative) better than the current best — this avoids treating tiny numerical fluctuations as real progress.
4. Once `patience = 5` consecutive epochs fail to improve on the best-so-far, the expert is judged to have entered its **plateau phase**, and T⁺ is set to the epoch index of that best-so-far checkpoint (i.e., the epoch where the plateau *began*, not where it was detected).

This is run across all datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather), all architectures (DLinear, MLP, CNN), and 3 seeds (7, 42, 123), producing a boundary table (`all_seeds_phase_boundaries_valloss_new.csv`) keyed by `(dataset, model, seed)`. This is referred to as **Method B** (validation-loss plateau) as opposed to an earlier parameter-distance-based detector (**Method A**, `h_detect_phase_boundary.py` — "when do the expert's weights stop moving?"). Method B was adopted as the primary boundary detector because it ties the switch point directly to a generalization signal (validation loss) rather than a raw weight-movement heuristic.

## 4. Phase-Aware Matching Algorithm

Given the detected T⁺ for a given expert trajectory, the `PhaseAwareMTTDistiller` (used in place of the standard `MTTDistiller`) applies:

- **Epochs < T⁺ (early phase):** standard parameter matching against the expert's weight checkpoints (identical to original MTT).
- **Epochs ≥ T⁺ (late phase):** prediction matching — the student's outputs on the synthetic sequence are matched against the expert's outputs, instead of matching weights.

Both `param_mtt` (baseline) and `phase_aware_mtt` (proposed) are run from **the same expert trajectory and the same synthetic initialization** (RNG state is explicitly saved right after expert training and restored before each distillation call) so that the matching-loss type is the *only* variable that differs between the two conditions — isolating its effect from any confound due to different random seeds or initial synthetic samples.

## 5. Experimental Setup

- **Baseline (`h1_cross_arch_baseline.py`):** establishes the cross-architecture overfitting problem itself. Distills synthetic data from one expert architecture, then evaluates it by training *every* student architecture (DLinear, MLP, CNN, LSTM) on that same synthetic set and comparing to a real-data baseline via `mse_ratio = transfer_mse / real_mse`. A same-architecture (diagonal) vs. cross-architecture (off-diagonal) MSE-ratio gap is the evidence of overfitting.
- **Phase-aware evaluation (`h2_cross_arch_phase_aware_early_stop.py`):** repeats the same expert → synthetic → cross-architecture-student evaluation matrix, but generates synthetic data with both `param_mtt` and `phase_aware_mtt` from the identical expert trajectory, and compares their `mse_ratio` matrices.
- **Distillation config (shared by both methods):**
  - `expert_epochs = 80`, SGD (`lr=0.01`, `momentum=0.9`)
  - `n_distill_steps = 1000`
  - `n_synthetic = 384` synthetic points
  - `synthetic_lr = 5.0`, `student_lr = 0.1`, `student_steps = 20`, `snapshot_student_steps = 50`, `trajectory_gap = 5`
  - Evaluation: early stopping on real validation loss for both the real-data baseline and the synthetic-trained student (`eval_max_epochs = 300`, `patience = 10`), so results are comparable across differently-sized real/synthetic training sets.
- **Datasets:** ETTh1, ETTh2, ETTm1, ETTm2, Weather (`seq_len = pred_len = 96`).
- **Architectures:** DLinear, MLP, CNN as experts/students (LSTM included in the earlier cross-arch baseline matrix).
- **Seeds:** 7, 42, 123.

## 6. Implementation Reference

| File | Role |
|---|---|
| `h1_cross_arch_baseline.py` | Establishes cross-architecture overfitting under standard (parameter-matching) MTT |
| `h_detect_phase_boundary_valloss.py` | Detects the phase boundary T⁺ per (dataset, architecture, seed) from the expert's validation-loss plateau |
| `h2_cross_arch_phase_aware_early_stop.py` | Runs the phase-aware vs. standard MTT comparison across the full cross-architecture student matrix, using the detected T⁺ values |

## 7. Summary

The overfitting problem is framed as a consequence of parameter matching remaining active through the *entire* expert trajectory, including the late-training regime where the expert's weight geometry becomes architecture-specific. By detecting, per expert run, the epoch at which validation loss plateaus (T⁺) and switching the matching objective from parameter-space to prediction-space beyond that point, the distillation process retains a precise, cheap-to-match signal early on while avoiding the transfer of architecture-specific late-phase weight structure — aiming to reduce the cross-architecture MSE-ratio gap without sacrificing same-architecture distillation quality.
