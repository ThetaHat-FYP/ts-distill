"""
Gradient Flow Diagnostic
========================
Runs a minimal MTT outer loop (no full evaluation) and plots four diagnostics:

  1. param_loss vs param_dist  — is the student closing the gap on the expert?
  2. grand_loss                — normalised trajectory matching loss (should drop)
  3. grad_norm                 — L2 norm of meta-gradient flowing into synthetic_data
                                 (key: if this is ~1e-6, optimization is broken)
  4. seq_delta                 — mean absolute drift of synthetic_data from init
                                 (key: if this is ~0 after 30 steps, nothing is learning)

Subplot 5 overlays the first channel of synthetic_data at step 0 vs step N
against the first channel of real training data so you can see if synthetic
visually changes at all.

Run from the project root:
    python -m example.experiments.debug_gradient_flow

Edit DATASET and MODEL below to switch targets.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — saves PNG instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent.parent))

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader
from ts_distill.distillation_core.initializer.random_sample_initializer import RandomSampleInitializer
from ts_distill.models.factory import create_model
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATASET     = 'ETTm1'   # change to 'weather', 'ETTh1', etc.
MODEL       = 'MLP'     # change to 'CNN', 'LSTM', 'DLinear'
EXPERT_EP   = 20        # keep small so this script finishes fast
OUTER_STEPS = 30        # MTT outer loop steps
STUDENT_STEPS = 20
STUDENT_LR    = 0.1
SYNTHETIC_LR  = 5
SEQ_LEN       = 96
PRED_LEN      = 96
N_SYNTHETIC   = 384
BATCH_SIZE    = 64
TRAJ_GAP      = 5

DATASET_CONFIGS = {
    'ETTm1': {'csv_path': 'example/ETTm1.csv', 'split_mode': 'benchmark_borders',
               'border1s': [0, 12*30*96-96, 12*30*96+4*30*96-96],
               'border2s': [12*30*96, 12*30*96+4*30*96, 12*30*96+8*30*96],
               'in_features': 7},
    'ETTh1': {'csv_path': 'example/ETTh1.csv', 'split_mode': 'benchmark_borders',
               'border1s': [0, 12*30*24-96, 12*30*24+4*30*24-96],
               'border2s': [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24],
               'in_features': 7},
    'weather': {'csv_path': 'example/weather.csv', 'split_mode': 'ratios',
                'train_ratio': 0.7, 'val_ratio': 0.1, 'in_features': 21},
}
MODEL_KWARGS = {'MLP': {}, 'CNN': {}, 'LSTM': {'hidden_dim': 16, 'num_layer': 1},
                'DLinear': {'individual': False}}

# ── Setup ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
window_size = SEQ_LEN + PRED_LEN
dataset_cfg = DATASET_CONFIGS[DATASET]
in_features = dataset_cfg['in_features']

df_raw = pd.read_csv(dataset_cfg['csv_path'])
values = df_raw.iloc[:, 1:].values.astype(np.float32)

train_start, train_end, *_ = get_data_splits(
    values=values, window_size=window_size, seq_len=SEQ_LEN, dataset_cfg=dataset_cfg
)
scaler = StandardScaler()
scaler.fit(values[train_start:train_end])
data           = scaler.transform(values)
raw_train      = torch.tensor(data[train_start:train_end], dtype=torch.float32)
train_windows  = torch.tensor(make_windows(data[train_start:train_end], window_size), dtype=torch.float32)

def make_model():
    return create_model(MODEL, SEQ_LEN, PRED_LEN, in_features, MODEL_KWARGS[MODEL])

# ── Expert training ───────────────────────────────────────────────────────────
print(f"\nTraining expert ({DATASET} × {MODEL}) for {EXPERT_EP} epochs...")
recorder     = SimpleRecorder(record_every=1)
expert_model = make_model()
Trainer(
    model     = expert_model,
    optimizer = torch.optim.SGD(expert_model.parameters(), lr=0.01, momentum=0.9),
    criterion = torch.nn.MSELoss(),
    device    = device,
    seq_len   = SEQ_LEN,
).fit(
    dataloader = MiniBatchLoader(train_windows, batch_size=BATCH_SIZE),
    epochs     = EXPERT_EP,
    callbacks  = [recorder, SimpleCallback()],
)
print(f"Expert recorded {len(recorder.trajectory)} checkpoints.\n")

# ── Synthetic init ────────────────────────────────────────────────────────────
synthetic_data = RandomSampleInitializer().initialize_sequence(raw_train, N_SYNTHETIC)
synthetic_data = synthetic_data.detach().clone().to(device)
synthetic_data.requires_grad_(True)
syn_init_np    = synthetic_data.detach().cpu().numpy().copy()

optimizer_img = torch.optim.SGD([synthetic_data], lr=SYNTHETIC_LR, momentum=0.5)
matcher       = MSEMatcher()
criterion     = torch.nn.MSELoss()

# ── Diagnostic outer loop ─────────────────────────────────────────────────────
log = {'step': [], 'param_loss': [], 'param_dist': [], 'grand_loss': [],
       'grad_norm': [], 'seq_delta': []}

print(f"Running {OUTER_STEPS} outer-loop steps...")
print(f"{'Step':>5}  {'param_loss':>12}  {'param_dist':>12}  "
      f"{'grand_loss':>12}  {'grad_norm':>12}  {'seq_delta':>12}")
print("-" * 75)

for step in range(OUTER_STEPS):
    optimizer_img.zero_grad()

    start_ckpt, end_ckpt     = recorder.sample_checkpoint_pair(step_gap=TRAJ_GAP)
    expert_start_weights     = start_ckpt['weights']
    expert_end_weights       = end_ckpt['weights']

    student_model = make_model().to(device)

    # ── Build differentiable params dict ──────────────────────────────────────
    params = {}
    for name, param in student_model.named_parameters():
        base = expert_start_weights.get(name, param.detach()).to(device)
        params[name] = base.clone().detach().requires_grad_(True)
    start_params = {k: v for k, v in params.items()}

    # ── Derive windows ────────────────────────────────────────────────────────
    M       = synthetic_data.shape[0]
    windows = [synthetic_data[i: i + window_size] for i in range(M - window_size + 1)]
    all_wins = torch.stack(windows)
    n_wins   = all_wins.shape[0]

    # ── Inner loop ────────────────────────────────────────────────────────────
    for _ in range(STUDENT_STEPS):
        param_list = list(params.values())
        idx        = torch.randperm(n_wins, device=device)[:min(BATCH_SIZE, n_wins)]
        batch      = all_wins[idx]
        inputs, targets = batch[:, :SEQ_LEN, :], batch[:, SEQ_LEN:, :]

        preds = torch.func.functional_call(student_model, params, (inputs,))
        loss  = criterion(preds, targets)
        grads = torch.autograd.grad(loss, param_list, create_graph=True, allow_unused=True)
        params = {
            name: p - STUDENT_LR * (g if g is not None else torch.zeros_like(p))
            for (name, p), g in zip(params.items(), grads)
        }

    # ── Loss ──────────────────────────────────────────────────────────────────
    final_list, start_list, target_list = [], [], []
    for name, sp in start_params.items():
        if name not in expert_end_weights:
            continue
        tp = expert_end_weights[name].to(device).detach()
        final_list.append(params[name])
        start_list.append(sp)
        target_list.append(tp)

    param_loss = matcher.calculate_loss(final_list,  target_list)
    param_dist = matcher.calculate_loss(start_list,  target_list)
    grand_loss = param_loss / (param_dist + 1e-12)

    grand_loss.backward()

    grad_norm  = synthetic_data.grad.norm().item() if synthetic_data.grad is not None else 0.0
    optimizer_img.step()

    seq_delta  = (synthetic_data.detach().cpu() - torch.tensor(syn_init_np)).abs().mean().item()

    log['step'].append(step + 1)
    log['param_loss'].append(param_loss.item())
    log['param_dist'].append(param_dist.item())
    log['grand_loss'].append(grand_loss.item())
    log['grad_norm'].append(grad_norm)
    log['seq_delta'].append(seq_delta)

    print(f"{step+1:>5}  {param_loss.item():>12.4f}  {param_dist.item():>12.4f}  "
          f"{grand_loss.item():>12.6f}  {grad_norm:>12.2e}  {seq_delta:>12.2e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Gradient Flow Diagnostic — {DATASET} × {MODEL}", fontsize=13)

steps = log['step']

ax = axes[0, 0]
ax.plot(steps, log['param_loss'], label='param_loss', color='tab:blue')
ax.plot(steps, log['param_dist'], label='param_dist', color='tab:orange', linestyle='--')
ax.set_title('param_loss vs param_dist')
ax.set_xlabel('outer step')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(steps, log['grand_loss'], color='tab:red')
ax.set_title('grand_loss  (= param_loss / param_dist)')
ax.set_xlabel('outer step')
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.semilogy(steps, log['grad_norm'], color='tab:purple')
ax.set_title('grad_norm  (meta-gradient into synthetic_data)\nsmall = optimization is broken')
ax.set_xlabel('outer step')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.semilogy(steps, [max(d, 1e-15) for d in log['seq_delta']], color='tab:green')
ax.set_title('seq_delta  (mean |syn - syn_init|)\nshould grow if optimization works')
ax.set_xlabel('outer step')
ax.grid(True, alpha=0.3)

# Synthetic channel 0 at step 0 vs final vs real
ax = axes[1, 1]
syn_final_np = synthetic_data.detach().cpu().numpy()
real_np      = raw_train.numpy()
t_syn  = np.arange(N_SYNTHETIC)
t_real = np.arange(min(N_SYNTHETIC, len(real_np)))
ax.plot(t_real, real_np[:len(t_real), 0], label='real ch0',  alpha=0.6, color='black')
ax.plot(t_syn,  syn_init_np[:, 0],        label='syn ch0 @ init',  alpha=0.7, linestyle='--', color='tab:blue')
ax.plot(t_syn,  syn_final_np[:, 0],       label=f'syn ch0 @ step {OUTER_STEPS}', alpha=0.7, color='tab:red')
ax.set_title('Channel 0: real vs synthetic (init and final)')
ax.set_xlabel('timestep')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Histogram of (syn_final - syn_init) to see if changes are real or rounding noise
ax = axes[1, 2]
delta_vals = (syn_final_np - syn_init_np).flatten()
ax.hist(delta_vals, bins=60, color='tab:cyan', edgecolor='none')
ax.set_title(f'Distribution of (syn_final - syn_init)\nmean={delta_vals.mean():.2e}  std={delta_vals.std():.2e}')
ax.set_xlabel('Δ value')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).parent / 'results' / f'debug_grad_flow_{DATASET}_{MODEL}.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=120)
print(f"\nPlot saved → {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY  {DATASET} × {MODEL}")
print(f"{'='*60}")
print(f"  grad_norm  min={min(log['grad_norm']):.2e}  max={max(log['grad_norm']):.2e}")
print(f"  seq_delta  after {OUTER_STEPS} steps = {log['seq_delta'][-1]:.2e}")
print(f"  grand_loss step1={log['grand_loss'][0]:.4f}  step{OUTER_STEPS}={log['grand_loss'][-1]:.4f}")
if log['seq_delta'][-1] < 1e-4:
    print("\n  *** DIAGNOSIS: seq_delta near zero → optimization not moving synthetic_data ***")
    print("  *** Fix: increase student_lr (try 0.1) or increase student_steps (try 50) ***")
elif log['grand_loss'][-1] >= log['grand_loss'][0] * 0.99:
    print("\n  *** DIAGNOSIS: grand_loss not decreasing → student not matching expert ***")
else:
    print("\n  Optimization appears to be working. Check transfer_mse separately.")
