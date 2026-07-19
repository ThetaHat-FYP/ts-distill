"""
Post-Fix Visualization Helper
=============================
Standalone plotting utility — plots synthetic data before vs after the FFT
amplitude post-fix, against real data, in both time and frequency domains.

Kept in its own file so it can be added/removed from experiment_matrix.py
with a single import + single call.  See the fenced snippet at the bottom of
this docstring for how to wire it in.

    # ── POST-FIX VIZ (remove this block to disable) ───────────────────
    from example.experiments.plot_postfix import plot_before_after
    plot_before_after(syn_before, syn_after, raw_train_data,
                      dataset_name, model_name, results_dir)
    # ──────────────────────────────────────────────────────────────────
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — save PNG, don't open a window
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def plot_before_after(
    syn_before,
    syn_after,
    real_train,
    dataset_name: str,
    model_name:   str,
    results_dir,
    channel: int = 0,
):
    """
    Plot synthetic-before vs synthetic-after post-fix, against real data.

    Two panels:
      Left  — time domain: one channel over time (real sample, syn before, syn after)
      Right — frequency domain: magnitude spectrum (mean real, syn before, syn after)

    Args:
        syn_before   : (N, C) distilled synthetic sequence BEFORE post-fix.
        syn_after    : (N, C) synthetic sequence AFTER post-fix.
        real_train   : (T, C) real training sequence.
        dataset_name : e.g. 'ETTh1' — used in title and filename.
        model_name   : e.g. 'MLP'   — used in title and filename.
        results_dir  : directory (str or Path) to save the PNG into.
        channel      : which channel to plot (default 0).
    """
    before = _to_np(syn_before)
    after  = _to_np(syn_after)
    real   = _to_np(real_train)

    N = before.shape[0]
    c = channel

    # Mean real magnitude spectrum over all length-N segments (matches the metric).
    n_segs = len(real) // N
    if n_segs == 0:
        mean_real_amp = np.abs(np.fft.rfft(real[:N, c]))
    else:
        amps = [np.abs(np.fft.rfft(real[i*N:(i+1)*N, c])) for i in range(n_segs)]
        mean_real_amp = np.mean(amps, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Synthetic before vs after post-fix — {dataset_name} × {model_name} "
                 f"(channel {c})", fontsize=13)

    # ── Time domain ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(real[:N, c],   label='real (first window)', color='black', alpha=0.5, linewidth=1)
    ax.plot(before[:, c],  label='synthetic BEFORE',    color='tab:blue',  alpha=0.8)
    ax.plot(after[:, c],   label='synthetic AFTER',     color='tab:red',   alpha=0.8)
    ax.set_title('Time domain')
    ax.set_xlabel('timestep')
    ax.set_ylabel('value (scaled)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Frequency domain ─────────────────────────────────────────────────────
    ax = axes[1]
    freqs = np.fft.rfftfreq(N)
    ax.plot(freqs, mean_real_amp,                    label='real (mean spectrum)', color='black', alpha=0.6)
    ax.plot(freqs, np.abs(np.fft.rfft(before[:, c])), label='synthetic BEFORE',   color='tab:blue', alpha=0.8)
    ax.plot(freqs, np.abs(np.fft.rfft(after[:, c])),  label='synthetic AFTER',    color='tab:red',  alpha=0.8)
    ax.set_title('Frequency domain (FFT magnitude)')
    ax.set_xlabel('frequency (cycles/step)')
    ax.set_ylabel('magnitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f'postfix_viz_{dataset_name}_{model_name}.png'
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"   [viz] Post-fix comparison saved → {out_path}")
    return out_path
