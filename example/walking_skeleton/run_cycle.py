"""
MTT Pipeline — User Entry Point
================================
This is the script you run to perform time-series dataset distillation
using the Matching Training Trajectories (MTT) algorithm.

Workflow
--------
  1.  Edit CONFIG below to choose your dataset, model, and hyper-parameters.
  2.  Run from the project root:
          python -m example.walking_skeleton.run_cycle
      or simply:
          python example/walking_skeleton/run_cycle.py

What this script does
---------------------
  Step 1 — Load raw CSV data, compute train/val/test splits, normalise.
  Step 2 — Train an "expert" model and record its weight trajectory.
  Step 3 — Run the MTT distillation loop to learn a compact synthetic sequence.
  Step 4 — Train two evaluation models (one on real data, one on synthetic)
            and compare their test-set MSE to measure distillation quality.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader as TorchDataLoader
from ts_distill.distillation_core.initializer.geommetry_sequence_initializer import GeometrySequenceInitializer
from ts_distill.distillation_core.initializer.uncertainty_sequence_initializer import UncertaintySampleInitializer

# ── Make the project root importable (needed when running as a plain script) ──
sys.path.append(str(Path(__file__).parent.parent.parent))

# ── Framework — data utilities ────────────────────────────────────────────────
from ts_distill.data_pipeline.splitter import get_data_splits, make_windows
from ts_distill.data_pipeline.data_loader.mini_batch_loader import MiniBatchLoader

# ── Framework — models ────────────────────────────────────────────────────────
from ts_distill.models.factory import create_model

# ── Framework — distillation ──────────────────────────────────────────────────
from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.distillation_core.initializer.random_sample_initializer import RandomSampleInitializer

# ── Framework — training infrastructure ──────────────────────────────────────
from ts_distill.trainer.trainer.trainer import Trainer
from ts_distill.trainer.callback.simple_callback import SimpleCallback

# ── Framework — trajectory tools ──────────────────────────────────────────────
from ts_distill.trajectory.recorder.simple_recorder import SimpleRecorder
from ts_distill.trajectory.matcher.mse_matcher import MSEMatcher

# ── Framework — evaluation ────────────────────────────────────────────────────
from ts_distill.evaluation.evaluation import Evaluator

from ts_distill.evaluation.hybrid_evaluation.hybrid import BaseHybridEvaluator
from ts_distill.evaluation.hybrid_evaluation.hybrid import RandomAnchorSelector


# =============================================================================
# CONFIG — edit this section to run different experiments
# =============================================================================

CONFIG = {
    # ── Pipeline selectors ───────────────────────────────────────────────────
    'active_model':   'MLP',
    'active_dataset': 'ETTm1',

    # ── Forecasting dimensions ───────────────────────────────────────────────
    'in_features': 7,    # Number of multivariate channels in the dataset
    'seq_len':    96,    # Look-back window fed to the model as input
    'pred_len':   96,    # Forecast horizon the model must predict

    # ── Dataset configurations ───────────────────────────────────────────────
    # benchmark_borders: row indices that reproduce published paper splits.
    # ratios:            proportional splits for custom datasets.
    'datasets': {
        'ETTh1': {
            'csv_path':   'example/ETTh1.csv',
            'split_mode': 'benchmark_borders',
            # Hourly data — 12 months train, 4 months val, 4 months test
            'border1s': [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96],
            'border2s': [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        },
        'ETTh2': {
            'csv_path':   'example/ETTh2.csv',
            'split_mode': 'benchmark_borders',
            'border1s': [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96],
            'border2s': [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
        },
        'ETTm1': {
            'csv_path':   'example/ETTm1.csv',
            'split_mode': 'benchmark_borders',
            # 15-minute data — multiply by 96 instead of 24
            'border1s': [0, 12 * 30 * 96 - 96, 12 * 30 * 96 + 4 * 30 * 96 - 96],
            'border2s': [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        },
        'ETTm2': {
            'csv_path':   'example/ETTm2.csv',
            'split_mode': 'benchmark_borders',
            'border1s': [0, 12 * 30 * 96 - 96, 12 * 30 * 96 + 4 * 30 * 96 - 96],
            'border2s': [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
        },
        'General': {
            'csv_path':    'example/electricity.csv',
            'split_mode':  'ratios',
            'train_ratio': 0.6,
            'val_ratio':   0.2,
        },
    },

    # ── Model-specific kwargs (passed to create_model) ────────────────────────
    # Add any constructor keyword arguments your chosen model needs here.
    'models': {
    'DLinear': {'individual': False},
    'LSTM':    {'hidden_dim': 16, 'num_layer': 1},
    'MLP':     {},
    'CNN':     {},

},

    # ── Expert training ───────────────────────────────────────────────────────
    'expert_epochs':   80,    # Total SGD epochs for the expert trajectory
    'expert_lr':       0.01,
    'expert_momentum': 0.9,

    # ── Distillation loop ─────────────────────────────────────────────────────
    'n_distill_steps':        300,  # Outer-loop iterations (matches HDT paper)
    'n_synthetic':            384,  # Length of the synthetic continuous sequence
    'synthetic_lr':           0.1,  # Learning rate for the synthetic sequence tensor
    'student_lr':            0.01,  # Inner-loop student learning rate
    'student_steps':           20,  # Inner-loop gradient steps (matches HDT paper)
    'snapshot_student_steps':  50,  # Steps used to train the temp validation student
    'batch_size':              64,
    'trajectory_gap':           5,  # Step gap when sampling expert checkpoint pairs

    # ── Evaluation ────────────────────────────────────────────────────────────
    # Both real and synthetic evaluation models use the same protocol:
    # train up to eval_max_epochs with early stopping on the real val set.
    # This ensures a fair comparison (both converge under the same criterion).
    'eval_max_epochs':    300,   # Hard cap; early stopping fires well before this
    'eval_lr':          0.001,
    'eval_batch_size':     64,
    'early_stop_patience': 10,   # Val epochs without improvement → restore best weights
}

# Derived constant — total timesteps per sliding window
CONFIG['window_size'] = CONFIG['seq_len'] + CONFIG['pred_len']


# =============================================================================
# Model factory helper
# =============================================================================

def make_model():
    """
    Create a fresh, randomly-initialised instance of the active model.

    Called multiple times throughout the pipeline (expert, distiller inner
    loop, evaluation), so each call returns an independent copy with its own
    weights.  Reads from CONFIG so you only need to change 'active_model' once.
    """
    return create_model(
        model_type   = CONFIG['active_model'],
        seq_len      = CONFIG['seq_len'],
        pred_len     = CONFIG['pred_len'],
        in_features  = CONFIG['in_features'],
        model_kwargs = CONFIG['models'][CONFIG['active_model']],
    )



# =============================================================================
# Main pipeline
# =============================================================================

def main():
    print(f"\nMTT Pipeline | Model: {CONFIG['active_model']} | Dataset: {CONFIG['active_dataset']}")
    print("=" * 70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Load and normalise data
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading and normalising data...")

    dataset_cfg = CONFIG['datasets'][CONFIG['active_dataset']]
    df_raw      = pd.read_csv(dataset_cfg['csv_path'])
    values      = df_raw.iloc[:, 1:].values.astype(np.float32)  # drop the date column

    # Compute row-level split boundaries (train / val / test)
    train_start, train_end, val_start, val_end, test_start, test_end = get_data_splits(
        values      = values,
        window_size = CONFIG['window_size'],
        seq_len     = CONFIG['seq_len'],
        dataset_cfg = dataset_cfg,
    )

    # Fit StandardScaler ONLY on training rows — prevents data leakage into
    # validation and test sets.
    scaler = StandardScaler()
    scaler.fit(values[train_start:train_end])
    data = scaler.transform(values)

    # Slice each split and convert to overlapping (window, timestep, feature) tensors.
    # Shape: (N_windows, window_size, in_features)
    train_data = torch.tensor(make_windows(data[train_start:train_end], CONFIG['window_size']), dtype=torch.float32)
    val_data   = torch.tensor(make_windows(data[val_start:val_end],     CONFIG['window_size']), dtype=torch.float32)
    test_data  = torch.tensor(make_windows(data[test_start:test_end],   CONFIG['window_size']), dtype=torch.float32)

    print(f"   Train: {train_data.shape}  Val: {val_data.shape}  Test: {test_data.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Train the expert model and record its weight trajectory
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Training expert model and recording trajectory...")

    recorder       = SimpleRecorder(record_every=1)   # save weights every epoch
    expert_model   = make_model()
    expert_trainer = Trainer(
        model     = expert_model,
        optimizer = torch.optim.SGD(
            expert_model.parameters(),
            lr       = CONFIG['expert_lr'],
            momentum = CONFIG['expert_momentum'],
        ),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = CONFIG['seq_len'],
    )

    # MiniBatchLoader is a lightweight in-memory loader — no PyTorch Dataset needed.
    expert_loader = MiniBatchLoader(train_data, batch_size=CONFIG['batch_size'])

    # Pass recorder and printer as callbacks — they hook into fit() automatically.
    expert_trainer.fit(
        dataloader = expert_loader,
        epochs     = CONFIG['expert_epochs'],
        callbacks  = [recorder, SimpleCallback()],
    )

    print(f"   Checkpoints recorded: {len(recorder.get_trajectory())}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Distil a compact synthetic sequence using MTT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/4] Distilling synthetic sequence...")

    # Raw (un-windowed) training data is needed for initialisation.
    # The distiller works on a single continuous sequence, not a bag of windows.
    raw_train_data = torch.tensor(data[train_start:train_end], dtype=torch.float32)

    n_synthetic = CONFIG['n_synthetic']

    # --- Initialise the synthetic sequence -----------------------------------
    # Create the initializer once so the same instance is both used to produce
    # the starting tensor AND stored inside the distiller for future reference.
    # initializer    = RandomSampleInitializer()
    # synthetic_init = initializer.initialize_sequence(raw_train_data, n_synthetic)
    initializer    = GeometrySequenceInitializer()
    synthetic_init = initializer.initialize_sequence(raw_train_data, n_synthetic)
    # initializer = UncertaintySampleInitializer()
    # synthetic_init = initializer.initialize_sequence(raw_train_data, n_synthetic)
    
    # --- Build the distiller -------------------------------------------------
    distiller = MTTDistiller(
        initializer           = initializer,
        matcher               = MSEMatcher(),
        model_factory         = make_model,   # called fresh on each inner loop
        expert_recorder       = recorder,
        expert_epochs         = CONFIG['trajectory_gap'],
        syn_batch_size        = CONFIG['batch_size'],
        synthetic_lr          = CONFIG['synthetic_lr'],
        student_lr            = CONFIG['student_lr'],
        student_steps         = CONFIG['student_steps'],
        snapshot_student_steps= CONFIG['snapshot_student_steps'],
        seq_len               = CONFIG['seq_len'],
        pred_len              = CONFIG['pred_len'],
    )

    # val_data is passed so the distiller tracks the best snapshot
    # (lowest validation MSE) and returns it instead of the final-step result.
    synthetic_sequence = distiller.distill(
        synthetic_init  = synthetic_init,
        n_steps         = CONFIG['n_distill_steps'],
        val_data        = val_data,
    )

    print(f"   Compressed: {len(raw_train_data)} timesteps → {n_synthetic} timesteps")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Evaluate: real-data model vs. synthetic-data model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating...")

    # Reset random seed to ensure the real-data model baseline is identical 
    # regardless of how many random numbers were consumed in Step 3.
    torch.manual_seed(42)

    # -- 4a. Train on full real data (with early stopping on the val set) -----
    real_model   = make_model()
    real_trainer = Trainer(
        model     = real_model,
        optimizer = torch.optim.Adam(real_model.parameters(), lr=CONFIG['eval_lr']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = CONFIG['seq_len'],
    )
    real_loader = TorchDataLoader(train_data, batch_size=CONFIG['eval_batch_size'], shuffle=True)

    real_trainer.fit(
        dataloader = real_loader,
        epochs     = CONFIG['eval_max_epochs'],
        val_loader = eval_val_loader,
        patience   = CONFIG['early_stop_patience'],
    )

    # -- 4b. Train on distilled synthetic data (early stopping on real val set)
    # Converting the 1-D synthetic sequence into sliding windows first.
    syn_windows = torch.tensor(
        make_windows(synthetic_sequence.cpu().numpy(), CONFIG['window_size']),
        dtype=torch.float32,
    )

    torch.manual_seed(1)
    syn_model   = make_model()
    syn_trainer = Trainer(
        model     = syn_model,
        optimizer = torch.optim.Adam(syn_model.parameters(), lr=CONFIG['eval_lr']),
        criterion = torch.nn.MSELoss(),
        device    = device,
        seq_len   = CONFIG['seq_len'],
    )
    syn_loader = TorchDataLoader(syn_windows, batch_size=CONFIG['eval_batch_size'], shuffle=True)
    syn_trainer.fit(
        dataloader = syn_loader,
        epochs     = CONFIG['eval_max_epochs'],
        val_loader = eval_val_loader,
        patience   = CONFIG['early_stop_patience'],
    )

    # -- 4c. Score both models on the real held-out test set ------------------
    evaluator         = Evaluator(seq_len=CONFIG['seq_len'], batch_size=CONFIG['eval_batch_size'])
    real_metrics      = evaluator.test_on_real(real_model, test_data.to(device))
    synthetic_metrics = evaluator.test_on_real(syn_model,  test_data.to(device))

    # print("\n[4/4b] Hybrid Evaluation...")

    # hybrid_evaluator = BaseHybridEvaluator(
    #     anchor_selector=RandomAnchorSelector(),
    #     seq_len=CONFIG['seq_len'],
    #     batch_size=CONFIG['batch_size'],
    #     device=device
    # )

    # hybrid_results = hybrid_evaluator.evaluate_mixing(
    #     synthetic_data=synthetic_sequence,
    #     real_train_data=raw_train_data,
    #     real_test_loader=TorchDataLoader(
    #         test_data,
    #         batch_size=CONFIG['eval_batch_fulldata']
    #     ),
    #     model_fn=make_model,
    #     window_size=CONFIG['window_size'],
    #     mixing_ratios=(0.1, 0.2, 0.5)
    # )

    # print("\n" + "=" * 60)
    # print("Hybrid Evaluation (MSE vs Real Data Ratio)")
    # print("=" * 60)

    # for ratio, metrics in hybrid_results.items():
    #     percent = ratio.replace("hybrid_", "")
    #     print(f"Real Data {percent:>3}%  →  MSE: {metrics['MSE']:.6f}")

    # print("=" * 60)
    # hybrid_evaluator.plot_hybrid_results(hybrid_results)

    print("\nResults:")
    print("Real MSE:", real_metrics['MSE'])
    print("Synthetic MSE:", synthetic_metrics['MSE'])


if __name__ == "__main__":
    main()