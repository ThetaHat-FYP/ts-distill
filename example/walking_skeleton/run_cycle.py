"""
MTT Walking Skeleton - End-to-End Pipeline
Time series data distillation using Matching Training Trajectories.
"""

import torch
import pandas as pd
import sys
from pathlib import Path

from ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from ts_distill.models.dlinear import DLinear
from ts_distill.models.lstm import LSTM
from ts_distill.models.mlp import MLP
from ts_distill.models.cnn import CNN


sys.path.append(str(Path(__file__).parent.parent.parent))

from example.walking_skeleton.etth1_loader import ETTh1DataLoader
from example.walking_skeleton.mock_components import ( SimpleRecorder, SimpleEvaluator, 
    SimpleTrainer, SimpleCallback, MSEMatcher, RealSampleInitializer
)
from src.ts_distill.distillation_core.distillation_algorithm.condtsf import CondTSFDistiller
from src.ts_distill.distillation_core.distillation_algorithm.frepo import FRePODistiller
# from src.ts_distill.distillation_core.distillation_algorithm.mtt import MTTDistiller
from src.ts_distill.data_pipeline.data_windowing.fixed_windowing import FixedWindowing
from src.ts_distill.data_pipeline.data_preprocessor.normalization import StandardNormalization


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════



CONFIG = {
    # Data
    'csv_path': 'example/ETTh1.csv',
    'use_all_channels': True,    # All 7 ETTh1 channels

    # Windowing
    'window_size': 192,          # seq_len + pred_len = 96 + 96
    'stride': 1,

    # Expert Training
    'expert_epochs': 80,
    'expert_lr': 0.01,           # must be close to student_lr so trajectories are matchable
    'expert_momentum': 0.9,

    # Distillation  (CondTSF paper settings for MTT baseline)
    'n_distill_steps': 300,      # CondTSF: 300 outer iterations
    'n_synthetic': 384,          # M=384 per HDT/CondTSF paper Table 1
    'compression_ratio': 0.0333, # kept for FRePO distiller
    'synthetic_lr': 0.1,
    'student_lr': 0.01,
    'student_steps': 20,         # CondTSF: 20 student unroll steps
    'batch_size': 64,            # CondTSF: batch size 64
    'trajectory_gap': 5,
    'cond_gap': 3,
    'beta': 0.01,
    'frepo_online_lr': 0.001,
    'frepo_online_updates': 10,
    'frepo_ridge_lambda': 0.001,

    # Evaluation - Full Data baseline  (standard DLinear: Adam lr=0.0001, 10 epochs)
    'eval_epochs_fulldata': 10,
    'eval_lr_fulldata': 0.0001,
    'eval_batch_fulldata': 32,

    # Evaluation - Synthetic data  (small dataset needs more epochs to saturate)
    'eval_epochs_synthetic': 300,
    'eval_lr_synthetic': 0.001,
    'eval_batch_synthetic': 64,

    # Model
    'hidden_size': 16,
    'num_layers': 1,

    # Model Setup  (paper §B.4: l=96, t=96, 7-channel multivariate)
    'in_features': 7,            # All 7 ETTh1 channels
    'seq_len': 96,               # Tin
    'pred_len': 96               # Tout
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


# Update your factory function
def create_model():
    return DLinear(
        seq_len=CONFIG['seq_len'],
        pred_len=CONFIG['pred_len'],
        channels=CONFIG['in_features'],
        individual=True
    )

    # return MLP(
    #     seq_len=CONFIG['seq_len'],
    #     pred_len=CONFIG['pred_len'],
    # )

    # return LSTM(
    #     input_dim=CONFIG['in_features'],
    #     hidden_dim=CONFIG['hidden_size'],
    #     num_layer=CONFIG['num_layers'],
    #     seq_len=CONFIG['seq_len'],
    #     pred_len=CONFIG['pred_len'],
    # )

    # return CNN(
    #     channel=CONFIG['in_features'],
    #     seq_len=CONFIG['seq_len'],
    #     pred_len=CONFIG['pred_len'],
    # )


class MiniBatchLoader:
    """Yields shuffled mini-batches from a tensor dataset each epoch."""
    def __init__(self, data, batch_size=64, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = self.data.shape[0]
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            yield self.data[idx[i : i + self.batch_size]]


def main():
    """Execute MTT distillation pipeline."""
    
    print("\nMTT Data Distillation Pipeline  [ETTh1, 7ch, seq=96, pred=96, 60/20/20 split, M=384]")
    print("=" * 70)
    
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
    # Load data  (ETT 60:20:20 chronological split per paper §B.4)
    print("\n1. Loading data...")
    total_rows = len(pd.read_csv(CONFIG['csv_path']))
    n_train = int(total_rows * 0.60)
    n_val   = int(total_rows * 0.20)
    n_test_start = n_train + n_val
    n_test = total_rows - n_test_start

    train_loader = ETTh1DataLoader(
        CONFIG['csv_path'],
        n_samples=n_train,
        seq_len=1,
        batch_size=n_train,
        start_idx=0,
        single_sequence=True,
        use_all_channels=CONFIG['use_all_channels']
    )
    train_loader.load_data()

    test_loader = ETTh1DataLoader(
        CONFIG['csv_path'],
        n_samples=n_test,
        seq_len=1,
        batch_size=n_test,
        start_idx=n_test_start,
        single_sequence=True,
        use_all_channels=CONFIG['use_all_channels']
    )
    test_loader.load_data()
    
    train_data = train_loader.data
    test_data = test_loader.data
    print(f"   Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Normalize  (per-channel: each of the 7 channels normalized independently)
    print("2. Normalizing data...")
    normalizer = StandardNormalization(per_channel=True)
    train_data = normalizer.fit_transform(train_data)
    test_data = normalizer.transform(test_data)
    
    # Apply windowing
    print("3. Applying windowing...")
    windowing = FixedWindowing(
        window_size=CONFIG['window_size'],
        stride=CONFIG['stride']
    )
    train_data, _ = windowing.create_windows(train_data)
    test_data, _ = windowing.create_windows(test_data)
    print(f"   Windows: {train_data.shape}")
    
    # Train expert
    print("4. Training expert model...")
    expert_model = create_model()
    recorder = SimpleRecorder(record_every=1)
    optimizer = torch.optim.SGD(
        expert_model.parameters(), 
        lr=CONFIG['expert_lr'], 
        momentum=CONFIG['expert_momentum']
    )
    criterion = torch.nn.MSELoss()
    
    trainer = SimpleTrainer(
        model=expert_model,
        optimizer=optimizer,
        criterion=criterion,
        device='cpu'
    )
    
    callbacks = [recorder, SimpleCallback()]
    dataloader = MiniBatchLoader(train_data, batch_size=CONFIG['batch_size'])
    trainer.fit(dataloader, epochs=CONFIG['expert_epochs'], callbacks=callbacks)
    print(f"   Checkpoints: {len(recorder.get_trajectory())}")
    
    # Distill synthetic data
    print("5. Distilling synthetic data...")
    distiller = MTTDistiller(
        initializer=RealSampleInitializer(),
        matcher=MSEMatcher(),
        model_factory=create_model,
        expert_recorder=recorder,
        expert_epochs=CONFIG['trajectory_gap'],
        syn_batch_size=CONFIG['batch_size'],
        synthetic_lr=CONFIG['synthetic_lr'],
        student_lr=CONFIG['student_lr'],
        student_steps=CONFIG['student_steps']
    )

    # distiller = CondTSFDistiller(
    #     initializer=RealSampleInitializer(),
    #     matcher=MSEMatcher(),
    #     model_factory=create_model,
    #     teacher_state_dict={k: v.detach().cpu().clone() for k, v in expert_model.state_dict().items()},
    #     expert_epochs=CONFIG['trajectory_gap'],
    #     syn_batch_size=CONFIG['batch_size'],
    #     synthetic_lr=CONFIG['synthetic_lr'],
    #     student_lr=CONFIG['student_lr'],
    #     student_steps=CONFIG['student_steps'],
    #     cond_gap=CONFIG['cond_gap'],
    #     beta=CONFIG['beta'],
    #     device=device
    # )

    # distiller = FRePODistiller(
    #     initializer=RealSampleInitializer(),
    #     matcher=MSEMatcher(),
    #     model_factory=create_model,
    #     synthetic_lr=CONFIG['synthetic_lr'],
    #     online_lr=CONFIG['frepo_online_lr'],
    #     online_updates=CONFIG['frepo_online_updates'],
    #     syn_batch_size=min(128, max(1, int(train_data.shape[0] * CONFIG['compression_ratio']))),
    #     real_batch_size=256,
    #     ridge_lambda=CONFIG['frepo_ridge_lambda'],
    #     device=device,
    # )
    
    n_synthetic = CONFIG['n_synthetic']
    synthetic_data = distiller.distill(
        train_data,
        n_steps=CONFIG['n_distill_steps'],
        n_synthetic=n_synthetic
    )
    print(f"   Compression: {train_data.shape[0]} → {n_synthetic} samples ({n_synthetic/train_data.shape[0]*100:.1f}%)")
    
    # Evaluate
    print("6. Evaluating performance...")

    # Full-data evaluator: standard DLinear protocol (10 epochs, Adam lr=0.0001, batch=32)
    full_evaluator = SimpleEvaluator(
        model_factory=create_model,
        n_epochs=CONFIG['eval_epochs_fulldata'],
        lr=CONFIG['eval_lr_fulldata'],
        batch_size=CONFIG['eval_batch_fulldata']
    )

    # Synthetic evaluator: extended training to saturate small dataset (500 epochs)
    syn_evaluator = SimpleEvaluator(
        model_factory=create_model,
        n_epochs=CONFIG['eval_epochs_synthetic'],
        lr=CONFIG['eval_lr_synthetic'],
        batch_size=CONFIG['eval_batch_synthetic']
    )

    real_model = full_evaluator.train_on_synthetic(train_data)
    real_metrics = full_evaluator.test_on_real(real_model, test_data.to(device))

    synthetic_model = syn_evaluator.train_on_synthetic(synthetic_data)
    synthetic_metrics = syn_evaluator.test_on_real(synthetic_model, test_data.to(device))
    
    performance_ratio = (synthetic_metrics['MSE'] / real_metrics['MSE']) * 100
    performance_retention = (real_metrics['MSE'] / synthetic_metrics['MSE']) * 100
    
    print("\nResults:")
    print(f"   Real MSE:      {real_metrics['MSE']:.6f}")
    print(f"   Synthetic MSE: {synthetic_metrics['MSE']:.6f}")
    print(f"   Error Ratio:   {(synthetic_metrics['MSE'] / real_metrics['MSE']):.2f}x original error")
    print(f"   Perf Retained: {performance_retention:.1f}%")
    print(f"   Compression:   {n_synthetic}/{train_data.shape[0]} ({n_synthetic/train_data.shape[0]*100:.2f}%)")
    print("\n" + "=" * 70)
    print("Pipeline complete.\n")


    return {
        'train_data': train_data,
        'synthetic_data': synthetic_data,
        'real_mse': real_metrics['MSE'],
        'synthetic_mse': synthetic_metrics['MSE'],
        'performance_ratio': performance_ratio
    }


if __name__ == "__main__":
    results = main()
