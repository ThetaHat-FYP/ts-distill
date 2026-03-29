"""
MTT Walking Skeleton - End-to-End Pipeline
Time series data distillation using Matching Training Trajectories.
"""

import os
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from example.walking_skeleton.mock_components import (
    SimpleLSTM, SimpleRecorder, SimpleEvaluator, 
    SimpleTrainer, SimpleCallback, MSEMatcher, RealSampleInitializer, StatelessDLinear
)
from example.walking_skeleton.mtt_distiller import MTTDistiller
from src.ts_distill.data_pipeline.data_loader import CSVDataLoader
from src.ts_distill.data_pipeline.data_windowing.fixed_windowing import FixedWindowing
from src.ts_distill.data_pipeline.data_preprocessor.normalization import StandardNormalization


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════



CONFIG = {
    # Data
    # Override without editing this file:
    #   PowerShell: $env:TS_DISTILL_CSV_PATH = "C:\\Users\\piyum\\Downloads\\my_file.csv"
    #   PowerShell: $env:TS_DISTILL_TARGET_COLUMN = "OT"
    #   PowerShell: $env:TS_DISTILL_CSV_ENCODING = "cp1252"  # optional
    #   Then run:   python example/walking_skeleton/run_cycle.py
    'csv_path': os.getenv('TS_DISTILL_CSV_PATH', 'example/ETTh1.csv'),
    'csv_encoding': os.getenv('TS_DISTILL_CSV_ENCODING', ''),
    'n_train_samples': 15000,
    'n_test_samples': 3000,
    'test_start_idx': 5000,
    'target_column': os.getenv('TS_DISTILL_TARGET_COLUMN', 'OT'),
    
    # Windowing
    'window_size': 48,
    'stride': 1,
    
    # Expert Training
    'expert_epochs': 80,
    'expert_lr': 0.0005, 
    'expert_momentum': 0.9,
    
    # Distillation
    'n_distill_steps': 50,
    'compression_ratio': 0.1,
    'synthetic_lr': 0.1,
    'student_lr': 0.0003,
    'student_steps': 5,
    'trajectory_gap': 5,
    'n_synthetic': 10, 
    
    # Evaluation
    'eval_epochs': 50,
    'eval_lr': 0.001,
    
    # Model
    'hidden_size': 16,
    'num_layers': 1,

    # Model Setup
    'in_features': 1,      # Assuming univariate for ETTh1 target column
    'seq_len': 24,         # Tin
    'pred_len': 24         # Tout
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


# Update your factory function
def create_model():
    return StatelessDLinear(
        seq_len=CONFIG['seq_len'], 
        pred_len=CONFIG['pred_len'],
        channels=CONFIG['in_features']
    )


class SingleBatchLoader:
    """Wrapper to make single batch compatible with trainer interface."""
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        yield self.data


def _make_single_sequence(values: np.ndarray, start_idx: int, n_samples: int) -> torch.Tensor:
    """Create a single univariate sequence tensor of shape (1, T, 1)."""
    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0, got {start_idx}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if start_idx >= len(values):
        raise ValueError(f"start_idx {start_idx} is >= dataset length {len(values)}")

    end_idx = min(start_idx + n_samples, len(values))
    sequence = np.asarray(values[start_idx:end_idx], dtype=np.float32).copy()
    return torch.from_numpy(sequence).unsqueeze(0).unsqueeze(-1)


def main():
    """Execute MTT distillation pipeline."""
    
    print("\nMTT Data Distillation Pipeline")
    print("=" * 70)
    
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
    # Load data
    print("\n1. Loading data...")
    read_csv_kwargs = {}
    if CONFIG.get('csv_encoding'):
        read_csv_kwargs['encoding'] = CONFIG['csv_encoding']

    csv_loader = CSVDataLoader(read_csv_kwargs=read_csv_kwargs)
    df = csv_loader.load_data(CONFIG['csv_path'])
    csv_loader.view_data(n_rows=5)

    target_column = CONFIG['target_column']
    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    series = df[target_column]
    numeric = pd.to_numeric(series, errors="coerce")
    dropped = int(numeric.isna().sum())
    if dropped:
        print(
            f"   Note: dropped {dropped} non-numeric/blank rows from '{target_column}' before training."
        )

    values = numeric.dropna().to_numpy(dtype=np.float32, copy=True)
    train_data = _make_single_sequence(values, start_idx=0, n_samples=CONFIG['n_train_samples'])
    test_data = _make_single_sequence(values, start_idx=CONFIG['test_start_idx'], n_samples=CONFIG['n_test_samples'])
    print(f"   Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Normalize
    print("2. Normalizing data...")
    normalizer = StandardNormalization()
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
    dataloader = SingleBatchLoader(train_data)
    trainer.fit(dataloader, epochs=CONFIG['expert_epochs'], callbacks=callbacks)
    print(f"   Checkpoints: {len(recorder.get_trajectory())}")
    
    # Distill synthetic data
    print("5. Distilling synthetic data...")
    distiller = MTTDistiller(
        initializer=RealSampleInitializer(),
        matcher=MSEMatcher(),
        model_factory=create_model,
        expert_recorder=recorder,
        synthetic_lr=CONFIG['synthetic_lr'],
        student_lr=CONFIG['student_lr'],
        student_steps=CONFIG['student_steps']
    )
    
    n_synthetic = max(1, int(train_data.shape[0] * CONFIG['compression_ratio']))
    synthetic_data = distiller.distill(
        train_data,
        n_steps=CONFIG['n_distill_steps'],
        n_synthetic=n_synthetic
    )
    print(f"   Compression: {train_data.shape[0]} → {n_synthetic} samples")
    
    # Evaluate
    print("6. Evaluating performance...")
    evaluator = SimpleEvaluator(
        model_factory=create_model,
        n_epochs=CONFIG['eval_epochs'],
        lr=CONFIG['eval_lr']
    )
    
    real_model = evaluator.train_on_synthetic(train_data)
    # CHANGE: test_data.to(torch.device) -> test_data.to(device)
    real_metrics = evaluator.test_on_real(real_model, test_data.to(device))
    
    synthetic_model = evaluator.train_on_synthetic(synthetic_data)
    # CHANGE: test_data.to(torch.device) -> test_data.to(device)
    synthetic_metrics = evaluator.test_on_real(synthetic_model, test_data.to(device))
    
    performance_ratio = (synthetic_metrics['MSE'] / real_metrics['MSE']) * 100
    performance_retention = (real_metrics['MSE'] / synthetic_metrics['MSE']) * 100
    
    print("\nResults:")
    print(f"   Real MSE:      {real_metrics['MSE']:.6f}")
    print(f"   Synthetic MSE: {synthetic_metrics['MSE']:.6f}")
    print(f"   Error Ratio:   {(synthetic_metrics['MSE'] / real_metrics['MSE']):.2f}x original error")
    print(f"   Perf Retained: {performance_retention:.1f}%")
    print(f"   Compression:   {CONFIG['compression_ratio']*100:.1f}%")
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
