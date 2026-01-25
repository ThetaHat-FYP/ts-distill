"""
MTT Walking Skeleton - End-to-End Pipeline
Time series data distillation using Matching Training Trajectories.
"""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from example.walking_skeleton.etth1_loader import ETTh1DataLoader
from example.walking_skeleton.mock_components import (
    SimpleLSTM, SimpleRecorder, SimpleEvaluator, 
    SimpleTrainer, SimpleCallback, MSEMatcher, RealSampleInitializer
)
from example.walking_skeleton.mtt_distiller import MTTDistiller
from src.ts_distill.data_pipeline.data_windowing.fixed_windowing import FixedWindowing
from src.ts_distill.data_pipeline.data_preprocessor.normalization import StandardNormalization


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Data
    'csv_path': 'example/ETTh1.csv',
    'n_train_samples': 1000,
    'n_test_samples': 200,
    'test_start_idx': 5000,
    'target_column': 'OT',
    
    # Windowing
    'window_size': 96,
    'stride': 1,
    
    # Expert Training
    'expert_epochs': 50,
    'expert_lr': 0.001,
    
    # Distillation
    'n_distill_steps': 40,
    'compression_ratio': 0.1,
    'synthetic_lr': 0.1,
    'student_lr': 0.01,
    'student_steps': 10,
    
    # Evaluation
    'eval_epochs': 50,
    'eval_lr': 0.001,
    
    # Model
    'hidden_size': 16,
    'num_layers': 1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def create_model():
    """Create LSTM model instance."""
    return SimpleLSTM(
        input_size=1, 
        hidden_size=CONFIG['hidden_size'], 
        num_layers=CONFIG['num_layers']
    )


class SingleBatchLoader:
    """Wrapper to make single batch compatible with trainer interface."""
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        yield self.data


def main():
    """Execute MTT distillation pipeline."""
    
    print("\nMTT Data Distillation Pipeline")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Load data
    print("\n1. Loading data...")
    train_loader = ETTh1DataLoader(
        CONFIG['csv_path'], 
        n_samples=CONFIG['n_train_samples'],
        seq_len=1, 
        batch_size=CONFIG['n_train_samples'],
        target_column=CONFIG['target_column'],
        start_idx=0,
        single_sequence=True
    )
    train_loader.load_data()
    
    test_loader = ETTh1DataLoader(
        CONFIG['csv_path'],
        n_samples=CONFIG['n_test_samples'],
        seq_len=1,
        batch_size=CONFIG['n_test_samples'],
        target_column=CONFIG['target_column'],
        start_idx=CONFIG['test_start_idx'],
        single_sequence=True
    )
    test_loader.load_data()
    
    train_data = train_loader.data
    test_data = test_loader.data
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
    recorder = SimpleRecorder()
    optimizer = torch.optim.Adam(expert_model.parameters(), lr=CONFIG['expert_lr'])
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
    real_metrics = evaluator.test_on_real(real_model, test_loader)
    
    synthetic_model = evaluator.train_on_synthetic(synthetic_data)
    synthetic_metrics = evaluator.test_on_real(synthetic_model, test_loader)
    
    performance_ratio = (synthetic_metrics['MSE'] / real_metrics['MSE']) * 100
    
    print("\nResults:")
    print(f"   Real MSE:      {real_metrics['MSE']:.6f}")
    print(f"   Synthetic MSE: {synthetic_metrics['MSE']:.6f}")
    print(f"   Performance:   {performance_ratio:.1f}%")
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
