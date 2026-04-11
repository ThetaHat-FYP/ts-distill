"""
MTT Walking Skeleton - End-to-End Pipeline
Time series data distillation using Matching Training Trajectories.
"""

import torch
import sys
from pathlib import Path

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
    'n_train_samples': 15000,
    'n_test_samples': 3000,
    'test_start_idx': 5000,
    'target_column': 'OT',
    
    # Windowing
    'window_size': 48,
    'stride': 1,
    
    # Expert Training
    'expert_epochs': 80,
    'expert_lr': 0.0005, 
    'expert_momentum': 0.9,
    
    # Distillation
    'n_distill_steps': 50,
    'compression_ratio': 0.0333,
    'synthetic_lr': 0.1,
    'student_lr': 0.0003,
    'student_steps': 5,
    'trajectory_gap': 5,
    'n_synthetic': 10, 
    'cond_gap': 3,
    'beta': 0.01,
    'frepo_online_lr': 0.001,
    'frepo_online_updates': 10,
    'frepo_ridge_lambda': 0.001,
    
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
    return DLinear(
        seq_len=CONFIG['seq_len'], 
        pred_len=CONFIG['pred_len'],
        channels=CONFIG['in_features'],
        individual=False
    )

    # return MLP(seq_len=CONFIG['seq_len'], pred_len=CONFIG['pred_len'])

    # return LSTM(
    #     input_dim=CONFIG['in_features'],
    #     embed_dim=CONFIG['seq_len'],
    #     hidden_dim=CONFIG['hidden_size'],
    #     num_layer=CONFIG['num_layers'],
    #     horizon=CONFIG['pred_len']
    # )

    # return CNN(channel=CONFIG['in_features'])


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    
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
    # distiller = MTTDistiller(
    #     initializer=RealSampleInitializer(),
    #     matcher=MSEMatcher(),
    #     model_factory=create_model,
    #     expert_recorder=recorder,
    #     expert_epochs=CONFIG['trajectory_gap'],
    #     synthetic_lr=CONFIG['synthetic_lr'],
    #     student_lr=CONFIG['student_lr'],
    #     student_steps=CONFIG['student_steps']
    # )

    # distiller = CondTSFDistiller(
    #     initializer=RealSampleInitializer(),
    #     matcher=MSEMatcher(),
    #     model_factory=create_model,
    #     teacher_state_dict={k: v.detach().cpu().clone() for k, v in expert_model.state_dict().items()},
    #     expert_epochs=CONFIG['trajectory_gap'],
    #     synthetic_lr=CONFIG['synthetic_lr'],
    #     student_lr=CONFIG['student_lr'],
    #     student_steps=CONFIG['student_steps'],
    #     cond_gap=CONFIG['cond_gap'],
    #     beta=CONFIG['beta'],
    #     device=device
    # )

    distiller = FRePODistiller(
        initializer=RealSampleInitializer(),
        matcher=MSEMatcher(),
        model_factory=create_model,
        synthetic_lr=CONFIG['synthetic_lr'],
        online_lr=CONFIG['frepo_online_lr'],
        online_updates=CONFIG['frepo_online_updates'],
        syn_batch_size=min(128, max(1, int(train_data.shape[0] * CONFIG['compression_ratio']))),
        real_batch_size=256,
        ridge_lambda=CONFIG['frepo_ridge_lambda'],
        device=device,
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
