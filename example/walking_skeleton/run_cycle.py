"""
═══════════════════════════════════════════════════════════════════════════════
MTT WALKING SKELETON - Unified Testing Framework
═══════════════════════════════════════════════════════════════════════════════

HOW TO USE:
1. Implement your component in src/ (inherit from base classes)
2. Replace the component below (lines 30-60)
3. Run: python example/walking_skeleton/run_cycle.py

That's it! The pipeline will automatically use your component.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# 🔌 REPLACE COMPONENTS HERE - Just swap the imports and assignments
# ═══════════════════════════════════════════════════════════════════════════════

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CORE COMPONENTS (Required - Always Needed)                                  │
# └─────────────────────────────────────────────────────────────────────────────┘

# ─── DATA LOADER ───
from example.walking_skeleton.etth1_loader import ETTh1DataLoader
csv_path = 'example/ETTh1.csv'
# Replace: from example.walking_skeleton.mock_components import MockDataLoader
# Replace: data_loader_class = MockDataLoader

# ─── MODEL ───
from example.walking_skeleton.mock_components import SimpleLSTM
def create_model():
    return SimpleLSTM(input_size=1, hidden_size=16, num_layers=1)
# Replace: from src.ts_distill.models.transformer import TransformerModel
# Replace: def create_model(): return TransformerModel(...)
# Replace: def create_model(): return TransformerModel(...)

# ─── TRAJECTORY RECORDER ───
from example.walking_skeleton.mock_components import SimpleRecorder
recorder_class = SimpleRecorder
# Replace: from src.ts_distill.trajectory.recorder.disk_recorder import DiskRecorder
# Replace: recorder_class = DiskRecorder

# ─── EVALUATOR ───
from example.walking_skeleton.mock_components import SimpleEvaluator
evaluator_class = SimpleEvaluator
# Replace: from src.ts_distill.evaluation.advanced_evaluator import AdvancedEvaluator
# Replace: evaluator_class = AdvancedEvaluator

# ─── MATCHER (trajectory matching strategy) ───
from example.walking_skeleton.mock_components import MSEMatcher
matcher_class = MSEMatcher
# Replace: from src.ts_distill.trajectory.matcher.cosine_matcher import CosineMatcher
# Replace: matcher_class = CosineMatcher

# ─── INITIALIZER (synthetic data initialization) ───
from example.walking_skeleton.mock_components import RealSampleInitializer
initializer_class = RealSampleInitializer
# Replace: from src.ts_distill.distillation_core.initializer.kmeans_init import KMeansInitializer
# Replace: initializer_class = KMeansInitializer

# ─── DISTILLER (distillation algorithm) ───
from example.walking_skeleton.mtt_distiller import MTTDistiller
distiller_class = MTTDistiller
# Replace: from src.ts_distill.distillation_core.distribution_distiller import DistributionDistiller
# Replace: distiller_class = DistributionDistiller


# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ OPTIONAL COMPONENTS (Advanced Features - Set to None if not needed)         │
# └─────────────────────────────────────────────────────────────────────────────┘

# ─── WINDOWING (adaptive window sizing) ───
from src.ts_distill.data_pipeline.data_windowing.fixed_windowing import FixedWindowing
windowing = FixedWindowing(window_size=96, stride=1)
# Use: from src.ts_distill.data_pipeline.data_windowing.adaptive_windowing import AdaptiveWindowing
# Use: windowing = AdaptiveWindowing(min_window=5, max_window=15, variance_threshold=0.5, stride=2)

# ─── TRAJECTORY SELECTOR (select informative trajectories) ───
trajectory_selector = None
# Use: from src.ts_distill.trajectory.selector.diversity_selector import DiversitySelector
# Use: trajectory_selector = DiversitySelector(n_trajectories=5, method='kmeans')

# ─── SAMPLE SELECTOR (select representative samples) ───
sample_selector = None
# Use: from src.ts_distill.evaluation.sample_selector.uncertainty_selector import UncertaintySelector
# Use: sample_selector = UncertaintySelector(selection_ratio=0.3)

# ─── ANCHOR SELECTOR (anchor-based selection) ───
anchor_selector = None
# Use: from src.ts_distill.trajectory.selector.anchor_selector import AnchorSelector
# Use: anchor_selector = AnchorSelector(n_anchors=10, strategy='farthest')

# ─── HYBRID MIXER (mix real and synthetic data) ───
hybrid_mixer = None
# Use: from src.ts_distill.evaluation.hybrid_evaluation.hybrid_mixer import HybridMixer
# Use: hybrid_mixer = HybridMixer(mix_ratio=0.5, strategy='interleave')

# ─── CUSTOM LOSS FUNCTION (alternative to MSE) ───
custom_loss = None
# Use: from src.ts_distill.distillation_core.distilation_loss.contrastive_loss import ContrastiveLoss
# Use: custom_loss = ContrastiveLoss(temperature=0.5)

# ─── CUSTOM OPTIMIZER (alternative to Adam) ───
optimizer_factory = None
# Use: optimizer_factory = lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)
# Default: Uses Adam if None

# ─── NORMALIZATION (data preprocessing) ───
from src.ts_distill.data_pipeline.data_preprocessor.normalization import StandardNormalization
normalizer = StandardNormalization()
# Use: from src.ts_distill.data_pipeline.data_preprocessor.normalization import MinMaxNormalization
# Use: normalizer = MinMaxNormalization()

# ─── VISUALIZER (plot real vs synthetic data) ───
from src.ts_distill.visualization import TimeSeriesVisualizer
visualizer = TimeSeriesVisualizer(figsize=(14, 5), max_samples=5, feature_idx=0, concatenate=True)
# Set to None to disable: visualizer = None

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline code below - No need to modify
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    """Main testing pipeline"""
    
    print("\n" + "="*70)
    print("   MTT WALKING SKELETON - Testing Your Components")
    print("="*70 + "\n")
    
    torch.manual_seed(42)
    
    # ────────────────────────────────────────────────────────────
    # STEP 1: Load Data
    # ────────────────────────────────────────────────────────────
    print("📊 STEP 1: Loading Data")
    print("─"*70)
    
    train_loader = ETTh1DataLoader(csv_path, n_samples=1000, seq_len=1, batch_size=1000, 
                                   target_column='OT', start_idx=0, single_sequence=True)
    train_loader.load_data()
    
    test_loader = ETTh1DataLoader(csv_path, n_samples=200, seq_len=1, batch_size=200, 
                                  target_column='OT', start_idx=5000, single_sequence=True)
    test_loader.load_data()
    
    train_data = train_loader.data
    test_data = test_loader.data
    
    print(f"✓ Train data: {train_data.shape}")
    print(f"✓ Test data: {test_data.shape}")
    print(f"✓ Train range: [{train_data.min():.2f}, {train_data.max():.2f}]\n")
    
    # ────────────────────────────────────────────────────────────
    # STEP 1.5: Normalize Data
    # ────────────────────────────────────────────────────────────
    print("🔧 Normalizing Data")
    print("─"*70)
    
    train_data = normalizer.fit_transform(train_data)
    test_data = normalizer.transform(test_data)
    
    print(f"✓ Normalized train range: [{train_data.min():.2f}, {train_data.max():.2f}]")
    
    # ────────────────────────────────────────────────────────────
    # STEP 2: Apply Windowing
    # ────────────────────────────────────────────────────────────
    if windowing is not None:
        print("\n🪟 STEP 2: Applying Fixed Windowing")
        print("─"*70)
        
        train_data, _ = windowing.create_windows(train_data)
        test_data, _ = windowing.create_windows(test_data)
        
        stats = windowing.get_statistics()
        print(f"✓ Windowed train data: {train_data.shape}")
        print(f"✓ Window size: {stats['window_size']}, Stride: {stats['stride']}")
        print(f"✓ Total windows: {stats['total_windows']}")
    
    print(f"\n✓ Final train data: {train_data.shape}")
    print(f"✓ Final test data: {test_data.shape}\n")
    
    # ────────────────────────────────────────────────────────────
    # STEP 3: Train Expert Model
    # ────────────────────────────────────────────────────────────
    print("🎓 STEP 3: Training Expert Model")
    print("─"*70)
    
    expert_model = create_model()
    recorder = recorder_class()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(expert_model.parameters(), lr=0.001)
    
    recorder.on_train_begin(expert_model)
    
    for epoch in range(20):
        optimizer.zero_grad()
        output = expert_model(train_data)
        
        # Handle both windowed and non-windowed targets
        if train_data.shape[1] == 1:
            targets = train_data[:, -1:, :]
        else:
            targets = train_data[:, -1, :].unsqueeze(1)
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            recorder.record_checkpoint(expert_model, epoch)
            if (epoch + 1) % 4 == 0:
                print(f"  Epoch {epoch+1}/20 | Loss: {loss.item():.6f}")
    
    recorder.on_train_end(expert_model)
    recorder = recorder_class()
    
    # Simple training loop
    expert_model.train()
    
    # Use custom optimizer if provided
    if optimizer_factory is not None:
        optimizer = optimizer_factory(expert_model.parameters())
    else:
        optimizer = torch.optim.Adam(expert_model.parameters(), lr=0.001)
    
    # Use custom loss if provided
    if custom_loss is not None:
        criterion = custom_loss
    else:
        criterion = torch.nn.MSELoss()
    
    recorder.on_train_begin(expert_model)
    
    for epoch in range(20):
        optimizer.zero_grad()
        output = expert_model(train_data)
        
        if train_data.shape[1] == 1:
            targets = train_data[:, -1:, :]
        else:
            targets = train_data[:, -1, :].unsqueeze(1)
        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            recorder.record_checkpoint(expert_model, epoch)
            if (epoch + 1) % 4 == 0:
                print(f"  Epoch {epoch+1}/20 | Loss: {loss.item():.6f}")
    
    recorder.on_train_end(expert_model)
    
    # Apply trajectory selection (if configured)
    if trajectory_selector is not None:
        print("🎯 Applying Trajectory Selection")
        print("─"*70)
        original_count = len(recorder.get_trajectory())
        recorder = trajectory_selector.select(recorder)
        selected_count = len(recorder.get_trajectory())
        print(f"✓ Selected {selected_count}/{original_count} trajectories")
    
    print(f"✓ Expert trained ({len(recorder.get_trajectory())} checkpoints)\n")
    
    # ────────────────────────────────────────────────────────────
    # STEP 4: Distill Synthetic Data
    # ────────────────────────────────────────────────────────────
    print("⚗️  STEP 4: Distilling Synthetic Data")
    print("─"*70)
    
    # Configure distiller with optional components
    distiller_config = {
        'initializer': initializer_class(),
        'matcher': matcher_class(),
        'model_factory': create_model,
        'expert_recorder': recorder,
        'synthetic_lr': 0.1,
        'student_lr': 0.01,
        'student_steps': 10
    }
    
    # Add anchor selector if configured
    if anchor_selector is not None:
        distiller_config['anchor_selector'] = anchor_selector
    
    distiller = distiller_class(**distiller_config)
    
    # Calculate 10% of real windowed data for compression
    n_synthetic = max(1, int(train_data.shape[0] * 0.1))
    temp_data = train_data.squeeze(0).unsqueeze(-1) if train_data.shape[0] == 1 else train_data
    synthetic_data = distiller.distill(temp_data, n_steps=30, n_synthetic=n_synthetic)
    
    print(f"✓ Synthetic data: {synthetic_data.shape}")
    print(f"✓ Compression: {temp_data.shape[0]} samples → {n_synthetic} samples")
    print(f"✓ Compression ratio: {(n_synthetic / temp_data.shape[0] * 100):.1f}%\n")
    
    # ────────────────────────────────────────────────────────────
    # STEP 5: Evaluate Performance
    # ────────────────────────────────────────────────────────────
    print("📈 STEP 5: Evaluating Performance")
    print("─"*70)
    
    evaluator = evaluator_class(model_factory=create_model, n_epochs=10, lr=0.001)
    
    # Train on real data
    real_model = evaluator.train_on_synthetic(train_data)
    real_metrics = evaluator.test_on_real(real_model, test_loader)
    
    # Train on synthetic data
    synthetic_model = evaluator.train_on_synthetic(synthetic_data)
    synthetic_metrics = evaluator.test_on_real(synthetic_model, test_loader)
    
    # Train on hybrid mix (if configured)
    hybrid_metrics = None
    if hybrid_mixer is not None:
        print("\n🔀 Evaluating Hybrid Mix")
        print("─"*70)
        hybrid_data = hybrid_mixer.mix(train_data, synthetic_data)
        hybrid_model = evaluator.train_on_synthetic(hybrid_data)
        hybrid_metrics = evaluator.test_on_real(hybrid_model, test_loader)
        print(f"✓ Hybrid MSE: {hybrid_metrics['MSE']:.6f}")
    
    # Results
    performance_ratio = (synthetic_metrics['MSE'] / real_metrics['MSE']) * 100
    compression = (n_synthetic / temp_data.shape[0]) * 100
    
    print(f"\n{'Metric':<20} {'Real Data':<15} {'Synthetic Data':<15}")
    print("─"*70)
    print(f"{'Samples':<20} {temp_data.shape[0]:<15} {n_synthetic:<15}")
    print(f"{'Test MSE':<20} {real_metrics['MSE']:<15.6f} {synthetic_metrics['MSE']:<15.6f}")
    print(f"{'Performance':<20} {'100%':<15} {f'{performance_ratio:.1f}%':<15}")
    print(f"{'Compression':<20} {'100%':<15} {f'{compression:.1f}%':<15}")
    
    # ────────────────────────────────────────────────────────────
    # STEP 6: Visualize Results (if configured)
    # ────────────────────────────────────────────────────────────
    if visualizer is not None:
        print("\n📊 STEP 6: Generating Visualizations")
        print("─"*70)
        
        print("Plotting real data...")
        visualizer.plot_data(train_data, title='Real ETTh1 Temperature Data (OT)')
        visualizer.save_plot('1_real_data.png')
        visualizer.close()
        
        print("Plotting synthetic data...")
        visualizer.plot_data(synthetic_data, title='Synthetic Temperature Data')
        visualizer.save_plot('2_synthetic_data.png')
        visualizer.close()
        
        print("Plotting comparison...")
    visualizer.plot_comparison(train_data, synthetic_data, title='Real vs Synthetic Temperature')
    visualizer.save_plot('3_comparison.png')
    visualizer.close()
    
    print("✓ Saved 3 visualization plots\n")
    
    # ────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("   ✅ TESTING COMPLETE")
    print("═"*70)

    
    return {
        'train_data': train_data,
        'synthetic_data': synthetic_data,
        'real_mse': real_metrics['MSE'],
        'synthetic_mse': synthetic_metrics['MSE'],
        'performance_ratio': performance_ratio
    }


if __name__ == "__main__":
    results = main()
