import matplotlib.pyplot as plt
import torch
import numpy as np
from .base import BaseVisualizer

from ts_distill._logging import get_logger

logger = get_logger(__name__)


class TimeSeriesVisualizer(BaseVisualizer):
    
    def __init__(self, figsize=(12, 6), max_samples=10, feature_idx=0, concatenate=True):
        self.figsize = figsize
        self.max_samples = max_samples
        self.feature_idx = feature_idx
        self.concatenate = concatenate
        self.fig = None
        self.ax = None
    
    def plot_data(self, data, title=None):
        data = self._to_numpy(data)
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        if self.concatenate:
            n_samples = min(len(data), self.max_samples)
            continuous_data = np.concatenate(data[:n_samples], axis=0)
            self.ax.plot(continuous_data, alpha=0.8, linewidth=1.5, color='blue')
        else:
            n_samples = min(len(data), self.max_samples)
            for i in range(n_samples):
                self.ax.plot(data[i], alpha=0.6, linewidth=1.5)
        
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Value')
        self.ax.set_title(title or 'Time Series Data')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig
    
    def plot_comparison(self, real_data, synthetic_data, title=None):
        real_data = self._to_numpy(real_data)
        synthetic_data = self._to_numpy(synthetic_data)
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        if self.concatenate:
            n_real = min(len(real_data), self.max_samples)
            n_synthetic = min(len(synthetic_data), self.max_samples)
            
            real_continuous = np.concatenate(real_data[:n_real], axis=0)
            synthetic_continuous = np.concatenate(synthetic_data[:n_synthetic], axis=0)
            
            self.ax.plot(real_continuous, alpha=0.7, linewidth=1.5, color='blue', label='Real')
            self.ax.plot(synthetic_continuous, alpha=0.7, linewidth=1.5, color='orange', label='Synthetic')
        else:
            n_real = min(len(real_data), self.max_samples)
            n_synthetic = min(len(synthetic_data), self.max_samples)
            
            for i in range(n_real):
                self.ax.plot(real_data[i], alpha=0.5, linewidth=1.5, color='blue',
                            label='Real' if i == 0 else None)
            
            for i in range(n_synthetic):
                self.ax.plot(synthetic_data[i], alpha=0.5, linewidth=1.5, color='orange',
                            label='Synthetic' if i == 0 else None)
        
        self.ax.set_xlabel('Time Steps')
        self.ax.set_ylabel('Value')
        self.ax.set_title(title or 'Real vs Synthetic Data')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig
    
    def plot_overlay(self, real_data, synthetic_data, title=None):
        return self.plot_comparison(real_data, synthetic_data, title)
    
    def plot_side_by_side(self, real_data, synthetic_data, n_samples=3):
        """
        Plot real and synthetic windows side-by-side for direct visual comparison.
        
        Args:
            real_data: Tensor of shape (N_real, seq_len, features)
            synthetic_data: Tensor of shape (N_syn, seq_len, features)
            n_samples: Number of sample pairs to plot (default: 3)
        
        Returns:
            matplotlib figure
        """
        real_data = self._to_numpy(real_data)
        synthetic_data = self._to_numpy(synthetic_data)
        
        # Sample random indices
        n_real = min(len(real_data), n_samples)
        n_syn = min(len(synthetic_data), n_samples)
        n_rows = max(n_real, n_syn)
        
        real_indices = np.random.choice(len(real_data), n_real, replace=False)
        syn_indices = np.random.choice(len(synthetic_data), n_syn, replace=False)
        
        # Calculate global min/max from real data for consistent y-axis
        global_min = np.min(real_data)
        global_max = np.max(real_data)
        y_margin = (global_max - global_min) * 0.05
        ylim = (global_min - y_margin, global_max + y_margin)
        
        # Create subplots
        self.fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot real data (left column)
        for i in range(n_rows):
            ax_left = axes[i, 0]
            if i < n_real:
                ax_left.plot(real_data[real_indices[i]], color='blue', linewidth=2, alpha=0.8)
                ax_left.set_title(f'Real Sample {i+1}', fontsize=12, fontweight='bold')
            else:
                ax_left.axis('off')
            
            ax_left.set_ylim(ylim)
            ax_left.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax_left.set_xlabel('Timestep', fontsize=10)
            ax_left.set_ylabel('Normalized Value', fontsize=10)
        
        # Plot synthetic data (right column)
        for i in range(n_rows):
            ax_right = axes[i, 1]
            if i < n_syn:
                ax_right.plot(synthetic_data[syn_indices[i]], color='orange', linewidth=2, alpha=0.8)
                ax_right.set_title(f'Syn Sample {i+1}', fontsize=12, fontweight='bold')
            else:
                ax_right.axis('off')
            
            ax_right.set_ylim(ylim)
            ax_right.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax_right.set_xlabel('Timestep', fontsize=10)
            ax_right.set_ylabel('Normalized Value', fontsize=10)
        
        # Main title
        self.fig.suptitle('Visual Fidelity Check: Real vs Synthetic Windows', 
                         fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        return self.fig
    
    def save_plot(self, filepath):
        if self.fig is None:
            raise ValueError("No plot to save. Create a plot first.")
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
    
    def show(self):
        plt.show()
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _to_numpy(self, data):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        
        if len(data.shape) == 3:
            data = data[:, :, self.feature_idx]
        
        return data
