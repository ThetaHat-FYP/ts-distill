import matplotlib.pyplot as plt
import torch
import numpy as np
from .base import BaseVisualizer


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
    
    def save_plot(self, filepath):
        if self.fig is None:
            raise ValueError("No plot to save. Create a plot first.")
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
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
