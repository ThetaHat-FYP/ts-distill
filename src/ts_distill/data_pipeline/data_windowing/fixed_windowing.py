import torch
from .base import BaseWindowing


class FixedWindowing(BaseWindowing):
    
    def __init__(self, window_size=96, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.statistics = {}
    
    def create_windows(self, data):
        if torch.is_tensor(data):
            data = data.squeeze(0) if data.shape[0] == 1 else data
            
            if len(data.shape) == 1:
                data = data.unsqueeze(-1)
            
            n_samples = data.shape[0]
            n_features = data.shape[1] if len(data.shape) > 1 else 1
            
            windows = []
            for i in range(0, n_samples - self.window_size + 1, self.stride):
                window = data[i:i + self.window_size]
                windows.append(window)
            
            windowed_data = torch.stack(windows)
            
            self.statistics = {
                'total_windows': len(windows),
                'window_size': self.window_size,
                'stride': self.stride,
                'original_length': n_samples
            }
            
            return windowed_data, self.statistics
        
        raise TypeError("Input must be a torch.Tensor")
    
    def view_windowed_sample(self, index: int):
        """Simple print of window shape (not critical for current usage)"""
        stats = self.get_statistics()
        print(f"Window {index}: shape=({stats['window_size']}, features)")
        return stats
    
    def get_statistics(self):
        return self.statistics
