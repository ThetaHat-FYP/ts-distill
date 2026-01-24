from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    
    @abstractmethod
    def plot_data(self, data, title=None):
        pass
    
    @abstractmethod
    def plot_comparison(self, real_data, synthetic_data, title=None):
        pass
    
    @abstractmethod
    def save_plot(self, filepath):
        pass
