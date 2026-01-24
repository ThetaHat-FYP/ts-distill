from abc import ABC, abstractmethod
import pandas as pd

class BaseWindowing(ABC):
    """
    Interface for slicing time-series.
    Allows implementation of 'FixedWindowing' or 'AdaptiveWindowing' later.
    """
    
    @abstractmethod
    def create_windows(self, data: pd.DataFrame, **kwargs):
        """
        Key method to generate X (inputs) and Y (targets).
        Returns: (inputs, targets) tensors/arrays.
        """
        pass

    @abstractmethod
    def view_windowed_sample(self, index: int):
        """Visualizes or prints a single window sample (Input vs Target)."""
        pass