from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):
    """Interface for cleaning and scaling."""
    
    @abstractmethod
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean', fill_value=None) -> pd.DataFrame:
        """
        Strategies: 'drop', 'mean', 'median', 'mode', 'constant'.
        """
        pass

    @abstractmethod
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Methods: 'minmax', 'standard' (z-score).
        Should save scaler params to reverse later if needed.
        """
        pass

    @abstractmethod
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8):
        """Returns (train_set, test_set)."""
        pass