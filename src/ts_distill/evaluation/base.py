from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """Interface for testing synthetic data quality."""

    @abstractmethod
    def train_on_synthetic(self, synthetic_data, model):
        """Trains a fresh model from scratch using ONLY synthetic data."""
        pass

    @abstractmethod
    def test_on_real(self, model, real_test_loader):
        """
        Evaluates the model on real held-out data.
        Returns: Dictionary of metrics {'RMSE': x, 'MAE': y}
        """
        pass