from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Interface for measuring model accuracy on held-out real data."""

    @abstractmethod
    def test_on_real(self, model, real_test_data):
        """
        Evaluate a trained model on real test data.

        Args:
            model          : A trained nn.Module.
            real_test_data : Tensor (N, window_size, features).

        Returns:
            dict of metrics e.g. {'MSE': float, 'RMSE': float}.
        """
        pass
