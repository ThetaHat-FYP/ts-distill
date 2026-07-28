"""
Evaluator interface — every utility measurement goes through this contract.

Fidelity metrics ask whether the synthetic data LOOKS like a time series;
evaluators ask whether it TRAINS a model that works. The rule is always the
same: train on whatever data is being judged, then score on REAL held-out test
data. Scoring on synthetic data would only measure self-consistency.
"""

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
