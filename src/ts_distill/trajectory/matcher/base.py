from abc import ABC, abstractmethod

class BaseTrajectoryMatcher(ABC):
    """Interface for calculating distance between Student and Expert."""

    @abstractmethod
    def calculate_loss(self, student_params, expert_params) -> float:
        """
        Computes the distance metric (MSE, Cosine, etc.)
        Must be differentiable!
        """
        pass