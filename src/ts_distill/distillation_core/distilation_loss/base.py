from abc import ABC, abstractmethod

class BaseDistillationLoss(ABC):
    """Interface for additional losses (e.g., Frequency Loss)."""

    @abstractmethod
    def compute(self, synthetic_output, real_output):
        """Calculates specific loss component."""
        pass