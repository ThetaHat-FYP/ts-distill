from abc import ABC, abstractmethod

class BaseInitializer(ABC):
    """Interface for starting the synthetic data."""

    @abstractmethod
    def initialize(self, shape, real_data_reference=None):
        """
        Returns initial tensor.
        real_data_reference: Optional, for 'RealSampleInitializer'.
        """
        pass