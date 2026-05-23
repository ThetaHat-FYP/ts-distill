from abc import ABC, abstractmethod

import numpy as np


class BaseTemporalMetric(ABC):
    """
    Interface for computing a temporal statistical metric between a real
    and a synthetic time-series sequence.

    Data contract
    -------------
    Both arrays are un-windowed, continuously-indexed, and already normalised
    (StandardScaler fitted on the real training split):

      real      : shape (T, C)  — full real training sequence
      synthetic : shape (M, C)  — distilled synthetic sequence (typically M=384)

    Each subclass returns a per-channel result so that MetricAggregator can
    build both the feature-wise table (one row per channel) and the main table
    (one row per experiment, channels averaged).

    Most metrics return shape (C,).
    ACFMetric returns shape (C, 2) — column 0 = acf_short, column 1 = acf_long.
    """

    @abstractmethod
    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
        """
        Compute the metric independently for every channel.

        Args:
            real      (np.ndarray): Shape (T, C).
            synthetic (np.ndarray): Shape (M, C).

        Returns:
            np.ndarray: Per-channel result.
                Most metrics  → shape (C,)
                ACFMetric     → shape (C, 2)
        """
        pass
