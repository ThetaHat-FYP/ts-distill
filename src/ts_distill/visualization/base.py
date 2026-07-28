"""
Visualizer interface — plotting helpers for inspecting synthetic data.

Plots are how the temporal damage becomes obvious: distilled data that scores
well on MSE can look nothing like a time series, and a before/after plot of the
FFT post-fix shows the restored periodicity at a glance.
"""

from abc import ABC, abstractmethod


class BaseVisualizer(ABC):
    """Contract for plotters: draw one sequence, compare two, save the result."""

    @abstractmethod
    def plot_data(self, data, title=None):
        """Plot a single sequence."""
        pass

    @abstractmethod
    def plot_comparison(self, real_data, synthetic_data, title=None):
        """Plot real against synthetic for side-by-side inspection."""
        pass

    @abstractmethod
    def save_plot(self, filepath):
        """Write the current figure to disk."""
        pass
