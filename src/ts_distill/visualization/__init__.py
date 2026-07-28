"""
Visualization — plots for inspecting real vs synthetic sequences.

Used to produce the before/after post-fix figures: the same synthetic data at
alpha = 0 and alpha > 0, where restored periodicity is visible directly rather
than only through a metric.
"""

from .base import BaseVisualizer
from .ts_visualizer import TimeSeriesVisualizer

__all__ = ['BaseVisualizer', 'TimeSeriesVisualizer']
