"""
Sample selectors
================
Interface for choosing which real windows ("anchors") get mixed into synthetic
data during hybrid evaluation.

Concrete implementations live in
``ts_distill.evaluation.hybrid_evaluation.hybrid`` (RandomAnchorSelector,
UniformStrideSelector, ImportanceWeightedSelector, DiversityAnchorSelector).
"""

from ts_distill.evaluation.sample_selector.base import BaseAnchorSelector

__all__ = ['BaseAnchorSelector']
