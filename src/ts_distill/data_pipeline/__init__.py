"""
Data pipeline
=============
Loading, splitting, and windowing of raw time-series data.

Sub-packages
------------
data_loader : CSV loading and mini-batch iteration.

Modules
-------
splitter    : Train/val/test boundary computation and sliding-window creation.
"""

from ts_distill.data_pipeline.splitter import get_data_splits, make_windows

__all__ = [
    'get_data_splits',
    'make_windows',
]
