"""
Data loading — file readers and the training-time batch iterator.

CSVDataLoader    reads a benchmark CSV into a DataFrame
MiniBatchLoader  yields shuffled mini-batches straight from a tensor

MiniBatchLoader reshuffles with `torch.randperm` on every epoch, so it consumes
one RNG draw per epoch. That is what makes expert training reproducible under a
fixed seed — and what makes it sensitive to any other code that draws RNG in
between.
"""

from .base import BaseDataLoader
from .csv_loader import CSVDataLoader

__all__ = [
    "BaseDataLoader",
    "CSVDataLoader",
]
