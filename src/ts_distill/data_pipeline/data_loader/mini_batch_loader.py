"""
Mini-Batch Loader
-----------------
A lightweight, tensor-native data loader designed for in-memory datasets.
It avoids the overhead of PyTorch's full DataLoader for cases where the
entire dataset fits in RAM (or GPU memory).
"""

import torch


class MiniBatchLoader:
    """
    Yields randomly shuffled mini-batches directly from a tensor.

    Unlike torch.utils.data.DataLoader, this operates on a plain tensor
    rather than a Dataset object — no workers, no collation, no overhead.
    Ideal for synthetic data and small-to-medium real splits that fit in memory.

    Args:
        data (torch.Tensor): Full dataset, shape (N, *).
        batch_size (int):    Samples per yielded batch.
        shuffle (bool):      Reshuffle indices on every iteration.

    Example::

        loader = MiniBatchLoader(train_data, batch_size=64)
        for batch in loader:          # shape: (64, window_size, features)
            loss = model(batch)...
    """

    def __init__(self, data: torch.Tensor, batch_size: int = 64, shuffle: bool = True):
        self.data       = data
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __iter__(self):
        n   = self.data.shape[0]
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            yield self.data[idx[i : i + self.batch_size]]

    def __len__(self) -> int:
        """Number of batches per full pass over the data."""
        return (self.data.shape[0] + self.batch_size - 1) // self.batch_size
