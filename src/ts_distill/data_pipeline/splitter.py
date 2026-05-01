"""
Data Splitter
-------------
Utilities for computing train / validation / test index boundaries and for
converting a 2-D time-series array into overlapping fixed-size windows.

Two split strategies are supported:

  benchmark_borders
      Hard-coded row indices that exactly reproduce the train/val/test splits
      used in published papers (e.g., Informer, DLinear on ETT datasets).
      The config must supply 'border1s' (start indices) and 'border2s' (end
      indices) lists, each with three entries for [train, val, test].

  ratios
      Proportional splits using 'train_ratio' and 'val_ratio' floats.
      Useful for custom or general-purpose datasets.

In both modes a seq_len-sized overlap is preserved at split boundaries so
that the first sliding window of each later split has a complete lookback.
"""

from typing import Tuple

import numpy as np


def get_data_splits(
    values: np.ndarray,
    window_size: int,
    seq_len: int,
    dataset_cfg: dict,
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute row-level index boundaries for train, validation, and test splits.

    Args:
        values (np.ndarray): Raw data array, shape (T, C).
        window_size (int):   Total window length (seq_len + pred_len). Kept as
                             a parameter for future use with adaptive windowing.
        seq_len (int):       Input look-back length. Used to compute the
                             seq_len-overlap between adjacent splits.
        dataset_cfg (dict):  One entry from CONFIG['datasets']. Must contain
                             'split_mode' and the corresponding keys.

    Returns:
        Tuple[int, int, int, int, int, int]:
            (train_start, train_end, val_start, val_end, test_start, test_end)

    Raises:
        ValueError: If 'split_mode' is not recognised.
    """
    mode = dataset_cfg['split_mode']

    if mode == 'benchmark_borders':
        # Work on copies so the original config lists are never mutated.
        border1s = list(dataset_cfg['border1s'])
        border2s = list(dataset_cfg['border2s'])

        # Recalculate the overlap boundaries dynamically.
        # This makes the splits correct even if seq_len changes from the default.
        border1s[1] = border2s[0] - seq_len   # val starts seq_len before train ends
        border1s[2] = border2s[1] - seq_len   # test starts seq_len before val ends

        train_start, train_end = border1s[0], border2s[0]
        val_start,   val_end   = border1s[1], border2s[1]
        test_start,  test_end  = border1s[2], border2s[2]

    elif mode == 'ratios':
        n_total = len(values)
        n_train = int(n_total * dataset_cfg['train_ratio'])
        n_val   = int(n_total * dataset_cfg['val_ratio'])

        train_start, train_end = 0, n_train
        # Pull val_start back by seq_len so windows at the boundary are complete.
        val_start,   val_end   = n_train - seq_len, n_train + n_val
        test_start,  test_end  = n_train + n_val - seq_len, n_total

    else:
        raise ValueError(
            f"Unknown split_mode '{mode}'. "
            f"Choose 'benchmark_borders' or 'ratios'."
        )

    return train_start, train_end, val_start, val_end, test_start, test_end


def make_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convert a 2-D time-series array into a 3-D array of overlapping windows.

    Each window slides one step forward from the previous one, so consecutive
    windows share (window_size - 1) rows. Within each window, the first
    seq_len rows are the model input and the remaining pred_len rows are the
    forecast target — but this function does not split them; the Trainer does.

    Args:
        arr (np.ndarray): Shape (T, C) — T timesteps, C channels/features.
        window_size (int): Length of each window (seq_len + pred_len).

    Returns:
        np.ndarray: Shape (N, window_size, C) where N = T - window_size + 1.

    Example::

        # 8640 training rows, window_size = 192  →  8449 windows
        train_windows = make_windows(data[train_start:train_end], 192)
        # train_windows.shape == (8449, 192, 7)
    """
    return np.stack(
        [arr[i : i + window_size] for i in range(len(arr) - window_size + 1)]
    )
