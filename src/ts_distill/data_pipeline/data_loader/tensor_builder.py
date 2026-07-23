from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch


CategoricalStrategy = Literal["codes"]
DateStrategy = Literal["to_unix_seconds"]
MissingStrategy = Literal["zero", "ffill", "median"]


@dataclass(frozen=True)
class TabularToTensorConfig:
    """Configuration for converting a full CSV dataframe into numeric tensors.

    This keeps the walking skeleton generic: any CSV with mixed dtypes can be converted
    into a multivariate numeric sequence.
    """

    categorical_strategy: CategoricalStrategy = "codes"
    date_strategy: DateStrategy = "to_unix_seconds"
    missing_strategy: MissingStrategy = "zero"


def dataframe_to_feature_matrix(
    df: pd.DataFrame,
    *,
    config: TabularToTensorConfig = TabularToTensorConfig(),
) -> np.ndarray:
    """Convert ALL columns of a dataframe into a numeric feature matrix.

    Strategy:
    - Numeric-like columns -> float
    - Datetime-like columns -> unix seconds (float)
    - Other/object columns -> categorical codes (float)

    Returns:
        np.ndarray of shape (T, F) with dtype float32.
    """

    if df.shape[0] == 0:
        raise ValueError("Dataframe is empty; cannot build features.")

    feature_columns: list[np.ndarray] = []

    for col_name in df.columns:
        series = df[col_name]

        # 1) Try numeric
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_non_na = int(numeric.notna().sum())

        if numeric_non_na > 0:
            values = numeric.astype(np.float32).to_numpy(copy=True)
            feature_columns.append(values)
            continue

        # 2) Try datetime
        dt = pd.to_datetime(series, errors="coerce")
        dt_non_na = int(dt.notna().sum())
        if dt_non_na > 0:
            if config.date_strategy == "to_unix_seconds":
                # pandas datetime64[ns] -> int64 nanoseconds (via numpy)
                dt_np = dt.to_numpy(dtype="datetime64[ns]")
                ns = dt_np.astype("int64").astype(np.float64)
                seconds = ns / 1e9
                seconds[dt.isna().to_numpy()] = np.nan
                feature_columns.append(seconds.astype(np.float32))
                continue

        # 3) Categorical fallback
        if config.categorical_strategy == "codes":
            codes = pd.Categorical(series).codes.astype(np.float32)
            # pandas uses -1 for NaN/unseen
            feature_columns.append(np.asarray(codes, dtype=np.float32))
            continue

        raise ValueError(f"Unsupported conversion strategy for column: {col_name}")

    matrix = np.stack(feature_columns, axis=1).astype(np.float32, copy=False)

    # Handle missing values / invalids
    matrix[~np.isfinite(matrix)] = np.nan

    if config.missing_strategy == "zero":
        matrix = np.nan_to_num(matrix, nan=0.0)
    elif config.missing_strategy == "ffill":
        matrix = pd.DataFrame(matrix).ffill().fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    elif config.missing_strategy == "median":
        medians = np.nanmedian(matrix, axis=0)
        inds = np.where(np.isnan(matrix))
        matrix[inds] = np.take(medians, inds[1])
        matrix = np.nan_to_num(matrix, nan=0.0)
    else:
        raise ValueError(f"Unknown missing_strategy: {config.missing_strategy}")

    return matrix


def make_sequence_tensor_from_dataframe(
    df: pd.DataFrame,
    *,
    start_idx: int,
    n_rows: int,
    config: TabularToTensorConfig = TabularToTensorConfig(),
) -> torch.Tensor:
    """Build a single multivariate sequence tensor of shape (1, T, F)."""

    if start_idx < 0:
        raise ValueError(f"start_idx must be >= 0, got {start_idx}")
    if n_rows <= 0:
        raise ValueError(f"n_rows must be > 0, got {n_rows}")
    if start_idx >= len(df):
        raise ValueError(f"start_idx {start_idx} is >= dataset length {len(df)}")

    end_idx = min(start_idx + n_rows, len(df))
    window_df = df.iloc[start_idx:end_idx]

    features = dataframe_to_feature_matrix(window_df, config=config)
    tensor = torch.from_numpy(features).unsqueeze(0)  # (1, T, F)
    return tensor
