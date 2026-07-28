"""
CSV loading — read a benchmark file into a DataFrame, nothing more.

Deliberately stops at loading. Splitting, scaling, and windowing must happen in
that order and only after the train boundary is known, or test statistics leak
into training; those steps live in `splitter` and the calling script.

Callers drop column 0 themselves — the benchmark CSVs carry a date column that
is not a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
from pandas.errors import DtypeWarning

from .base import BaseDataLoader


@dataclass
class CSVDataLoader(BaseDataLoader):
    """Generic CSV loader that implements `BaseDataLoader`.

    Notes:
        - Stores the last loaded dataframe in `data`.
        - Pass any `pandas.read_csv` kwargs via `read_csv_kwargs`.
    """

    read_csv_kwargs: dict[str, Any] = field(default_factory=dict)
    data: pd.DataFrame | None = None
    file_path: str | None = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Read a CSV into a DataFrame and cache it on `self.data`.

        Args:
            file_path (str): Path to the CSV.

        Returns:
            pd.DataFrame: The loaded table, column 0 still the date column.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        if path.is_dir():
            raise IsADirectoryError(f"Expected a CSV file path, got directory: {file_path}")

        read_csv_kwargs = dict(self.read_csv_kwargs)
        # Reduce noisy dtype warnings for large/mixed-type CSVs.
        read_csv_kwargs.setdefault("low_memory", False)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DtypeWarning)
                self.data = pd.read_csv(path, **read_csv_kwargs)
        except UnicodeDecodeError:
            # If the caller didn't specify an encoding, retry with a common Windows-compatible
            # fallback. This keeps the loader usable for "real world" CSV exports.
            if "encoding" in read_csv_kwargs:
                raise

            fallback_kwargs = dict(read_csv_kwargs)
            fallback_kwargs["encoding"] = "latin1"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DtypeWarning)
                self.data = pd.read_csv(path, **fallback_kwargs)
        self.file_path = str(path)
        return self.data

    def view_data(self, n_rows: int = 5) -> None:
        """
        Print the first `n_rows` with all columns visible.

        Prints rather than logs on purpose: the output IS the return value here,
        because the caller explicitly asked to see the data.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data(file_path) first.")

        n_rows = max(0, int(n_rows))
        with pd.option_context(
            "display.max_columns", None,
            "display.width", None,
            "display.expand_frame_repr", False,
            "display.max_colwidth", 80,
        ):
            print(self.data.head(n_rows).to_string(index=False))
