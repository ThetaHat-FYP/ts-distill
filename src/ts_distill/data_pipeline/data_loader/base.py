"""
Data loader interface — read a raw dataset file into a DataFrame.

Deliberately thin, and deliberately does no splitting, scaling, or windowing.
Those steps must happen in a fixed order and only after the train boundary is
known, so they live in `splitter` and in the calling script instead.
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseDataLoader(ABC):
    """Interface for loading and initial viewing of data."""
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a generic source (CSV, Parquet, etc)."""
        pass

    @abstractmethod
    def view_data(self, n_rows: int = 5) -> None:
        """Prints the top n rows and columns to console."""
        pass