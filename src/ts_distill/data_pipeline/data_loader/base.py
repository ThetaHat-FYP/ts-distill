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