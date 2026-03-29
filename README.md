# ts-distill

## Generic CSV loading

Use `CSVDataLoader` to load and preview any `.csv` file:

```python
from src.ts_distill.data_pipeline.data_loader import CSVDataLoader

loader = CSVDataLoader()
df = loader.load_data("path/to/your_file.csv")
loader.view_data(n_rows=5)
```