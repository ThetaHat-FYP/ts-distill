import torch
from river import drift

from .base import BaseWindowing


class ADWINWindowing(BaseWindowing):
    """Adaptive windowing based on concept-drift detection (ADWIN).

    This implementation detects drift points in a time-series stream using
    River's `ADWIN` detector, then generates **fixed-length** sliding windows
    within each stable segment.

    Why fixed-length windows?
    - Downstream trainers/models typically expect a rectangular tensor
      shaped like `(n_windows, window_size, n_features)`.
    - Variable-length windows would require padding or ragged batching.

    Parameters
    ----------
    window_size:
        Length of each produced training window.
    stride:
        Step size between consecutive windows within a segment.
    delta:
        ADWIN confidence parameter. Smaller values = fewer drifts.
    feature_index:
        If set, ADWIN monitors only this feature (column) over time.
        If None, ADWIN monitors an aggregate value across features.
    reduction:
        How to aggregate a feature vector into a scalar when `feature_index`
        is None. One of: `"mean"`, `"sum"`, `"l2"`.
    min_segment_size:
        Segments shorter than this are ignored for window generation.
        Defaults to `window_size`.
    warmup:
        Number of initial timesteps to feed ADWIN before allowing drift splits.

    Minimal usage
    -------------
    ```python
    import torch
    from src.ts_distill.data_pipeline.data_windowing import ADWINWindowing

    x = torch.randn(1, 10_000, 4)  # (batch=1, time, features)
    windowing = ADWINWindowing(window_size=48, stride=1, delta=0.002)
    windows, stats = windowing.create_windows(x)
    print(windows.shape)
    print(stats["drift_points"])
    ```
    """

    def __init__(
        self,
        window_size: int = 96,
        stride: int = 1,
        *,
        delta: float = 0.002,
        feature_index: int | None = None,
        reduction: str = "mean",
        min_segment_size: int | None = None,
        warmup: int = 0,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.delta = float(delta)
        self.feature_index = feature_index
        self.reduction = reduction
        self.min_segment_size = int(min_segment_size) if min_segment_size is not None else int(window_size)
        self.warmup = int(warmup)

        self._statistics: dict = {}

    def _scalarize(self, x_t: torch.Tensor) -> float:
        if self.feature_index is not None:
            return float(x_t[int(self.feature_index)].item())

        if self.reduction == "mean":
            return float(x_t.mean().item())
        if self.reduction == "sum":
            return float(x_t.sum().item())
        if self.reduction == "l2":
            return float(torch.linalg.vector_norm(x_t).item())

        raise ValueError('reduction must be one of: "mean", "sum", "l2"')

    def create_windows(self, data):
        if not torch.is_tensor(data):
            raise TypeError("Input must be a torch.Tensor")

        # Match FixedWindowing behavior
        data = data.squeeze(0) if data.ndim == 3 and data.shape[0] == 1 else data
        if data.ndim == 1:
            data = data.unsqueeze(-1)

        n_samples = int(data.shape[0])
        n_features = int(data.shape[1])

        detector = drift.ADWIN(delta=self.delta)
        drift_points: list[int] = []

        for t in range(n_samples):
            detector.update(self._scalarize(data[t]))
            if t >= self.warmup and detector.drift_detected:
                # Drift detected *after* ingesting data[t]; treat t as the first
                # point of the new regime.
                drift_points.append(t)

        # Build stable segments [start, end)
        cut_points = [0] + drift_points + [n_samples]
        segments: list[tuple[int, int]] = []
        for s, e in zip(cut_points[:-1], cut_points[1:]):
            if (e - s) >= self.min_segment_size:
                segments.append((s, e))

        windows: list[torch.Tensor] = []
        for s, e in segments:
            for i in range(s, e - self.window_size + 1, self.stride):
                windows.append(data[i : i + self.window_size])

        if windows:
            windowed_data = torch.stack(windows)
        else:
            windowed_data = torch.empty((0, self.window_size, n_features), dtype=data.dtype, device=data.device)

        self._statistics = {
            "total_windows": int(windowed_data.shape[0]),
            "window_size": self.window_size,
            "stride": self.stride,
            "original_length": n_samples,
            "n_features": n_features,
            "delta": self.delta,
            "feature_index": self.feature_index,
            "reduction": self.reduction,
            "warmup": self.warmup,
            "drift_points": drift_points,
            "n_segments": len(segments),
            "segment_lengths": [e - s for s, e in segments],
        }

        return windowed_data, self._statistics

    def view_windowed_sample(self, index: int):
        stats = self.get_statistics()
        print(
            f"Window {index}: shape=({stats.get('window_size')}, {stats.get('n_features')}) | "
            f"drifts={len(stats.get('drift_points', []))}"
        )
        return stats

    def get_statistics(self):
        return self._statistics
