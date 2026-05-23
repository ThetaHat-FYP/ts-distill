"""
Standalone initializer diagnostics.

Creates a diagnostic plot and volatility summary to compare the
GeometrySequenceInitializer against random sliding windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional
    sns = None

import matplotlib.pyplot as plt

from ts_distill.data_pipeline.splitter import get_data_splits
from ts_distill.distillation_core.initializer.geommetry_sequence_initializer import (
    GeometrySequenceInitializer,
)


DEFAULT_DATASET_CONFIG: Dict[str, Dict[str, object]] = {
    "ETTh1": {
        "csv_path": "example/ETTh1.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
    },
    "ETTh2": {
        "csv_path": "example/ETTh2.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96],
        "border2s": [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24],
    },
    "ETTm1": {
        "csv_path": "example/ETTm1.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 96 - 96, 12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s": [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
    },
    "ETTm2": {
        "csv_path": "example/ETTm2.csv",
        "split_mode": "benchmark_borders",
        "border1s": [0, 12 * 30 * 96 - 96, 12 * 30 * 96 + 4 * 30 * 96 - 96],
        "border2s": [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96],
    },
}


def _load_raw_train_data(
    dataset_name: str,
    seq_len: int,
    pred_len: int,
    override_csv: str | None,
) -> Tuple[np.ndarray, str]:
    dataset_cfg = DEFAULT_DATASET_CONFIG.get(dataset_name)
    if dataset_cfg is None:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")

    csv_path = Path(override_csv) if override_csv else Path(dataset_cfg["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Provide --csv-path or place the file."
        )

    df_raw = pd.read_csv(csv_path)
    values = df_raw.iloc[:, 1:].values.astype(np.float32)

    window_size = seq_len + pred_len
    train_start, train_end, _, _, _, _ = get_data_splits(
        values=values,
        window_size=window_size,
        seq_len=seq_len,
        dataset_cfg=dataset_cfg,
    )

    scaler = StandardScaler()
    train_values = values[train_start:train_end]
    train_values = scaler.fit_transform(train_values)

    return train_values, str(csv_path)


def _make_mock_data(num_rows: int, num_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 32.0 * np.pi, num_rows, dtype=np.float32)
    base = np.sin(t) + 0.5 * np.sin(0.25 * t)
    data = np.stack([base + 0.1 * rng.standard_normal(num_rows) for _ in range(num_features)], axis=1)
    return data.astype(np.float32)


def _compute_geometry_components(
    raw_train_data: torch.Tensor,
    n_synthetic: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    windows = raw_train_data.unfold(0, n_synthetic, 1).transpose(1, 2)
    flat_windows = windows.reshape(windows.shape[0], -1)
    mean_window = flat_windows.mean(dim=0, keepdim=True)
    distances = torch.cdist(flat_windows, mean_window).squeeze()
    best_idx = torch.argmin(distances).item()
    best_seq = raw_train_data[best_idx : best_idx + n_synthetic].clone()
    return mean_window.squeeze(0), best_seq, best_idx


def _per_step_mean(seq_2d: torch.Tensor) -> torch.Tensor:
    return seq_2d.mean(dim=1)


def _volatility(series_1d: torch.Tensor) -> float:
    diffs = torch.abs(series_1d[1:] - series_1d[:-1])
    return float(diffs.mean().item())


def _plot_diagnostics(
    raw_train_data: torch.Tensor,
    geometry_seq: torch.Tensor,
    geometry_noisy_seq: torch.Tensor | None,
    geometry_start: int,
    random_windows_seed_42: torch.Tensor,
    random_starts_seed_42: torch.Tensor,
    random_windows_seed_500: torch.Tensor,
    random_starts_seed_500: torch.Tensor,
    output_path: Path,
    dataset_label: str,
) -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid")

    full_series = _per_step_mean(raw_train_data).cpu().numpy()
    geometry_steps = _per_step_mean(geometry_seq).cpu().numpy()
    geometry_noisy_steps = None
    if geometry_noisy_seq is not None:
        geometry_noisy_steps = _per_step_mean(geometry_noisy_seq).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(full_series, label="Full training series", color="#1f77b4", linewidth=1.5)

    geometry_end = geometry_start + len(geometry_steps)
    ax.axvspan(geometry_start, geometry_end, color="#ff7f0e", alpha=0.25, label="Geometry window")
    ax.plot(
        range(geometry_start, geometry_end),
        geometry_steps,
        color="#ff7f0e",
        linewidth=2,
    )

    if geometry_noisy_steps is not None:
        ax.plot(
            range(geometry_start, geometry_end),
            geometry_noisy_steps,
            color="#9467bd",
            linewidth=2,
            alpha=0.9,
            label="Geometry window + noise",
        )

    for start, window in zip(random_starts_seed_42.tolist(), random_windows_seed_42):
        window_steps = _per_step_mean(window).cpu().numpy()
        end = start + len(window_steps)
        ax.axvspan(start, end, color="#2ca02c", alpha=0.15)
        ax.plot(
            range(start, end),
            window_steps,
            color="#2ca02c",
            linewidth=1.5,
            alpha=0.8,
            label="Random window (seed 42)",
        )

    for start, window in zip(random_starts_seed_500.tolist(), random_windows_seed_500):
        window_steps = _per_step_mean(window).cpu().numpy()
        end = start + len(window_steps)
        ax.axvspan(start, end, color="#d62728", alpha=0.12)
        ax.plot(
            range(start, end),
            window_steps,
            color="#d62728",
            linewidth=1.5,
            alpha=0.8,
            label="Random window (seed 500)",
        )

    ax.set_title(f"Full Series with Geometry/Random Window Overlays ({dataset_label})")
    ax.set_xlabel("Timestep index")
    ax.set_ylabel("Mean across features")
    ax.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry vs random initializer diagnostics")
    parser.add_argument("--dataset", default="ETTm1", help="Dataset key in DEFAULT_DATASET_CONFIG")
    parser.add_argument("--csv-path", default=None, help="Override CSV path")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--n-synthetic", type=int, default=384)
    parser.add_argument("--random-windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry-noise-std", type=float, default=0.08)
    parser.add_argument("--use-mock", action="store_true", help="Force synthetic mock data")
    parser.add_argument("--mock-rows", type=int, default=5000)
    parser.add_argument("--mock-features", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_label = args.dataset
    if args.use_mock:
        raw_train_np = _make_mock_data(args.mock_rows, args.mock_features, args.seed)
        source_label = "mock"
    else:
        try:
            raw_train_np, source_label = _load_raw_train_data(
                dataset_name=args.dataset,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                override_csv=args.csv_path,
            )
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            print("Falling back to mock data.")
            raw_train_np = _make_mock_data(args.mock_rows, args.mock_features, args.seed)
            source_label = "mock"

    raw_train_data = torch.tensor(raw_train_np, dtype=torch.float32)

    initializer = GeometrySequenceInitializer()
    geometry_seq = initializer.initialize_sequence(
        raw_train_data,
        args.n_synthetic,
        noise_std=0.0,
    ).detach()

    geometry_noisy_seq = None
    if args.geometry_noise_std > 0:
        geometry_noisy_seq = initializer.initialize_sequence(
            raw_train_data,
            args.n_synthetic,
            noise_std=args.geometry_noise_std,
        ).detach()

    mean_window_flat, computed_geo_seq, best_idx = _compute_geometry_components(
        raw_train_data, args.n_synthetic
    )
    mismatch = torch.max(torch.abs(geometry_seq - computed_geo_seq)).item()
    if mismatch > 1e-6:
        print(f"Warning: geometry sequence mismatch (max abs diff {mismatch:.6f}).")

    max_start = raw_train_data.shape[0] - args.n_synthetic
    if max_start <= 0:
        raise ValueError("Not enough rows to sample n_synthetic windows.")

    num_random = 1

    generator_42 = torch.Generator().manual_seed(42)
    random_starts_seed_42 = torch.randint(0, max_start + 1, (num_random,), generator=generator_42)
    random_windows_seed_42 = torch.stack(
        [raw_train_data[start : start + args.n_synthetic].clone() for start in random_starts_seed_42]
    )

    generator_500 = torch.Generator().manual_seed(500)
    random_starts_seed_500 = torch.randint(0, max_start + 1, (num_random,), generator=generator_500)
    random_windows_seed_500 = torch.stack(
        [raw_train_data[start : start + args.n_synthetic].clone() for start in random_starts_seed_500]
    )

    geometry_steps = _per_step_mean(geometry_seq)
    geo_vol = _volatility(geometry_steps)

    rand_vols_42 = []
    for window in random_windows_seed_42:
        rand_vols_42.append(_volatility(_per_step_mean(window)))

    rand_vols_500 = []
    for window in random_windows_seed_500:
        rand_vols_500.append(_volatility(_per_step_mean(window)))

    rand_mean_42 = float(np.mean(rand_vols_42))
    rand_mean_500 = float(np.mean(rand_vols_500))
    ratio_42 = geo_vol / rand_mean_42 if rand_mean_42 > 0 else float("inf")
    ratio_500 = geo_vol / rand_mean_500 if rand_mean_500 > 0 else float("inf")

    print("\nInitializer diagnostics summary")
    print("=" * 36)
    print(f"Data source: {source_label}")
    print(f"Geometry best index: {best_idx}")
    print(f"Geometry volatility: {geo_vol:.6f}")
    print(f"Random volatility mean (seed 42): {rand_mean_42:.6f}")
    print(f"Random volatility mean (seed 500): {rand_mean_500:.6f}")
    print(f"Volatility ratio (geometry/random, seed 42): {ratio_42:.4f}")
    print(f"Volatility ratio (geometry/random, seed 500): {ratio_500:.4f}")

    if ratio_42 < 0.5 or ratio_500 < 0.5:
        print("WARNING: Visual/Statistical proof of Mean/Score Collapse detected.")

    output_path = Path("outputs") / "diagnostics" / "initializer_analysis.png"
    _plot_diagnostics(
        raw_train_data=raw_train_data,
        geometry_seq=geometry_seq,
        geometry_noisy_seq=geometry_noisy_seq,
        geometry_start=best_idx,
        random_windows_seed_42=random_windows_seed_42,
        random_starts_seed_42=random_starts_seed_42,
        random_windows_seed_500=random_windows_seed_500,
        random_starts_seed_500=random_starts_seed_500,
        output_path=output_path,
        dataset_label=dataset_label,
    )

    print(f"\nSaved plot to: {output_path}")


if __name__ == "__main__":
    main()
