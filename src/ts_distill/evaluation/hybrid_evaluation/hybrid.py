import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from ts_distill.data_pipeline.splitter import make_windows
from ts_distill.evaluation.base import BaseEvaluator
from ts_distill.evaluation.sample_selector.base import BaseAnchorSelector
from ts_distill.evaluation.evaluation import Evaluator


class BaseHybridEvaluator(BaseEvaluator):
    """
    Evaluates forecasting models trained on a mix of real and synthetic
    (distilled) data, sweeping the real/synthetic ratio to measure how
    much real data is needed before performance matches/exceeds a
    synthetic-only baseline.
    """

    def __init__(
        self,
        anchor_selector: BaseAnchorSelector,
        seq_len: int,
        batch_size: int = 64,
        device='cpu',
        eval_max_epochs: int = 100,
        early_stop_patience: int = 10,
        eval_lr: float = 1e-3,
    ):
        super().__init__()
 
        self.selector = anchor_selector
 
        self.seq_len = seq_len
 
        self.batch_size = batch_size
 
        self.device = device
 
        self.eval_max_epochs = eval_max_epochs
        self.early_stop_patience = early_stop_patience
        self.eval_lr = eval_lr
 
    def test_on_real(self, model, test_data):
        # Accept either a raw tensor or a DataLoader; flatten the loader
        # back into a single tensor since Evaluator expects one.
        if isinstance(test_data, DataLoader):
            batches = []

            for batch in test_data:
                if isinstance(batch, (list, tuple)):
                    batches.append(batch[0])
                else:
                    batches.append(batch)

            test_data = torch.cat(batches, dim=0)

        evaluator = Evaluator(
            seq_len=self.seq_len,
            batch_size=self.batch_size,
        )

        return evaluator.test_on_real(model, test_data)
 
    def hybridmixture(
        self,
        synthetic_data,
        real_train_data,
        real_ratio: float,
        window_size: int,
        expert_losses=None,
    ):
        # Build the pool of candidate real windows to draw from.
        real_window_candidates = torch.tensor(
            make_windows(real_train_data.cpu().numpy(), window_size),
            dtype=torch.float32,
        )

        total_possible_windows = len(real_window_candidates)
        if real_ratio <= 0.0:
            # Synthetic-only training set.
            n_real_windows = 0
            real_windows = torch.empty((0, *real_window_candidates.shape[1:]), dtype=torch.float32)
        elif real_ratio >= 1.0:
            # Real-only training set (no synthetic mixing).
            n_real_windows = total_possible_windows
            real_windows = real_window_candidates.float()
        else:
            # Pick which real windows to include according to the
            # configured anchor selector (random / diversity / importance /
            # etc.) rather than always taking a fixed prefix.
            n_real_windows = max(1, int(total_possible_windows * real_ratio))

            if self.selector is not None:
                selected_indices = self.selector.select_indices(
                    real_window_candidates,
                    n_real_windows,
                    expert_losses=expert_losses,
                )
                real_windows = real_window_candidates[selected_indices].float()
            else:
                real_windows = real_window_candidates[:n_real_windows].float()

        synth_windows = torch.tensor(
            make_windows(
                synthetic_data.cpu().numpy(),
                window_size
            ),
            dtype=torch.float32,
        )

        # Combine the chosen real windows with all synthetic windows and
        # shuffle so batches are not ordered real-then-synthetic.
        combined = torch.cat(
            [real_windows, synth_windows],
            dim=0
        )

        perm = torch.randperm(len(combined))
        combined = combined[perm]

        return TensorDataset(combined)
 
    def _train_and_test(
        self,
        train_loader,
        test_loader,
        model_fn,
        val_loader=None,
    ):
        # Fresh model per call so each mixing ratio/seed is trained from
        # scratch and results are comparable.
        model = model_fn().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.eval_lr,
        )
        criterion = torch.nn.MSELoss()
 
        seq_len = self.seq_len
 
        def _split_batch(batch):
            # Each window is [seq_len input steps][forecast horizon steps];
            # split it into (x, y) for supervised training.
            data = batch[0].to(self.device)
            if data.dim() == 2:
                data = data.unsqueeze(-1)
            x = data[:, :seq_len, :]
            y = data[:, seq_len:, :]
            return x, y

        if val_loader is None:
            # No validation set: just train for a fixed, short number of
            # epochs (no early stopping possible without a val signal).
            model.train()
            for epoch in range(5):
                for batch in train_loader:
                    x, y = _split_batch(batch)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
        else:
            # Train with early stopping: keep the best-val-loss weights
            # seen so far and stop once patience is exhausted.
            best_val_loss = float('inf')
            best_state = None
            patience_counter = 0

            for epoch in range(self.eval_max_epochs):
                model.train()
                for batch in train_loader:
                    x, y = _split_batch(batch)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
 
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = _split_batch(batch)
                        output = model(x)
                        val_losses.append(criterion(output, y).item())
 
                val_loss = (
                    sum(val_losses) / len(val_losses)
                    if val_losses else float('inf')
                )
 
                if val_loss < best_val_loss - 1e-8:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.detach().clone()
                        for k, v in model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        break

            # Roll back to the best checkpoint rather than using the
            # weights from the final (possibly worse) epoch.
            if best_state is not None:
                model.load_state_dict(best_state)

        score = self.test_on_real(model, test_loader)
        return score
 
    def evaluate_mixing(
        self,
        synthetic_data,
        real_train_data,
        real_test_loader,
        model_fn,
        window_size,
        mixing_ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        expert_losses=None,
        real_val_data=None,
        seeds=None,
    ):
        """
        Sweep over `mixing_ratios`, training a fresh model at each real/
        synthetic ratio (optionally averaged over multiple seeds) and
        collecting the resulting real-data MSE, keyed as "hybrid_<pct>".
        """
        results = {}

        val_loader = None
        if real_val_data is not None:
            val_windows = torch.tensor(
                make_windows(real_val_data.cpu().numpy(), window_size),
                dtype=torch.float32,
            )
            val_loader = DataLoader(
                TensorDataset(val_windows),
                batch_size=self.batch_size,
                shuffle=False,
            )

        seed_list = list(seeds) if seeds else [None]

        for ratio in mixing_ratios:
            seed_mse = {}
            last_score = None

            for seed in seed_list:
                if seed is not None:
                    torch.manual_seed(seed)

                hybrid_dataset = self.hybridmixture(
                    synthetic_data=synthetic_data,
                    real_train_data=real_train_data,
                    real_ratio=ratio,
                    window_size=window_size,
                    expert_losses=expert_losses,
                )

                train_loader = DataLoader(
                    hybrid_dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )

                score = self._train_and_test(
                    train_loader,
                    real_test_loader,
                    model_fn,
                    val_loader=val_loader,
                )
                last_score = score
                seed_mse[seed if seed is not None else 0] = score['MSE']

            if seeds:
                # Multiple seeds: report mean/std across seeds so noise
                # from random init/shuffling doesn't look like a real effect.
                mse_values = list(seed_mse.values())
                results[f"hybrid_{int(ratio * 100)}"] = {
                    'MSE':      float(np.mean(mse_values)),
                    'MSE_std':  float(np.std(mse_values)),
                    'seed_mse': seed_mse,
                }
            else:
                results[f"hybrid_{int(ratio * 100)}"] = last_score

        return results
 
    def analyse_h2_cell(
        self,
        strategy_results: dict,
        dataset: str,
        model: str,
        epsilon: float = 0.01,
    ) -> dict:
        """
        Compare multiple anchor-selection strategies' `evaluate_mixing`
        results for one (dataset, model) cell to test H2: "the choice of
        real-window selection strategy meaningfully affects hybrid-mixing
        performance". Only ratio keys present for every strategy are used
        so the comparison is apples-to-apples.
        """
        strategies = list(strategy_results.keys())
        if len(strategies) < 2:
            raise ValueError(
                f"analyse_h2_cell requires at least 2 strategies to compare "
                f"for {dataset} x {model}; got {len(strategies)}."
            )
 
        common_ratio_keys = set(strategy_results[strategies[0]].keys())
        for s in strategies[1:]:
            common_ratio_keys &= set(strategy_results[s].keys())
 
        if not common_ratio_keys:
            raise ValueError(
                f"No ratio keys are common across strategies {strategies} "
                f"for {dataset} x {model}."
            )
 
        common_ratio_keys = sorted(
            common_ratio_keys,
            key=lambda k: int(k.replace('hybrid_', ''))
        )
 
        # For each ratio, rank strategies best-to-worst by MSE and record
        # how far apart the best and worst strategy are (the "spread").
        per_ratio_ranking = {}
        spreads = {}
        for ratio_key in common_ratio_keys:
            mses = {s: strategy_results[s][ratio_key]['MSE'] for s in strategies}
            per_ratio_ranking[ratio_key] = sorted(mses, key=mses.get)
            spreads[ratio_key] = max(mses.values()) - min(mses.values())

        # The ratio with the largest gap between strategies is the
        # strongest evidence (for or against) that strategy choice matters.
        best_ratio_key = max(spreads, key=spreads.get)
        strategy_delta = spreads[best_ratio_key]

        # Overall strategy ranking, averaged across all common ratios.
        mean_mse = {
            s: float(np.mean([strategy_results[s][k]['MSE'] for k in common_ratio_keys]))
            for s in strategies
        }
        strategy_ranking = sorted(mean_mse, key=mean_mse.get)
        best_strategy = strategy_ranking[0]
        worst_strategy = strategy_ranking[-1]

        # Whether the same strategy ranking holds at every ratio (a
        # consistent ranking is stronger evidence than one that flips).
        first_ranking = per_ratio_ranking[common_ratio_keys[0]]
        consistent_ranking = all(
            per_ratio_ranking[k] == first_ranking for k in common_ratio_keys
        )

        # H2 is "supported" if the largest observed strategy gap exceeds
        # the noise threshold epsilon.
        h2_supported = strategy_delta > epsilon
 
        return {
            'strategy_ranking': strategy_ranking,
            'best_strategy': best_strategy,
            'worst_strategy': worst_strategy,
            'mean_mse': mean_mse,
            'strategy_delta': strategy_delta,
            'best_ratio_key': best_ratio_key,
            'consistent_ranking': consistent_ranking,
            'h2_supported': h2_supported,
        }
 
    def plot_hybrid_results(self, results):
        """Plot MSE vs. real-data percentage from an `evaluate_mixing` result dict."""
        ratios = []
        scores = []

        for key, value in results.items():
            ratio = int(key.split("_")[1])
            ratios.append(ratio)
            scores.append(value['MSE'])

        # Results dict iteration order isn't guaranteed to be sorted by ratio.
        ratios, scores = zip(*sorted(zip(ratios, scores)))
 
        plt.figure()
        plt.plot(ratios, scores, marker='o')
 
        plt.xlabel("Real Data Percentage (%)")
        plt.ylabel("MSE")
        plt.title("Hybrid Data Mixing Performance")
 
        plt.grid()
        plt.show()
 
 
def compute_expert_losses(
    expert_model,
    raw_train_data,
    window_size: int,
    seq_len: int,
    batch_size: int = 64,
    device='cpu',
) -> torch.Tensor:
    """
    Compute per-window reconstruction/forecast loss for `expert_model` on
    each real window. Used by ImportanceWeightedSelector to prioritize
    windows the expert finds hardest (highest loss).
    """
    windows = torch.tensor(
        make_windows(raw_train_data.cpu().numpy(), window_size),
        dtype=torch.float32,
    )
    dataset = TensorDataset(windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
    expert_model = expert_model.to(device)
    expert_model.eval()
 
    losses = []
    criterion = torch.nn.MSELoss(reduction='none')
 
    with torch.no_grad():
        for (batch,) in loader:
            data = batch.to(device)
            x = data[:, :seq_len, :]
            y = data[:, seq_len:, :]
            output = expert_model(x)
            window_losses = criterion(output, y).mean(dim=(1, 2, 3))
            losses.append(window_losses.detach().cpu())
 
    if not losses:
        return torch.empty(0, dtype=torch.float32)
 
    return torch.cat(losses, dim=0)
 
 
class RandomAnchorSelector(BaseAnchorSelector):
    """Baseline: pick `n_samples` real windows uniformly at random."""

    def __init__(self, selector_seed=None):
        self.selector_seed = selector_seed

    def select_indices(self, data, n_samples: int, expert_losses=None):
        if self.selector_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.selector_seed)
            return torch.randperm(len(data), generator=generator)[:n_samples]
        return torch.randperm(len(data))[:n_samples]
 
class StartExtendSelector(BaseAnchorSelector):
    """Take the first `n_samples` windows in chronological order (a contiguous prefix)."""

    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        total = len(data)
        n     = min(n_samples, total)
        return torch.arange(n)

class UniformStrideSelector(BaseAnchorSelector):
    """Take every `stride`-th window so the sample spans the whole time range evenly."""

    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        total   = len(data)
        n       = min(n_samples, total)
        stride  = max(1, total // n)
        indices = torch.arange(0, total, stride)[:n]
        return indices

class ImportanceWeightedSelector(BaseAnchorSelector):
    """Pick the windows the expert model finds hardest (highest per-window loss)."""

    def __init__(self, selector_seed=None):
        self.selector_seed = selector_seed

    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        if expert_losses is None:
            # No losses supplied (e.g. no expert model available): fall
            # back to random selection rather than failing.
            import warnings
            warnings.warn(
                RuntimeWarning,
                stacklevel=2,
            )
            if self.selector_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(self.selector_seed)
                return torch.randperm(len(data), generator=generator)[:n_samples]
            return torch.randperm(len(data))[:n_samples]

        n              = min(n_samples, len(expert_losses))
        sorted_indices = torch.argsort(expert_losses, descending=True)
        return sorted_indices[:n]

class DiversityAnchorSelector(BaseAnchorSelector):
    """Greedy farthest-point sampling: pick windows that are maximally spread out in feature space."""

    def __init__(self, selector_seed=None):
        self.selector_seed = selector_seed

    def _extract_features(self, data: torch.Tensor) -> np.ndarray:
        # Summarize each window as (mean, std, min, max, linear trend
        # slope) per channel, then z-score each feature column so no
        # single statistic dominates the distance calculation below.
        mean = data.mean(dim=1)
        std  = data.std(dim=1)
        minv = data.min(dim=1).values 
        maxv = data.max(dim=1).values
 
        T  = data.size(1)
        t  = torch.arange(T, device=data.device).float() 
        t_mean = t.mean()
        t_var  = ((t - t_mean) ** 2).sum()
        slope  = ((data - data.mean(dim=1, keepdim=True)) *
                  (t - t_mean).view(1, -1, 1)).sum(dim=1) / (t_var + 1e-8)
 
        features = torch.cat([mean, std, minv, maxv, slope], dim=-1)
        feat_np  = features.detach().cpu().numpy()
 
        col_std  = feat_np.std(axis=0) + 1e-8
        col_mean = feat_np.mean(axis=0)
        return (feat_np - col_mean) / col_std
 
    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        # Greedy farthest-point sampling: start from window 0, then
        # repeatedly add whichever remaining window is farthest (in
        # feature space) from every window already selected. This
        # maximizes coverage/diversity instead of picking similar windows.
        N = len(data)
        n = min(n_samples, N)

        features = self._extract_features(data)

        selected  = [0]
        min_dists = np.full(N, np.inf)

        for _ in range(n - 1):
            last   = features[selected[-1]]
            dists  = np.linalg.norm(features - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -np.inf  # never re-pick an already-selected window
            selected.append(int(np.argmax(min_dists)))

        return torch.tensor(selected[:n], dtype=torch.long)

