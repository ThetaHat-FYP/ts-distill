"""
Hybrid data-mixing evaluation: trains models on real/synthetic data mixtures
at different real-data ratios (BaseHybridEvaluator) and compares anchor
selection strategies for picking which real windows go into that mixture
(RandomAnchorSelector, StartExtendSelector, UniformStrideSelector,
ImportanceWeightedSelector, DiversityAnchorSelector).
"""

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
    Evaluates model performance when training on a mixture of:
    - synthetic data
    - a percentage of real training data
 
    The goal is to measure whether synthetic data improves learning
    when combined with limited real data.
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
 
        # Strategy used for selecting anchor samples
        self.selector = anchor_selector
 
        # Input sequence length for forecasting
        self.seq_len = seq_len
 
        # Batch size for training/evaluation
        self.batch_size = batch_size
 
        # Training device ('cpu' or 'cuda')
        self.device = device
 
        # Evaluation hyperparameters — now actually used in _train_and_test
        # (FIX 1) whenever real_val_data is supplied to evaluate_mixing().
        self.eval_max_epochs = eval_max_epochs
        self.early_stop_patience = early_stop_patience
        self.eval_lr = eval_lr
 
    def test_on_real(self, model, test_data):
        """
        Required abstract method from BaseEvaluator.
 
        Evaluates the trained model on real test data using
        the standard Evaluator class.
        """
 
        # If test data is already a DataLoader, convert it back
        # into a single tensor
        if isinstance(test_data, DataLoader):
            batches = []
 
            for batch in test_data:
                if isinstance(batch, (list, tuple)):
                    batches.append(batch[0])
                else:
                    batches.append(batch)
 
            test_data = torch.cat(batches, dim=0)
 
        # Standard evaluator for real-data testing
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
        """
        Build a hybrid training set: every synthetic window, plus real windows
        drawn from the training corpus.

        What real_ratio actually controls
        ---------------------------------
        `real_ratio` is a fraction of the FULL real corpus, not the composition
        of the returned dataset:

            n_real  = int(total_real_windows * real_ratio)
            n_synth = ALL windows of synthetic_data   (never subsampled)

        Because the real corpus is typically far larger than the short distilled
        sequence, the result is much more real-heavy than `real_ratio` suggests.

        Example (ETTm2: 34369 real windows, a 384-step synthetic -> 193 windows):

            real_ratio = 0.2
            -> n_real  = 6873
            -> n_synth = 193
            -> 7066 windows total, which is 97.3% real BY COUNT

        So do not report `real_ratio` as the mixture composition. Derive the
        actual split from the returned length:

            n_synth = len(make_windows(synthetic_data, window_size))
            n_real  = len(dataset) - n_synth

        Args:
            synthetic_data (Tensor):  Distilled sequence, shape (M, C).
            real_train_data (Tensor): Un-windowed real training data, shape (T, C).
            real_ratio (float):       Fraction of the real corpus to draw, [0, 1].
                                      0.0 -> pure synthetic; 1.0 -> pure real.
            window_size (int):        seq_len + pred_len.
            expert_losses:            Optional per-window losses, forwarded to the
                                      anchor selector for importance weighting.

        Returns:
            TensorDataset: Shuffled hybrid windows.
        """

        real_window_candidates = torch.tensor(
            make_windows(real_train_data.cpu().numpy(), window_size),
            dtype=torch.float32,
        )
 
        total_possible_windows = len(real_window_candidates)
        if real_ratio <= 0.0:
            n_real_windows = 0
            real_windows = torch.empty((0, *real_window_candidates.shape[1:]), dtype=torch.float32)
        elif real_ratio >= 1.0:
            n_real_windows = total_possible_windows
            real_windows = real_window_candidates.float()
        else:
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
 
        # Convert synthetic series into sliding windows
        synth_windows = torch.tensor(
            make_windows(
                synthetic_data.cpu().numpy(),
                window_size
            ),
            dtype=torch.float32,
        )
 
        if real_ratio <= 0.0:
            n_synth = len(synth_windows)
        elif real_ratio >= 1.0:
            n_synth = 0
        else:
            # Compute number of synthetic windows needed to match ratio
            n_synth = int(
                n_real_windows * (1 - real_ratio) / real_ratio
            )
 
        # Prevent requesting more synthetic windows than available
        n_synth = min(n_synth, len(synth_windows))
 
        # Keep only required synthetic windows
        synth_windows = synth_windows[:n_synth]
 
        # Combine real + synthetic windows
        combined = torch.cat(
            [real_windows, synth_windows],
            dim=0
        )
 
        # Shuffle combined dataset
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
        """
        Trains a fresh model on hybrid data and evaluates it
        on real test data.
 
        FIX 1: when val_loader is provided, trains for up to
        self.eval_max_epochs with early stopping on val loss
        (patience = self.early_stop_patience) and restores the
        best-val-loss weights before testing. When val_loader is
        None (no real_val_data was passed to evaluate_mixing), falls
        back to the original fixed-5-epoch behaviour for backward
        compatibility with callers that don't supply validation data.
        """
 
        # Create new model instance
        model = model_fn().to(self.device)
 
        # Optimizer and loss — now uses self.eval_lr (FIX 1) instead of a
        # hardcoded 1e-3.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.eval_lr,
        )
        criterion = torch.nn.MSELoss()
 
        seq_len = self.seq_len
 
        def _split_batch(batch):
            """Split a window into (input, target) at seq_len."""
            data = batch[0].to(self.device)
            if data.dim() == 2:
                data = data.unsqueeze(-1)
            x = data[:, :seq_len, :]
            y = data[:, seq_len:, :]
            return x, y
 
        if val_loader is None:
            # ── Backward-compatible path: no validation data supplied ──
            # Original fixed-epoch behaviour, preserved so callers that
            # don't pass real_val_data (e.g. experiment_matrix.py) keep
            # working exactly as before.
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
            # ── Real early-stopping path (FIX 1) ────────────────────────
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
 
                # Validation pass
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
 
            if best_state is not None:
                model.load_state_dict(best_state)
 
        # Evaluate trained model on real test data
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
        Runs experiments for different real/synthetic mixing ratios.

        Parameters
        ----------
        real_val_data : optional real (continuous, unwindowed) validation
            series, same format as real_train_data. When provided (FIX 1),
            it is windowed the same way as real_train_data and used for
            early stopping in each ratio's training run. When omitted,
            training falls back to a fixed 5 epochs (original behaviour).
        seeds : optional list/tuple of ints. When provided, each ratio is
            repeated once per seed — `torch.manual_seed(seed)` is set before
            re-drawing the hybrid mixture (real-window selection + shuffle
            order) and before re-initializing + training the model — so
            every seed gets an independent run. The per-ratio result then
            reports the mean/std MSE across seeds plus the raw per-seed
            values. When omitted (default), behaviour is unchanged: one run
            per ratio using whatever the ambient RNG state is.

        Returns:
            {
                'hybrid_10': {'MSE': ..., 'RMSE': ...},               # seeds=None
                'hybrid_20': {'MSE': mean, 'MSE_std': std,
                              'seed_mse': {7: ..., 42: ..., 123: ...}}, # seeds given
                ...
            }
        """

        results = {}

        # Build the (windowed) validation loader once, shared across all
        # ratios in the sweep — mirrors how real_test_loader is shared.
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

                # Build hybrid dataset for current ratio (re-drawn per seed
                # so real-window selection and shuffle order vary too).
                hybrid_dataset = self.hybridmixture(
                    synthetic_data=synthetic_data,
                    real_train_data=real_train_data,
                    real_ratio=ratio,
                    window_size=window_size,
                    expert_losses=expert_losses,
                )

                # Training loader — FIX 2: uses self.batch_size instead of a
                # hardcoded 64.
                train_loader = DataLoader(
                    hybrid_dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )

                # Train + evaluate
                score = self._train_and_test(
                    train_loader,
                    real_test_loader,
                    model_fn,
                    val_loader=val_loader,
                )
                last_score = score
                seed_mse[seed if seed is not None else 0] = score['MSE']

            if seeds:
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
        FIX 3: Compare MSE profiles across selection strategies for one
        (dataset, model) cell — supports the H2/H3-style "does selection
        strategy matter" analysis used by h2_strategy_sweep.py.
 
        Parameters
        ----------
        strategy_results : dict
            {strategy_name: {ratio_key: {'MSE': float}, ...}, ...}
            e.g. {'S1': {'hybrid_5': {'MSE': 0.41}, 'hybrid_10': {...}},
                  'S2': {'hybrid_5': {'MSE': 0.43}, 'hybrid_10': {...}}}
        dataset, model : str
            Identifying labels only — not used in the computation, kept
            for symmetry with the calling code and easier debugging.
        epsilon : float
            Minimum MSE spread at the most-discriminating ratio required
            to call the strategy choice meaningful for this cell.
 
        Returns
        -------
        dict with keys:
            strategy_ranking   — list of strategy names, best (lowest mean
                                  MSE across common ratios) first
            best_strategy       — strategy_ranking[0]
            worst_strategy       — strategy_ranking[-1]
            mean_mse               — {strategy: mean MSE across common ratios}
            strategy_delta           — MSE spread (max - min across strategies)
                                        at whichever ratio shows the largest gap
            best_ratio_key             — the ratio key where that gap occurs
            consistent_ranking           — True if the best→worst strategy
                                            ordering is identical at every
                                            common ratio, not just on average
            h2_supported                   — True if strategy_delta > epsilon
        """
        strategies = list(strategy_results.keys())
        if len(strategies) < 2:
            raise ValueError(
                f"analyse_h2_cell requires at least 2 strategies to compare "
                f"for {dataset} x {model}; got {len(strategies)}."
            )
 
        # Only compare ratios present for every strategy being compared.
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
 
        # Per-ratio strategy ranking + spread
        per_ratio_ranking = {}
        spreads = {}
        for ratio_key in common_ratio_keys:
            mses = {s: strategy_results[s][ratio_key]['MSE'] for s in strategies}
            per_ratio_ranking[ratio_key] = sorted(mses, key=mses.get)  # best first
            spreads[ratio_key] = max(mses.values()) - min(mses.values())
 
        # The ratio where strategies diverge the most is the most
        # informative one for deciding whether strategy choice matters.
        best_ratio_key = max(spreads, key=spreads.get)
        strategy_delta = spreads[best_ratio_key]
 
        # Overall ranking by mean MSE across all common ratios
        mean_mse = {
            s: float(np.mean([strategy_results[s][k]['MSE'] for k in common_ratio_keys]))
            for s in strategies
        }
        strategy_ranking = sorted(mean_mse, key=mean_mse.get)
        best_strategy = strategy_ranking[0]
        worst_strategy = strategy_ranking[-1]
 
        # Consistent ranking = same best→worst order at every common ratio
        first_ranking = per_ratio_ranking[common_ratio_keys[0]]
        consistent_ranking = all(
            per_ratio_ranking[k] == first_ranking for k in common_ratio_keys
        )
 
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
        """
        Plot MSE vs percentage of real data used.
 
        Example:
            x-axis = 10, 20, 30, ...
            y-axis = MSE
        """
 
        ratios = []
        scores = []
 
        for key, value in results.items():
            # Extract ratio from key like "hybrid_30"
            ratio = int(key.split("_")[1])
            ratios.append(ratio)
 
            # Value expected format:
            # {'MSE': ...}
            scores.append(value['MSE'])
 
        # Sort by ratio for clean plotting
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
    """Compute per-window MSE losses for an expert model over the training series."""
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
    """
    Simple anchor selector that randomly picks samples.
 
    Useful as a baseline anchor selection strategy.
    """
 
    def __init__(self, selector_seed=None):
        self.selector_seed = selector_seed
 
    def select_indices(self, data, n_samples: int, expert_losses=None):
        """
        Select n_samples random indices from data.
 
        Returns:
            Tensor of random indices
        """
        if self.selector_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.selector_seed)
            return torch.randperm(len(data), generator=generator)[:n_samples]
        return torch.randperm(len(data))[:n_samples]
 
class StartExtendSelector(BaseAnchorSelector):
    """
    S2 — Start-and-extend (chronological selection).
 
    Selects the FIRST n_samples windows in chronological order.
    As the ratio increases, the selected set grows by appending later windows.
 
    Assumption: earlier data is more stable / representative.
 
    Expected weakness: on trending or non-stationary datasets (Exchange Rate),
    early windows reflect a different distributional regime than the test set.
    This strategy may actively hurt on such datasets at higher ratios.
 
    Deterministic — no seed required.
    """
 
    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        """Return first n_samples indices in chronological order."""
        total = len(data)
        n     = min(n_samples, total)
        return torch.arange(n)
 
class UniformStrideSelector(BaseAnchorSelector):
    """
    S3 — Uniform stride (evenly-spaced temporal coverage).
 
    Selects n_samples windows spaced uniformly across the full training period.
    stride = floor(total_windows / n_samples)
    indices = {0, stride, 2×stride, ...}
 
    Assumption: even temporal coverage is more informative than random or
    chronological selection because it samples all temporal phases equally.
 
    Strength: guaranteed coverage of every temporal region; deterministic.
    Weakness: may land in the middle of a trend rather than at its boundaries.
 
    Deterministic — no seed required.
    """
 
    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        """Return n_samples evenly-spaced indices across the data."""
        total   = len(data)
        n       = min(n_samples, total)
        stride  = max(1, total // n)
        indices = torch.arange(0, total, stride)[:n]
        return indices
 
class ImportanceWeightedSelector(BaseAnchorSelector):
    """
    S4 — Importance-weighted selection (highest expert-error windows first).
 
    Selects the n_samples windows where the EXPERT model had the HIGHEST
    prediction error (MSE per window). These are the windows that the expert
    encoded worst into the synthetic sequence — the information most likely
    to be missing from the distilled data.
 
    Assumption: windows with high expert error carry information that was NOT
    preserved during distillation. Adding these to the hybrid training set
    patches the specific gaps left by the synthetic sequence.
 
    Requires expert_losses tensor (shape: total_windows) computed by
    compute_expert_losses() before calling hybridmixture().
 
    If expert_losses is None, falls back to random selection (S1 behaviour)
    with a warning.
    """
 
 
    def __init__(self, selector_seed=None):
        self.selector_seed = selector_seed
 
    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
        """Return n_samples indices with highest expert loss."""
        if expert_losses is None:
            import warnings
            warnings.warn(
                "ImportanceWeightedSelector: expert_losses is None. "
                "Falling back to random selection. "
                "Pass expert_losses=compute_expert_losses(...) to hybridmixture().",
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
    """
    S5 — Diversity-based selection (greedy k-center / farthest-first traversal).
 
    Selects windows that maximally cover the feature space by always choosing
    the next window that is farthest from all already-selected windows.
 
    Strength:
        - Guarantees coverage of all temporal patterns with no redundancy
        - Deterministic and principled (true core-set approximation)
        - Strong baseline for hybrid mixing research
 
    Weakness:
        - O(n * N) compute per ratio step (acceptable for typical window counts)
    """
 
    def __init__(self, selector_seed=None):
        self.selector_seed = selector_seed
 
    def _extract_features(self, data: torch.Tensor) -> np.ndarray:
        """
        Convert each window into a normalized feature vector.
 
        Uses per-channel mean, std, min, max, and linear trend slope.
        All features are standardized so no single statistic dominates
        the distance calculation.
 
        data shape: (N, window_size, features)
        returns:    (N, feature_dim) numpy array, zero-mean unit-variance
        """
        mean = data.mean(dim=1)                  # (N, C)
        std  = data.std(dim=1)                   # (N, C)
        minv = data.min(dim=1).values            # (N, C)
        maxv = data.max(dim=1).values            # (N, C)
 
        # Linear trend slope per channel via least-squares normal equation:
        # slope = (T*sum(t*x) - sum(t)*sum(x)) / (T*sum(t^2) - sum(t)^2)
        T  = data.size(1)
        t  = torch.arange(T, device=data.device).float()           # (T,)
        t_mean = t.mean()
        t_var  = ((t - t_mean) ** 2).sum()
        slope  = ((data - data.mean(dim=1, keepdim=True)) *
                  (t - t_mean).view(1, -1, 1)).sum(dim=1) / (t_var + 1e-8)  # (N, C)
 
        features = torch.cat([mean, std, minv, maxv, slope], dim=-1)  # (N, 5*C)
        feat_np  = features.detach().cpu().numpy()
 
        # Standardize so each feature dimension has zero mean and unit variance
        col_std  = feat_np.std(axis=0) + 1e-8
        col_mean = feat_np.mean(axis=0)
        return (feat_np - col_mean) / col_std
 
    def select_indices(
        self,
        data,
        n_samples: int,
        expert_losses=None,
    ) -> torch.Tensor:
 
        """Choose which real windows to mix in, per this strategy."""
        N = len(data)
        n = min(n_samples, N)
 
        features = self._extract_features(data)   # (N, feature_dim)
 
        # Greedy farthest-first traversal (true k-center approximation):
        # Start with one fixed seed point, then always pick the point that
        # is farthest from ALL already-selected points. This maximises the
        # minimum pairwise distance in the selected set → true diversity.
        selected  = [0]
        # min_dists[i] = distance from point i to its nearest selected point
        min_dists = np.full(N, np.inf)
 
        for _ in range(n - 1):
            last   = features[selected[-1]]
            dists  = np.linalg.norm(features - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -np.inf          # exclude already selected
            selected.append(int(np.argmax(min_dists)))
 
        return torch.tensor(selected[:n], dtype=torch.long)

