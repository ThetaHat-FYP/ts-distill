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
        device='cpu'
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
        window_size: int
    ):
        """
        Creates a hybrid dataset by mixing:
        - synthetic windows
        - sampled real windows

        real_ratio determines percentage of real data in mixture.

        Example:
            real_ratio = 0.3
            -> 30% real windows
            -> 70% synthetic windows
        """

        # Total length of real time series
        T = len(real_train_data)

        # Total possible sliding windows from real data
        total_possible_windows = T - window_size + 1

        # Number of real windows to sample
        n_real_windows = max(1, int(total_possible_windows * real_ratio))

        # Spread sampling across full time series evenly
        step = max(1, total_possible_windows // n_real_windows)

        # Base starting indices
        base_starts = torch.arange(
            0,
            total_possible_windows,
            step
        )[:n_real_windows]

        # Add random jitter to avoid overly regular sampling
        jitter = torch.randint(
            0,
            max(1, step // 2),
            (len(base_starts),)
        )

        # Final start positions (clamped within bounds)
        starts = (base_starts + jitter).clamp(
            0,
            total_possible_windows - 1
        )

        # Extract real windows using sampled indices
        real_windows = torch.stack([
            real_train_data[s:s + window_size]
            for s in starts
        ]).float()

        # Convert synthetic series into sliding windows
        synth_windows = torch.tensor(
            make_windows(
                synthetic_data.cpu().numpy(),
                window_size
            ),
            dtype=torch.float32,
        )

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
        model_fn
    ):
        """
        Trains a fresh model on hybrid data and evaluates it
        on real test data.
        """

        # Create new model instance
        model = model_fn().to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )
        criterion = torch.nn.MSELoss()

        seq_len = self.seq_len

        model.train()

        # Train for fixed epochs
        for epoch in range(5):
            for batch in train_loader:
                data = batch[0].to(self.device)

                # Ensure shape = [batch, time, features]
                if data.dim() == 2:
                    data = data.unsqueeze(-1)

                # Split into input and target
                x = data[:, :seq_len, :]
                y = data[:, seq_len:, :]

                optimizer.zero_grad()

                # Forward pass
                output = model(x)

                # Forecasting loss
                loss = criterion(output, y)

                # Optional weighted loss example:
                # loss = real_weight * loss_real + synth_weight * loss_synth

                loss.backward()
                optimizer.step()

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
        mixing_ratios=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ):
        """
        Runs experiments for different real/synthetic mixing ratios.

        Returns:
            {
                'hybrid_10': {...},
                'hybrid_20': {...},
                ...
            }
        """

        results = {}

        for ratio in mixing_ratios:
            # Build hybrid dataset for current ratio
            hybrid_dataset = self.hybridmixture(
                synthetic_data=synthetic_data,
                real_train_data=real_train_data,
                real_ratio=ratio,
                window_size=window_size,
            )

            # Training loader
            train_loader = DataLoader(
                hybrid_dataset,
                batch_size=64,
                shuffle=True
            )

            # Train + evaluate
            score = self._train_and_test(
                train_loader,
                real_test_loader,
                model_fn
            )

            # Save results
            results[f"hybrid_{int(ratio * 100)}"] = score

        return results

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


class RandomAnchorSelector(BaseAnchorSelector):
    """
    Simple anchor selector that randomly picks samples.

    Useful as a baseline anchor selection strategy.
    """

    def select_indices(self, data, n_samples: int):
        """
        Select n_samples random indices from data.

        Returns:
            Tensor of random indices
        """
        return torch.randperm(len(data))[:n_samples]