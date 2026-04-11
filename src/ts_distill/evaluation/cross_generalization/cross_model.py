from typing import List, Callable, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import numpy as np
import torch

from ts_distill.evaluation.base import BaseEvaluator


class CrossModelEvaluator(BaseEvaluator):
    """
    Cross-model evaluator for synthetic time-series data.

    Evaluates whether synthetic data generalizes across architectures
    (LSTM, TCN, Transformer, etc.) using real test data.
    """

    def __init__(self, seed: int = 42, use_parallel: bool = False):
        super().__init__()
        self.seed = seed
        self.use_parallel = use_parallel
        self._set_seed(seed)

    # ----------------------------
    # Utilities
    # ----------------------------
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ----------------------------
    # Core evaluation
    # ----------------------------
    def evaluate_architectures(
        self,
        synthetic_data,
        real_train_data,
        real_test_loader,
        model_factories: Dict[str, Callable],
        real_ratio: float = 0.1,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate multiple architectures on synthetic + real mixed data.
        """

        print("\n=== Cross-Model Evaluation Started ===")
        print(f"Real ratio: {real_ratio}, Batch size: {batch_size}")

        if self.use_parallel:
            return self._evaluate_parallel(
                synthetic_data,
                real_train_data,
                real_test_loader,
                model_factories,
                real_ratio,
                batch_size
            )

        results = {}

        for arch_name, create_model_fn in model_factories.items():
            print(f"\n[INFO] Evaluating: {arch_name}")

            try:
                model_result = self._evaluate_single_model(
                    arch_name,
                    create_model_fn,
                    synthetic_data,
                    real_train_data,
                    real_test_loader,
                    real_ratio,
                    batch_size
                )
                results[arch_name] = model_result

            except Exception as e:
                print(f"[ERROR] {arch_name} failed: {e}")
                results[arch_name] = {
                    "error": str(e)
                }

        return results

    # ----------------------------
    # Single model evaluation
    # ----------------------------
    def _evaluate_single_model(
        self,
        arch_name: str,
        create_model_fn: Callable,
        synthetic_data,
        real_train_data,
        real_test_loader,
        real_ratio: float,
        batch_size: int
    ) -> Dict[str, Any]:

        # 1. Create model
        model = create_model_fn()

        # 2. Sample real subset
        real_subset = self._sample_real_data(real_train_data, real_ratio)

        # 3. Log dataset composition
        print(f"[{arch_name}] Real samples ratio: {real_ratio}")

        # 4. Merge datasets
        mixed_loader = self._merge_datasets(
            synthetic_data,
            real_subset,
            batch_size
        )

        # 5. Train
        train_metrics = self.train(model, mixed_loader)

        # 6. Evaluate
        test_metrics = self.test_on_real(model, real_test_loader)

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "real_ratio": real_ratio,
            "batch_size": batch_size
        }

    # ----------------------------
    # Parallel evaluation
    # ----------------------------
    def _evaluate_parallel(
        self,
        synthetic_data,
        real_train_data,
        real_test_loader,
        model_factories,
        real_ratio,
        batch_size
    ):

        results = {}

        def run_model(arch_name, fn):
            try:
                return arch_name, self._evaluate_single_model(
                    arch_name,
                    fn,
                    synthetic_data,
                    real_train_data,
                    real_test_loader,
                    real_ratio,
                    batch_size
                )
            except Exception as e:
                return arch_name, {"error": str(e)}

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_model, name, fn)
                for name, fn in model_factories.items()
            ]

            for future in as_completed(futures):
                arch_name, result = future.result()
                results[arch_name] = result

        return results

    # ----------------------------
    # Optional: helper analysis
    # ----------------------------
    def summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produces a simple comparison summary.
        """

        summary = {}

        for model_name, data in results.items():
            if "error" in data:
                continue

            test_metrics = data.get("test_metrics", {})

            summary[model_name] = {
                "main_score": test_metrics.get("accuracy")
                or test_metrics.get("loss")
                or test_metrics,
            }

        return summary