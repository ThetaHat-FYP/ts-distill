from typing import List, Callable, Dict
from ts_distill.evaluation.base import BaseEvaluator

class CrossModelEvaluator(BaseEvaluator):
    """
    Tests if synthetic data distilled on one architecture (e.g., LSTM)
    works well when training different architectures (e.g., TCN, Transformer).
    """

    def evaluate_architectures(self, synthetic_data, real_test_loader, model_factories: Dict[str, Callable]):
        """
        Args:
            synthetic_data: The distilled dataset.
            model_factories: Dictionary of model creators.
                             Ex: {'lstm': lambda: LSTM(...), 'tcn': lambda: TCN(...)}
        
        Returns:
            Dict: Performance scores for each architecture.
        """
        results = {}
        
        for arch_name, create_model_fn in model_factories.items():
            print(f"Evaluating Cross-Model Performance on: {arch_name}...")
            
            # 1. Create a fresh instance of the target architecture
            model = create_model_fn()
            
            # 2. Train it on the Synthetic Data (which was likely made by a different model)
            self.train_on_synthetic(synthetic_data, model)
            
            # 3. Test on Real Data
            metrics = self.test_on_real(model, real_test_loader)
            results[arch_name] = metrics
            
        return results