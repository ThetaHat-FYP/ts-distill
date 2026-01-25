"""
MTT Distiller - Matching Training Trajectories
Functional gradient-based data distillation algorithm.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from src.ts_distill.distillation_core.initializer.base import BaseInitializer
from src.ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher
from src.ts_distill.trajectory.recorder.base import BaseTrajectoryRecorder


class MTTDistiller(BaseDistiller):
    """
    Matching Training Trajectories distiller with functional gradient updates.
    
    Uses torch.func.functional_call to maintain computational graph through
    student training steps, enabling end-to-end backpropagation to synthetic data.
    """
    
    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory: Callable,
        expert_recorder: BaseTrajectoryRecorder,
        synthetic_lr: float = 0.01,
        student_lr: float = 0.01,
        student_steps: int = 10
    ) -> None:
        """
        Args:
            initializer: Strategy for initializing synthetic data
            matcher: Trajectory distance metric
            model_factory: Function returning new model instance
            expert_recorder: Recorder containing expert trajectories
            synthetic_lr: Learning rate for synthetic data optimization
            student_lr: Learning rate for student model training
            student_steps: Number of student training steps per iteration
        """
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.expert_recorder = expert_recorder
        self.synthetic_lr = synthetic_lr
        self.student_lr = student_lr
        self.student_steps = student_steps
        self.criterion = nn.MSELoss()
        
    def distill(
        self, 
        source_data: torch.Tensor, 
        n_steps: int, 
        n_synthetic: Optional[int] = None
    ) -> torch.Tensor:
        """
        Execute MTT distillation.
        
        Args:
            source_data: Real data for initialization reference
            n_steps: Number of distillation iterations
            n_synthetic: Number of synthetic samples to generate
            
        Returns:
            Distilled synthetic dataset
        """
        if n_synthetic is None:
            n_synthetic = 50
            
        # Initialize synthetic data
        synthetic_shape = (n_synthetic, source_data.shape[1], source_data.shape[2])
        synthetic_data = self.initializer.initialize(
            synthetic_shape,
            real_data_reference=source_data
        )
        
        synthetic_optimizer = torch.optim.SGD([synthetic_data], lr=self.synthetic_lr)
        
        print(f"Distilling {n_synthetic} samples over {n_steps} steps...")
        
        # Main distillation loop
        for step in range(n_steps):
            synthetic_optimizer.zero_grad()
            
            # Sample expert trajectory
            expert_checkpoint = self.expert_recorder.sample_checkpoint()
            expert_weights = expert_checkpoint['weights']
            
            # Initialize and train student
            student_model = self.model_factory()
            final_student_weights = self._train_student_functional(
                student_model,
                synthetic_data,
                expert_weights
            )
            
            # Calculate trajectory matching loss
            student_params = list(final_student_weights.values())
            expert_params = list(expert_weights.values())
            matching_loss = self.matcher.calculate_loss(student_params, expert_params)
            
            # Update synthetic data
            matching_loss.backward()
            synthetic_optimizer.step()
            
            # Progress logging
            if (step + 1) % 5 == 0 or step == 0:
                data_min = synthetic_data.min().item()
                data_max = synthetic_data.max().item()
                loss_val = matching_loss.item()
                print(f"[Step {step+1:>3}/{n_steps}] Loss: {loss_val:.4f} | "
                      f"Data Range: [{data_min:.2f}, {data_max:.2f}]")
        
        print("Distillation complete.\n")
        return synthetic_data.detach()
    
    def _train_student_functional(
        self, 
        student_model: nn.Module, 
        synthetic_data: torch.Tensor, 
        expert_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Train student using functional gradient updates.
        
        Maintains computational graph for meta-gradient computation.
        
        Args:
            student_model: Fresh model instance
            synthetic_data: Synthetic data with gradients enabled
            expert_weights: Target expert weights (unused but kept for reference)
            
        Returns:
            Final student weights with computational graph intact
        """
        params = {name: param.clone() for name, param in student_model.named_parameters()}
        param_list = list(params.values())
        
        for _ in range(self.student_steps):
            # Clear gradients
            for p in param_list:
                if p.grad is not None:
                    p.grad = None
            
            # Split into inputs (past) and targets (future)
            # Input: all timesteps except the last one
            # Target: only the last timestep
            inputs = synthetic_data[:, :-1, :]
            targets = synthetic_data[:, -1:, :]
            
            # Forward pass with functional call using only past data
            predictions = torch.func.functional_call(
                student_model,
                params,
                (inputs,)
            )
            
            # Compute loss and gradients
            loss = self.criterion(predictions, targets)
            grads = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                retain_graph=True
            )
            
            # Manual parameter update
            new_params = {}
            for (name, param), grad in zip(params.items(), grads):
                new_params[name] = param - self.student_lr * grad
            
            params = new_params
            param_list = list(params.values())
        
        return params
