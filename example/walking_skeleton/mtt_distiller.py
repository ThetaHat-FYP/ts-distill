"""
MTT Distiller Implementation
Implements Matching Training Trajectories algorithm with functional gradient updates.

CRITICAL: Uses torch.func.functional_call to maintain computational graph!
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ts_distill.distillation_core.distillation_algorithm.base import BaseDistiller
from src.ts_distill.distillation_core.initializer.base import BaseInitializer
from src.ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher


class MTTDistiller(BaseDistiller):
    """
    Matching Training Trajectories (MTT) Distiller.
    
    Key Insight: We need to backpropagate through the student training process
    to update the synthetic data. This requires functional gradient updates.
    
    Algorithm:
    1. Sample expert trajectory checkpoint
    2. Initialize student model 
    3. Train student on synthetic data (functional updates!)
    4. Match student trajectory to expert trajectory
    5. Backpropagate to update synthetic data
    """
    
    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory,
        expert_recorder,
        synthetic_lr=0.01,
        student_lr=0.01,
        student_steps=10,
        match_every=5
    ):
        """
        Args:
            initializer: How to initialize synthetic data
            matcher: How to measure trajectory distance
            model_factory: Function that returns new model instance
            expert_recorder: Recorder with expert trajectories
            synthetic_lr: Learning rate for updating synthetic data
            student_lr: Learning rate for student model
            student_steps: Number of student training steps per iteration
            match_every: Match trajectory every N steps
        """
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.expert_recorder = expert_recorder
        self.synthetic_lr = synthetic_lr
        self.student_lr = student_lr
        self.student_steps = student_steps
        self.match_every = match_every
        
        # Criterion for student training
        self.criterion = nn.MSELoss()
        
    def distill(self, source_data, n_steps: int, n_synthetic: int = None):
        """
        Main distillation loop.
        
        Args:
            source_data: Real data (used only for initialization reference)
            n_steps: Number of distillation iterations
            n_synthetic: Number of synthetic samples to generate (default: 50)
            
        Returns:
            Final synthetic dataset tensor
        """
        if n_synthetic is None:
            n_synthetic = 50
            
        print("\n" + "="*60)
        print("Starting MTT Distillation")
        print("="*60)
        
        # Step 1: Initialize synthetic data
        synthetic_shape = (n_synthetic, source_data.shape[1], source_data.shape[2])
        synthetic_data = self.initializer.initialize(
            synthetic_shape,
            real_data_reference=source_data
        )
        
        # Optimizer for synthetic data
        synthetic_optimizer = torch.optim.SGD([synthetic_data], lr=self.synthetic_lr)
        
        print(f"Synthetic Data Shape: {synthetic_data.shape}")
        print(f"Distillation Steps: {n_steps}")
        print(f"Student Steps per Iteration: {self.student_steps}")
        print(f"Match Trajectory Every: {self.match_every} steps\n")
        
        # Main distillation loop
        for step in range(n_steps):
            synthetic_optimizer.zero_grad()
            
            # Step 2: Sample expert trajectory
            expert_checkpoint = self.expert_recorder.sample_checkpoint()
            expert_weights = expert_checkpoint['weights']
            
            # Step 3: Initialize student model
            student_model = self.model_factory()
            
            # Step 4: Train student on synthetic data (FUNCTIONAL!)
            final_student_weights = self._train_student_functional(
                student_model,
                synthetic_data,
                expert_weights
            )
            
            # Step 5: Calculate trajectory matching loss
            student_params = list(final_student_weights.values())
            expert_params = list(expert_weights.values())
            
            matching_loss = self.matcher.calculate_loss(student_params, expert_params)
            
            # Step 6: Backpropagate to update synthetic data
            matching_loss.backward()
            synthetic_optimizer.step()
            
            # Logging
            if (step + 1) % 5 == 0 or step == 0:
                print(f"Step {step+1}/{n_steps} | Matching Loss: {matching_loss.item():.6f} | "
                      f"Synthetic Data Range: [{synthetic_data.min().item():.3f}, "
                      f"{synthetic_data.max().item():.3f}]")
        
        print("\n" + "="*60)
        print("Distillation Complete!")
        print("="*60 + "\n")
        
        return synthetic_data.detach()
    
    def _train_student_functional(self, student_model, synthetic_data, expert_weights):
        """
        Train student model using FUNCTIONAL gradient updates.
        
        CRITICAL: This maintains the computational graph so that
        loss.backward() in the outer loop can update synthetic_data.
        
        Standard optimizer.step() would break the graph!
        
        Args:
            student_model: Fresh model instance
            synthetic_data: Synthetic data (with requires_grad=True)
            expert_weights: Target expert weights (for trajectory matching)
            
        Returns:
            Final student weights (as dict with computational graph intact)
        """
        # Get initial parameters as dict
        params = {name: param.clone() for name, param in student_model.named_parameters()}
        
        # Convert params dict to list format for easier manipulation
        param_list = list(params.values())
        
        # Student training loop
        for t in range(self.student_steps):
            # Zero gradients for parameters
            for p in param_list:
                if p.grad is not None:
                    p.grad = None
            
            # Forward pass using functional_call
            # This allows us to pass arbitrary parameters while maintaining gradients
            predictions = torch.func.functional_call(
                student_model,
                params,
                (synthetic_data,)
            )
            
            # Create targets for time series forecasting
            # Handle both windowed and non-windowed data
            if synthetic_data.shape[1] == 1:
                targets = synthetic_data[:, -1:, :]
            else:
                # For windowed data: predict last timestep
                targets = synthetic_data[:, -1, :].unsqueeze(1)
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Compute gradients w.r.t. parameters
            grads = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,  # CRITICAL: Maintain graph for outer loop
                retain_graph=True
            )
            
            # Manual gradient update (functional style)
            new_params = {}
            for (name, param), grad in zip(params.items(), grads):
                new_params[name] = param - self.student_lr * grad
            
            params = new_params
            param_list = list(params.values())
        
        return params
    
    def _extract_param_tensors(self, model_or_dict):
        """
        Helper to extract parameter tensors from model or state_dict.
        
        Args:
            model_or_dict: nn.Module or dict of parameters
            
        Returns:
            List of parameter tensors
        """
        if isinstance(model_or_dict, dict):
            return [p for p in model_or_dict.values()]
        else:
            return [p for p in model_or_dict.parameters()]


class MTTDistillerSimplified(BaseDistiller):
    """
    Simplified MTT implementation with manual gradient computation.
    
    Alternative approach that explicitly computes gradients without
    torch.func.functional_call (for older PyTorch versions).
    """
    
    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory,
        expert_recorder,
        synthetic_lr=0.01,
        student_lr=0.01,
        student_steps=10
    ):
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.expert_recorder = expert_recorder
        self.synthetic_lr = synthetic_lr
        self.student_lr = student_lr
        self.student_steps = student_steps
        self.criterion = nn.MSELoss()
        
    def distill(self, source_data, n_steps: int):
        """Main distillation loop."""
        print("\n" + "="*60)
        print("Starting MTT Distillation (Simplified)")
        print("="*60)
        
        # Initialize synthetic data
        synthetic_shape = (50, source_data.shape[1], source_data.shape[2])
        synthetic_data = self.initializer.initialize(
            synthetic_shape,
            real_data_reference=source_data
        )
        
        synthetic_optimizer = torch.optim.SGD([synthetic_data], lr=self.synthetic_lr)
        
        print(f"Synthetic Data Shape: {synthetic_data.shape}")
        print(f"Distillation Steps: {n_steps}\n")
        
        for step in range(n_steps):
            synthetic_optimizer.zero_grad()
            
            # Sample expert trajectory
            expert_checkpoint = self.expert_recorder.sample_checkpoint()
            expert_weights = expert_checkpoint['weights']
            
            # Initialize student
            student_model = self.model_factory()
            
            # Move expert weights to same device
            for k in expert_weights.keys():
                expert_weights[k] = expert_weights[k].to(synthetic_data.device)
            
            # Train student with manual updates
            self._train_student_manual(student_model, synthetic_data)
            
            # Calculate matching loss
            matching_loss = 0.0
            for (s_name, s_param), (e_name, e_param) in zip(
                student_model.named_parameters(),
                expert_weights.items()
            ):
                matching_loss += torch.sum((s_param - e_param.detach()) ** 2)
            
            # Update synthetic data
            matching_loss.backward()
            synthetic_optimizer.step()
            
            if (step + 1) % 5 == 0 or step == 0:
                print(f"Step {step+1}/{n_steps} | Matching Loss: {matching_loss.item():.6f}")
        
        print("\nDistillation Complete!\n")
        return synthetic_data.detach()
    
    def _train_student_manual(self, model, synthetic_data):
        """Train student with manual gradient updates maintaining graph."""
        for t in range(self.student_steps):
            # Forward
            output = model(synthetic_data)
            targets = synthetic_data[:, -1, :].mean(dim=1, keepdim=True)
            loss = self.criterion(output, targets)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                model.parameters(),
                create_graph=True  # CRITICAL for meta-learning!
            )
            
            # Manual update
            with torch.no_grad():
                for param, grad in zip(model.parameters(), grads):
                    # This keeps the computational graph
                    param.data = param.data - self.student_lr * grad
