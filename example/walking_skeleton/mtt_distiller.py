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
    """Matching Training Trajectories (MTT) dataset distillation.
    
    Matches student training trajectory on synthetic data to expert trajectory on real data.
    NOTE: Model must not have stateful buffers (e.g., BatchNorm running stats).
    """
    
    def __init__(
        self,
        initializer: BaseInitializer,
        matcher: BaseTrajectoryMatcher,
        model_factory: Callable,
        expert_recorder: BaseTrajectoryRecorder,
        expert_epochs: int = 10,       # Distance between expert checkpoints (k)
        syn_batch_size: int = 128,     # Mini-batch size for synthetic data training
        synthetic_lr: float = 0.01,
        student_lr: float = 0.01,
        student_steps: int = 10,
        device: str = 'cpu'
    ) -> None:
        super().__init__(initializer, matcher)
        self.model_factory = model_factory
        self.expert_recorder = expert_recorder
        self.expert_epochs = expert_epochs
        self.syn_batch_size = syn_batch_size
        self.synthetic_lr = synthetic_lr
        self.student_lr = student_lr
        self.student_steps = student_steps
        self.device = device
        self.criterion = nn.MSELoss()
        
    def distill(
        self, 
        source_data: torch.Tensor, 
        n_steps: int, 
        n_synthetic: Optional[int] = None
    ) -> torch.Tensor:
        
        if n_synthetic is None:
            n_synthetic = 50
            
        # Initialize synthetic data
        synthetic_shape = (n_synthetic, source_data.shape[1], source_data.shape[2])
        synthetic_data = self.initializer.initialize(
            synthetic_shape,
            real_data_reference=source_data
        ).to(self.device)
        
        if not synthetic_data.requires_grad:
            synthetic_data.requires_grad_(True)
            
        synthetic_optimizer = torch.optim.Adam([synthetic_data], lr=self.synthetic_lr)
        
        print(f"Distilling {n_synthetic} samples over {n_steps} steps...")
        
        # MTT outer loop: Optimize synthetic data to match expert trajectory
        for step in range(n_steps):
            synthetic_optimizer.zero_grad()
            
            # Step 1: Sample expert trajectory pair (θ_t, θ_{t+k}) using the gap parameter
            start_checkpoint, end_checkpoint = self.expert_recorder.sample_checkpoint_pair(step_gap=self.expert_epochs)
            expert_start_weights = start_checkpoint['weights']  # θ_t
            expert_end_weights = end_checkpoint['weights']      # θ_{t+k}
            
            # Step 2: Initialize student with expert's start weights θ_t (CRITICAL)
            student_model = self.model_factory().to(self.device)
            
            # Step 3: Train student on synthetic data for k steps starting from θ_t
            final_student_weights = self._train_student_functional(
                student_model,
                synthetic_data,
                expert_start_weights
            )
            
            # Step 4: Calculate trajectory matching loss L_MTT = ||θ_student - θ_{t+k}||²
            matching_loss = 0.0
            for name in final_student_weights.keys():
                if name in expert_end_weights:
                    # Expert end weights are constants (detach)
                    expert_param = expert_end_weights[name].to(self.device).detach()
                    matching_loss += torch.sum(
                        (final_student_weights[name] - expert_param) ** 2
                    )
            
            # Step 5: Meta-optimization - backprop to synthetic data only
            matching_loss.backward()
            synthetic_optimizer.step()
            
            if (step + 1) % 5 == 0 or step == 0:
                print(f"[Step {step+1:>3}/{n_steps}] Loss: {matching_loss.item():.4f}")
                
        return synthetic_data.detach()
    
    def _train_student_functional(
        self, 
        student_model: nn.Module, 
        synthetic_data: torch.Tensor, 
        expert_start_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        
        # Initialize student parameters with expert's θ_t
        params = {}
        for name in student_model.state_dict().keys():
            if name in expert_start_weights:
                params[name] = expert_start_weights[name].to(self.device).clone().detach().requires_grad_(True)
            else:
                params[name] = student_model.state_dict()[name].clone().detach().requires_grad_(True)
        
        num_syn_samples = synthetic_data.shape[0]
        
        # Inner loop: k SGD steps on synthetic data
        for _ in range(self.student_steps):
            param_list = [params[name] for name in params.keys()]
            
            # Batching logic to prevent OOM
            batch_size = min(self.syn_batch_size, num_syn_samples)
            indices = torch.randperm(num_syn_samples)[:batch_size]
            batch_data = synthetic_data[indices]
            
            # Split data: past timesteps → future prediction
            seq_len = batch_data.shape[1] // 2
            inputs = batch_data[:, :seq_len, :]
            targets = batch_data[:, seq_len:, :]
            
            # Forward pass using functional call (preserves computational graph)
            predictions = torch.func.functional_call(
                student_model,
                params,
                (inputs,)
            )
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Compute gradients with create_graph=True for meta-gradient flow
            grads = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                allow_unused=False
            )
            
            # SGD update (functional, not in-place)
            params = {
                name: param - self.student_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            }
        
        return params