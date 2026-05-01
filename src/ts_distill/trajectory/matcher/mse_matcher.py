"""
MSE Trajectory Matcher
-----------------------
Measures the squared-error distance between two ordered lists of parameter
tensors.  Used in MTT's trajectory matching loss to score how close the
student's final parameters are to the expert's target parameters.

Why it must be differentiable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The matcher sits inside the outer meta-learning loop.  Its output (grand_loss)
is backpropagated all the way through the student's inner unrolling, through
the functional_call steps, and ultimately into the synthetic data tensor.
Any non-differentiable operation here would break the gradient signal.
"""

import torch

from ts_distill.trajectory.matcher.base import BaseTrajectoryMatcher


class MSEMatcher(BaseTrajectoryMatcher):
    """
    Computes the sum of squared differences between two parameter lists.

    Both lists must be the same length and each corresponding pair of tensors
    must have the same shape — this is guaranteed by the way MTTDistiller
    builds target_param_list and final_param_list.
    """

    def calculate_loss(
        self,
        student_params: list,
        expert_params: list,
    ) -> torch.Tensor:
        """
        Args:
            student_params (list[Tensor]): Parameter tensors from the unrolled
                                           student model (have gradients).
            expert_params  (list[Tensor]): Corresponding expert checkpoint
                                           tensors (will be detached here).

        Returns:
            Scalar tensor: Σ ||θ_student - θ_expert||²  (fully differentiable
            with respect to student_params, and through them to synthetic data).
        """
        return sum(
            torch.sum((s - e.detach()) ** 2)
            for s, e in zip(student_params, expert_params)
        )
