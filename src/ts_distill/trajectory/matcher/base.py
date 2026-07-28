"""
Trajectory matcher interface — the distance MTT actually minimises.

Given the student's parameters after training on synthetic data and the
expert's parameters at the corresponding trajectory point, return how far apart
they are. That scalar is the distillation loss.

Must be differentiable with respect to the student parameters: the gradient
travels from here back through the unrolled inner loop and into the synthetic
data tensor, which is the only thing being optimised.
"""

from abc import ABC, abstractmethod

class BaseTrajectoryMatcher(ABC):
    """Interface for calculating distance between Student and Expert."""

    @abstractmethod
    def calculate_loss(self, student_params, expert_params) -> float:
        """
        Computes the distance metric (MSE, Cosine, etc.)
        Must be differentiable!
        """
        pass