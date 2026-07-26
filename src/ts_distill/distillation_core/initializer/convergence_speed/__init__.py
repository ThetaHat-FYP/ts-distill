"""
Convergence-speed metric
========================
Area under the "downstream test MSE vs. distillation steps" curve. A smaller
AUC means the run spent less time at high error, i.e. it converged faster.

Used to compare initializer strategies on convergence speed rather than on a
single end-point MSE.
"""

from ts_distill.distillation_core.initializer.convergence_speed.base import (
    calculate_convergence_auc,
)

__all__ = ['calculate_convergence_auc']
