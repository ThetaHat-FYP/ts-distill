"""
Convergence-Speed — Base Utilities
===================================
Shared helper for the per-strategy curve scripts (random_curve.py,
geo_curve.py, uncertainty_curve.py). Computes the Area Under the Curve (AUC)
of downstream test MSE vs. distillation steps — a smaller AUC means the
model spent less time at higher error, i.e. faster convergence.
"""

import numpy as np


def calculate_convergence_auc(steps, mse_values) -> float:
    """
    Args:
        steps      (array-like): Distillation step checkpoints, e.g. [0, 100, ..., 600].
        mse_values (array-like): Downstream test MSE at each checkpoint.

    Returns:
        float: Area under the MSE-vs-steps curve (trapezoidal rule).
    """
    return float(np.trapezoid(mse_values, x=steps))
