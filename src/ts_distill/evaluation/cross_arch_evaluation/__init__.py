"""
Cross-architecture evaluation
=============================
Distilled data is only useful if it transfers. This package distils from one
expert architecture and then trains probes of EVERY architecture on the result,
producing the cross-architecture generalisation matrix.
"""

from ts_distill.evaluation.cross_arch_evaluation.cross_arch_comparison_evaluator import (
    CrossArchComparisonEvaluator,
)

__all__ = ['CrossArchComparisonEvaluator']
