"""
Evaluation
==========
Scoring distilled data by training a fresh probe model on it and measuring that
model's error on the real held-out test set.

Modules
-------
evaluation           : Evaluator — the standard "train on X, test on real" probe.

Sub-packages
------------
hybrid_evaluation    : Mixing synthetic with real data and finding the best mix
                       ratio.
cross_arch_evaluation: Distilling with one architecture and evaluating on others.
sample_selector      : Strategies for choosing which real windows to mix in.
"""

from ts_distill.evaluation.base import BaseEvaluator
from ts_distill.evaluation.evaluation import Evaluator

__all__ = [
    'BaseEvaluator',
    'Evaluator',
]
