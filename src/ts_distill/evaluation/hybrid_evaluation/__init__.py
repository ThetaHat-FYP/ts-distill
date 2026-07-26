"""
Hybrid evaluation
=================
Synthetic data alone is rarely optimal. Mixing a small fraction of real windows
back in usually beats both pure-synthetic and pure-real for a given budget.
This package measures that trade-off and picks the best mixing ratio.

Typical flow
------------
    evaluator = BaseHybridEvaluator(anchor_selector=RandomAnchorSelector(), ...)
    results   = evaluator.evaluate_mixing(..., mixing_ratios=(0.0, 0.1, 0.2))
    best      = find_optimal_ratio(ratios, mses)   # compromise programming
    dataset   = evaluator.hybridmixture(..., real_ratio=best['r_star'] / 100)

Anchor selectors decide WHICH real windows get mixed in (random, uniform stride,
importance-weighted, diversity-based).

`ratio_predictor` is an alternative, measurement-free route to r*: it estimates
the mix ratio from properties of the synthetic data itself, without running the
full ratio sweep. Import it directly from
``ts_distill.evaluation.hybrid_evaluation.ratio_predictor``.
"""

from ts_distill.evaluation.hybrid_evaluation.hybrid import (
    BaseHybridEvaluator,
    RandomAnchorSelector,
    StartExtendSelector,
    UniformStrideSelector,
    ImportanceWeightedSelector,
    DiversityAnchorSelector,
    compute_expert_losses,
)
from ts_distill.evaluation.hybrid_evaluation.optimal_ratio_finder import (
    find_optimal_ratio,
    interpolate_curve,
    extract_ratio_curve,
)

__all__ = [
    'BaseHybridEvaluator',
    'RandomAnchorSelector',
    'StartExtendSelector',
    'UniformStrideSelector',
    'ImportanceWeightedSelector',
    'DiversityAnchorSelector',
    'compute_expert_losses',
    'find_optimal_ratio',
    'interpolate_curve',
    'extract_ratio_curve',
]
