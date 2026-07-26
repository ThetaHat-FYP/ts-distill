"""
Forecasting models
==================
Small, self-contained forecasters used both as the expert (whose trajectory is
matched) and as the probe (trained on distilled data and scored on real data).

All models share the signature (batch, seq_len, in_features) ->
(batch, pred_len, in_features), so any model can stand in for any other — this
is what makes the cross-architecture evaluation possible.

Prefer `create_model(model_type, seq_len, pred_len, in_features, model_kwargs)`
over importing a class directly; it keeps calling code architecture-agnostic.
"""

from ts_distill.models.base import BaseForecaster
from ts_distill.models.factory import create_model
from ts_distill.models.mlp import MLP
from ts_distill.models.cnn import CNN
from ts_distill.models.lstm import LSTM
from ts_distill.models.dlinear import DLinear

__all__ = [
    'BaseForecaster',
    'create_model',
    'MLP',
    'CNN',
    'LSTM',
    'DLinear',
]
