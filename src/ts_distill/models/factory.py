"""
Model Factory
-------------
A central registry that maps model name strings to their constructors.

Adding a new model
~~~~~~~~~~~~~~~~~~
1.  Implement it as a subclass of BaseForecaster in ts_distill/models/.
2.  Import it below and add an entry to _MODEL_REGISTRY.
3.  Expose any model-specific kwargs in the CONFIG['models'] dict of your
    experiment script.

That's it — no other files need to change.
"""

from ts_distill.models.dlinear import DLinear
from ts_distill.models.lstm import LSTM
from ts_distill.models.mlp import MLP
from ts_distill.models.cnn import CNN

# ---------------------------------------------------------------------------
# Internal registry
# Each entry is a lambda that accepts (seq_len, pred_len, in_features, kwargs)
# and returns a freshly constructed nn.Module.
# ---------------------------------------------------------------------------
_MODEL_REGISTRY = {
    'DLinear': lambda sl, pl, c, kw: DLinear(
        seq_len=sl, pred_len=pl, channels=c, **kw
    ),
    'LSTM': lambda sl, pl, c, kw: LSTM(
        input_dim=c, seq_len=sl, pred_len=pl, **kw
    ),
    'MLP': lambda sl, pl, c, kw: MLP(
        seq_len=sl, pred_len=pl, **kw
    ),
    'CNN': lambda sl, pl, c, kw: CNN(
        channel=c, seq_len=sl, pred_len=pl, **kw
    ),
}


def create_model(
    model_type: str,
    seq_len: int,
    pred_len: int,
    in_features: int,
    model_kwargs: dict = None,
):
    """
    Instantiate a forecasting model by name.

    This is the single entry-point for model construction used by the expert
    trainer, the distiller's inner loop (model_factory), and the evaluation
    trainers.  Every call returns a fresh, randomly-initialised instance.

    Args:
        model_type (str):    Name of the model. Must be a key in _MODEL_REGISTRY.
                             Currently supported: 'DLinear', 'LSTM', 'MLP', 'CNN'.
        seq_len (int):       Number of input timesteps fed to the model.
        pred_len (int):      Number of output timesteps the model must forecast.
        in_features (int):   Number of input channels (multivariate dimension).
        model_kwargs (dict): Extra keyword arguments forwarded to the model's
                             __init__.  For example:
                               DLinear → {'individual': True}
                               LSTM    → {'hidden_dim': 32, 'num_layers': 2}

    Returns:
        nn.Module: An untrained model instance, ready to be passed to a Trainer.

    Raises:
        ValueError: If model_type is not found in the registry.

    Example::

        # Inside run_cycle.py — create a model factory closure for the distiller
        def make_model():
            return create_model(
                model_type  = CONFIG['active_model'],
                seq_len     = CONFIG['seq_len'],
                pred_len    = CONFIG['pred_len'],
                in_features = CONFIG['in_features'],
                model_kwargs= CONFIG['models'][CONFIG['active_model']],
            )

        distiller = MTTDistiller(model_factory=make_model, ...)
    """
    if model_kwargs is None:
        model_kwargs = {}

    if model_type not in _MODEL_REGISTRY:
        supported = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported models: {supported}"
        )

    return _MODEL_REGISTRY[model_type](seq_len, pred_len, in_features, model_kwargs)
