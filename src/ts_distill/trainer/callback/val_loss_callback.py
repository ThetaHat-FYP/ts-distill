"""
Validation-Loss Recorder Callback
----------------------------------
Records the model's validation loss at the end of every training epoch by
delegating to a caller-supplied `eval_fn`.

Trainer.fit() already accepts a `val_loader` for early stopping, but it does
not expose the per-epoch val-loss curve to callers. This callback plugs into
the same `callbacks` list (alongside e.g. SimpleRecorder) so the curve can be
captured without changing Trainer itself — used by the phase-boundary
detectors in ts_distill.trajectory.phase_detector, which need the full
val-loss trajectory rather than just the early-stopping decision.
"""

from typing import Callable, List

from ts_distill.trainer.callback.base import BaseCallback


class ValLossRecorderCallback(BaseCallback):
    """
    Args:
        eval_fn: Zero-argument callable returning the current validation
            loss (e.g. `lambda: trainer.eval_epoch(val_loader)`).
    """

    def __init__(self, eval_fn: Callable[[], float]) -> None:
        self.eval_fn = eval_fn
        self.val_losses: List[float] = []

    def on_train_begin(self, model, **kwargs):
        self.val_losses = []

    def on_epoch_end(self, model, epoch, loss, **kwargs):
        self.val_losses.append(self.eval_fn())

    def on_train_end(self, model, **kwargs):
        pass
