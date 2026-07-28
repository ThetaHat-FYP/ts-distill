"""
Simple Callback
---------------
A minimal training callback that prints a progress line to stdout.

Callbacks in this framework are objects passed to Trainer.fit().  The trainer
calls their hooks (on_train_begin, on_epoch_end, on_train_end) automatically.
You can stack multiple callbacks in the list — for example, a recorder and a
printer at the same time.
"""

from ts_distill.trainer.callback.base import BaseCallback

from ts_distill._logging import get_logger

logger = get_logger(__name__)


class SimpleCallback(BaseCallback):
    """
    Prints a loss update to the console every `print_every` epochs.

    Args:
        print_every (int): Frequency of log lines.  Default 4 means you see
                           output at epochs 4, 8, 12, ... keeping the console
                           readable without being too quiet.
    """

    def __init__(self, print_every: int = 4):
        self.print_every = print_every

    def on_train_begin(self, model, **kwargs):
        """Announce the start of training."""
        logger.info("   Training started...")

    def on_epoch_end(self, model, epoch: int, loss: float, **kwargs):
        """Log the loss every `print_every` epochs."""
        if (epoch + 1) % self.print_every == 0:
            logger.info(f"   Epoch {epoch + 1:>4} | Loss: {loss:.6f}")

    def on_train_end(self, model, **kwargs):
        """Announce completion."""
        logger.info("   Training complete.")
