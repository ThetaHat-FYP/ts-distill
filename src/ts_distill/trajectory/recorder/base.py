"""
Trajectory recorder interface — captures the expert's PATH, not its endpoint.

MTT matches how a model travels through parameter space, so the intermediate
weights are the training signal and must be saved as training happens; they
cannot be reconstructed from the final checkpoint.

Distillation samples (theta_start, theta_target) pairs a fixed gap apart from
what is recorded here, so the recorder's density sets which segments can be
matched at all.
"""

from abc import ABC, abstractmethod
from ts_distill.trainer.callback.base import BaseCallback


class BaseTrajectoryRecorder(BaseCallback, ABC):
    """
    Interface for saving model states. 
    Inherits from BaseCallback so it can be plugged directly into the Trainer.
    """

    # --- Contract 1: The Training Hook (From BaseCallback) ---
    def on_epoch_end(self, model, epoch, loss=None, **kwargs):
        """
        The Trainer calls this automatically. 
        We simply delegate this to record_checkpoint.
        """
        self.record_checkpoint(model, step=epoch)

    # --- Contract 2: The Recorder Specifics ---
    @abstractmethod
    def record_checkpoint(self, model, step: int):
        """
        The logic for capturing weights.
        (e.g., deepcopy(model.state_dict()) or saving to disk immediately)
        """
        pass

    @abstractmethod
    def save_buffer(self, file_path: str):
        """Writes the accumulated list of trajectories to disk."""
        pass

    @abstractmethod
    def load_buffer(self, file_path: str):
        """Loads trajectories from disk into memory."""
        pass
    
    @abstractmethod
    def get_trajectory(self):
        """Returns the current list of recorded weights (for immediate use)."""
        pass