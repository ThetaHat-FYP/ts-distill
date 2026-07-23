"""
FFT Amplitude Post-Fix
----------------------
Post-distillation temporal correction via spectral amplitude blending.

After MTT distillation produces a synthetic sequence, this module adjusts
the frequency energy distribution of the synthetic data toward the real
training data's spectral envelope — without changing the phase (which
encodes the shape/structure that MTT learned).

Mathematical operation (per channel c):

    FFT(syn[:, c])       ->  amp_syn  *  exp(i * phase_syn)
    mean FFT(real segs)  ->  amp_real

    amp_blended = (1 - alpha) * amp_syn  +  alpha * amp_real
    S_fixed[:, c] = IFFT( amp_blended * exp(i * phase_syn) )

alpha=0.0  ->  no change (pure distilled output)
alpha=1.0  ->  amplitude fully replaced by real's; phase (shape) unchanged

Research note:
    The Pareto trade-off between temporal quality (fft_distance) and
    distillation utility (transfer_mse) as alpha varies is the core
    experimental result.  A sweet-spot alpha where fft_distance drops
    significantly with minimal transfer_mse increase validates that
    post-hoc temporal correction is feasible.
"""

import numpy as np
import torch


class FFTAmplitudePostFix:
    """
    Post-processes a distilled synthetic sequence by blending its FFT
    amplitude spectrum toward the real training data's spectral envelope.

    Args:
        alpha (float): Blend strength in [0, 1].
                       0.0 = untouched, 1.0 = full amplitude replacement.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def _mean_real_amplitude(self, real_np: np.ndarray, seg_len: int) -> np.ndarray:
        """
        Compute mean FFT amplitude across non-overlapping segments of the
        real training sequence.

        Averaging over many segments (e.g. 90 for ETTm1 with seg_len=384)
        gives a stable spectral target that represents the full distribution,
        not a single window.

        Args:
            real_np: (M, C) real training sequence — M rows, C channels.
            seg_len: length of each segment (= N_synthetic).

        Returns:
            (rfft_bins, C) mean amplitude array.
        """
        n_segs = len(real_np) // seg_len
        if n_segs == 0:
            # real shorter than synthetic — use the whole real sequence
            return np.abs(np.fft.rfft(real_np[:seg_len], axis=0))

        amps = []
        for i in range(n_segs):
            seg = real_np[i * seg_len : (i + 1) * seg_len, :]
            amps.append(np.abs(np.fft.rfft(seg, axis=0)))  # (bins, C)
        return np.mean(amps, axis=0)  # (bins, C)

    def apply(
        self,
        synthetic: np.ndarray | torch.Tensor,
        real_train: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Apply FFT amplitude blending to a distilled synthetic sequence.

        Args:
            synthetic:   (N, C) distilled synthetic sequence.
                         Accepts numpy array or torch Tensor.
            real_train:  (M, C) raw training sequence (same channels as synthetic).
                         Accepts numpy array or torch Tensor.

        Returns:
            (N, C) numpy array — post-processed synthetic sequence.
            Values stay in the same scale as input (StandardScaler space).
        """
        syn_np  = synthetic.detach().cpu().numpy() if isinstance(synthetic,  torch.Tensor) else np.asarray(synthetic)
        real_np = real_train.detach().cpu().numpy() if isinstance(real_train, torch.Tensor) else np.asarray(real_train)

        N = syn_np.shape[0]
        mean_real_amp = self._mean_real_amplitude(real_np, seg_len=N)  # (bins, C)

        result = np.zeros_like(syn_np)
        for c in range(syn_np.shape[1]):
            syn_fft   = np.fft.rfft(syn_np[:, c])
            syn_amp   = np.abs(syn_fft)
            syn_phase = np.angle(syn_fft)

            blended_amp  = (1 - self.alpha) * syn_amp + self.alpha * mean_real_amp[:, c]
            fixed_fft    = blended_amp * np.exp(1j * syn_phase)
            result[:, c] = np.fft.irfft(fixed_fft, n=N)

        return result

    @staticmethod
    def fft_distance(
        synthetic:  np.ndarray | torch.Tensor,
        real_train: np.ndarray | torch.Tensor,
    ) -> float:
        """
        Mean per-channel RMSE between synthetic and real amplitude spectra.
        Uses the same segmented averaging so the metric is consistent with apply().

        Lower = better spectral match = better temporal frequency preservation.

        Args:
            synthetic:  (N, C) synthetic sequence.
            real_train: (M, C) real training sequence.

        Returns:
            Scalar float — mean spectral distance across channels.
        """
        postfix = FFTAmplitudePostFix(alpha=0.0)  # instance only for _mean_real_amplitude
        syn_np  = synthetic.detach().cpu().numpy()  if isinstance(synthetic,  torch.Tensor) else np.asarray(synthetic)
        real_np = real_train.detach().cpu().numpy() if isinstance(real_train, torch.Tensor) else np.asarray(real_train)

        N = syn_np.shape[0]
        mean_real_amp = postfix._mean_real_amplitude(real_np, seg_len=N)

        distances = []
        for c in range(syn_np.shape[1]):
            syn_amp = np.abs(np.fft.rfft(syn_np[:, c]))
            distances.append(float(np.sqrt(np.mean((syn_amp - mean_real_amp[:, c]) ** 2))))
        return float(np.mean(distances))
