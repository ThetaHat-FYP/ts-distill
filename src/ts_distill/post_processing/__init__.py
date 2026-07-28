"""
Post-processing — repairs temporal structure AFTER distillation.

Most distillation research builds temporal constraints INTO the objective. This
package takes the opposite route: distil for utility first, then repair the
frequency content as a separate step, so utility and fidelity can be traded
explicitly through one parameter instead of being entangled in the loss.

`FFTAmplitudePostFix(alpha)` keeps the synthetic phase and blends its amplitude
spectrum toward the real data's:

    alpha = 0   leave the distilled data untouched
    alpha = 1   fully adopt the real amplitude spectrum

The effect on spectral distance is exactly linear, d(alpha) = (1-alpha)*d(0),
so a target fidelity can be dialled in without re-running anything.
"""

from .fft_post_fix import FFTAmplitudePostFix

__all__ = ['FFTAmplitudePostFix']
