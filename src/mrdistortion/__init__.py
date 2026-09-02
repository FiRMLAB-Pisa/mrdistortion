"""Geometric distortion correction for MRI.

Two distortions, and they are not the same kind of problem.

A gradient's field departs from linearity away from isocentre, so the image is
sampled on a grid that is not the one it is displayed on. That departure is a
property of the coil, stated once by its manufacturer as spherical-harmonic
coefficients, and correcting it is deterministic: evaluate the harmonics over
the grid, and resample.

Susceptibility distortion is a property of the subject, not the scanner. It is
measured rather than looked up, and what measures it is a second acquisition
with the phase encoding reversed. That estimation is PyHySCO's, which is
GPL-3.0-only and therefore an optional extra: it is imported inside the call
that needs it, and never bundled.

No vendor coefficient table is bundled or persisted. A site whose coefficients
are not a file satisfies :class:`CoefficientAccessor` instead.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._gradunwarp import (
    CoefficientAccessor,
    GradientCoefficients,
    Gradunwarp,
    ImageGeometry,
)
from ._pyhysco import SusceptibilityCorrection, correct_susceptibility
from ._spiral import (
    ReadoutTiming,
    SpiralTransfer,
    autofocus,
    deblur,
    fit_transfer,
)

try:
    __version__ = _distribution_version(__name__)
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "CoefficientAccessor",
    "GradientCoefficients",
    "Gradunwarp",
    "ImageGeometry",
    "ReadoutTiming",
    "SpiralTransfer",
    "SusceptibilityCorrection",
    "__version__",
    "autofocus",
    "correct_susceptibility",
    "deblur",
    "fit_transfer",
]
