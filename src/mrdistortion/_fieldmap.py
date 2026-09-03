"""Off-resonance field maps from the phase of a single-echo acquisition.

At echo time ``TE`` the signal has already accrued ``2.pi.f.TE`` radians of
phase from off-resonance, so the field is readable directly from the image
phase — no search, no focus metric, and no ambiguity as long as ``|f|`` stays
under ``1/(2.TE)``. A short echo time buys a wide unambiguous range and pays for
it in sensitivity: at ``TE`` of 0.7 ms one radian of phase error is 227 Hz, so
whatever phase is not off-resonance has to be removed first.

What is not off-resonance is the coil phase and the object's own. Both are
removed by estimating the sensitivities from the same images, low-pass filtered:
combining with those leaves the phase that varies faster than the sensitivities
do, which is the field. Sensitivities from any other source leave a residual
that swamps the measurement.

Fat is off-resonance too, about 440 Hz below water at 3T, so it shows in the map
wherever there is fat. That is the field the acquisition actually experienced,
and correcting for it is right.
"""

from __future__ import annotations

import math

import torch

__all__ = ["field_map_from_phase"]


def _smooth(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-smooth a complex volume, separably, over its trailing 3 axes."""
    if sigma <= 0:
        return volume
    width = int(3 * sigma) | 1
    offsets = torch.arange(
        -(width // 2), width // 2 + 1, device=volume.device, dtype=torch.float32
    )
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    result = volume
    for axis in range(3):
        view = [1, 1, 1]
        view[axis] = width
        padding = [0] * 6
        padding[2 * (2 - axis)] = width // 2
        padding[2 * (2 - axis) + 1] = width // 2
        parts = []
        for component in (result.real, result.imag):
            padded = torch.nn.functional.pad(
                component[None, None], padding, mode="replicate"
            )
            parts.append(
                torch.nn.functional.conv3d(padded, kernel.reshape(1, 1, *view))[0, 0]
            )
        result = torch.complex(*parts)
    return result


def _sensitivities(coil_images: torch.Tensor, width: float) -> torch.Tensor:
    """Low-pass each coil image to get its sensitivity, keeping ``width`` of k-space."""
    spectrum = torch.fft.fftn(coil_images, dim=(1, 2, 3))
    for axis, length in enumerate(coil_images.shape[1:]):
        coordinate = torch.fft.fftfreq(length, device=coil_images.device) * 2.0
        taper = torch.where(
            coordinate.abs() < width,
            0.5 + 0.5 * torch.cos(math.pi * coordinate / width),
            torch.zeros((), device=coil_images.device),
        )
        view = [1, 1, 1, 1]
        view[axis + 1] = length
        spectrum = spectrum * taper.reshape(view)
    return torch.fft.ifftn(spectrum, dim=(1, 2, 3))


def field_map_from_phase(
    coil_images: torch.Tensor,
    *,
    echo_time: float,
    smoothing: float = 2.0,
    calibration_width: float = 0.06,
) -> torch.Tensor:
    """Estimate off-resonance from the phase of single-echo coil images.

    Parameters
    ----------
    coil_images
        Complex images, one per coil, shaped ``(coils, *spatial)`` with three
        spatial axes. Reconstruct them without coil combination: the
        sensitivities are estimated from these same images.
    echo_time
        Echo time in seconds. Frequencies beyond ``1/(2 * echo_time)`` wrap.
    smoothing
        Width, in voxels, of the Gaussian applied to the combined image before
        its phase is read. It should match the scale over which the field
        varies rather than the scale of the anatomy — a few millimetres.
        Smoothing the complex image rather than the phase avoids wrap artefacts.
    calibration_width
        Fraction of each k-space axis kept when low-pass filtering the coil
        images into sensitivities.

    Returns
    -------
    torch.Tensor
        Off-resonance in Hz at each voxel, real, shaped like one coil image.

    Raises
    ------
    ValueError
        If the images are not complex, or not ``(coils, *spatial)`` with three
        spatial axes.

    Examples
    --------
    >>> import torch
    >>> from mrdistortion import field_map_from_phase
    >>> coils = torch.ones(4, 8, 8, 8, dtype=torch.complex64)
    >>> field = field_map_from_phase(coils, echo_time=0.7e-3)
    >>> field.shape
    torch.Size([8, 8, 8])
    >>> bool(field.abs().max() < 1.0)
    True
    """
    if not coil_images.is_complex():
        raise ValueError("coil images must be complex")
    if coil_images.ndim != 4:
        raise ValueError(
            f"expected (coils, *spatial) with three spatial axes, got shape "
            f"{tuple(coil_images.shape)}"
        )
    if echo_time <= 0:
        raise ValueError("echo_time must be positive")

    sensitivities = _sensitivities(coil_images, calibration_width)
    combined = (coil_images * sensitivities.conj()).sum(0) / (
        sensitivities.abs().pow(2).sum(0) + 1e-12
    )
    return _smooth(combined, smoothing).angle() / (2 * math.pi * echo_time)
