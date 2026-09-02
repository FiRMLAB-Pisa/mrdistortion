r"""Off-resonance deblurring for spiral readouts.

A spiral reads k-space over milliseconds, so off-resonance accrues phase along
the readout and the image blurs by an amount that grows with distance from
resonance. The accrued phase depends on where in k-space each sample was taken,
through the readout time map ``t(k)``, and for a centre-out spiral that map is a
function of ``|k|`` alone.

Correcting it is a space-variant operation: every voxel wants its own
demodulation frequency. The resolution is to write the demodulation as a short
sum whose k-space factor does not depend on the field map,

.. math::  e^{-2 \\pi i f t(k)} = \\sum_m a_m(f)\\, b_m(k)

so the field map enters only through per-voxel weights. This module fits that
factorization with :math:`b_m` constrained to the family
:math:`e^{\\alpha |k|^2}`, whose members are the only radial functions that
factor over the axes. Each term is then a separable convolution of the
reconstructed image, and the correction is post-processing: it needs neither
the raw data nor a field-map acquisition.

Because ``|k|`` is rotation invariant, the same factorization serves a 3D
spiral-projection acquisition, where every arm is a rotation of one base
spiral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

__all__ = [
    "ReadoutTiming",
    "SpiralTransfer",
    "autofocus",
    "deblur",
    "fit_transfer",
]

Backend = Literal["auto", "fft", "conv"]


@dataclass(frozen=True)
class ReadoutTiming:
    """Readout time as a function of squared k-space radius.

    Parameters
    ----------
    squared_radius
        Uniform grid on ``[0, 1]``, the squared radius normalised by its maximum.
    times
        Readout time at each grid point, normalised to ``[0, 1]``.
    density
        Number of trajectory samples landing at each grid point, used to weight
        the fit towards the parts of k-space the acquisition actually visits.
    duration
        Readout duration in seconds.
    """

    squared_radius: np.ndarray
    times: np.ndarray
    density: np.ndarray
    duration: float

    @classmethod
    def from_trajectory(
        cls,
        trajectory: np.ndarray,
        duration: float,
        *,
        points: int = 512,
    ) -> ReadoutTiming:
        """Derive the time map of one readout arm.

        Parameters
        ----------
        trajectory
            k-space coordinates of a single arm, shaped ``(samples, ndim)``.
            Units are arbitrary; only the radius profile matters.
        duration
            Readout duration in seconds.
        points
            Size of the uniform grid the time map is resampled onto.

        Returns
        -------
        ReadoutTiming
            The resampled time map.

        Raises
        ------
        ValueError
            If the radius is not monotonic, which means the arm is neither
            centre-out nor inward. An out-in arm must be split into its two
            halves and each half deblurred separately: the halves separate
            individually, their sum does not.
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2:
            raise ValueError(f"expected (samples, ndim), got {trajectory.shape}")
        radius = np.linalg.norm(trajectory, axis=1)
        step = np.diff(radius)
        tolerance = 1e-6 * max(radius.max(), 1.0)
        if not (np.all(step >= -tolerance) or np.all(step <= tolerance)):
            raise ValueError(
                "readout radius is not monotonic, so time is not a function of "
                "|k|; split an out-in arm and deblur each half separately"
            )
        peak = radius.max()
        if peak <= 0:
            raise ValueError("trajectory has zero extent")

        squared = (radius / peak) ** 2
        sample_times = np.linspace(0.0, 1.0, len(radius))
        grid = np.linspace(0.0, 1.0, points)
        order = np.argsort(squared)
        times = np.interp(grid, squared[order], sample_times[order])
        edges = np.linspace(0.0, 1.0, points + 1)
        density = np.histogram(squared, bins=edges)[0].astype(float)
        return cls(grid, times, np.maximum(density, 1e-6), float(duration))


@dataclass(frozen=True)
class SpiralTransfer:
    """Separable factorization of the off-resonance transfer.

    Parameters
    ----------
    rates
        Complex rate of each term, shaped ``(terms,)``, in units of inverse
        squared normalised radius. Term ``m`` contributes the k-space factor
        ``exp(rates[m] * |k|**2 / kmax**2)``.
    weights
        Weight of each term at each tabulated frequency, shaped
        ``(terms, frequencies)``.
    frequencies
        Uniformly spaced off-resonance frequencies in Hz that ``weights`` is
        tabulated on. Voxels outside this range are clamped to its ends.
    """

    rates: np.ndarray
    weights: np.ndarray
    frequencies: np.ndarray

    @property
    def terms(self) -> int:
        """Number of separable terms."""
        return len(self.rates)

    def error(self, timing: ReadoutTiming) -> float:
        """Worst density-weighted RMS error of the factorization.

        Parameters
        ----------
        timing
            The time map the transfer was fitted to.

        Returns
        -------
        float
            The largest relative error over the tabulated frequencies.
        """
        exact = _exact_transfer(timing, self.frequencies)
        fitted = np.exp(np.outer(timing.squared_radius, self.rates)) @ self.weights
        weight = timing.density[:, None]
        residual = np.sum(weight * np.abs(fitted - exact) ** 2, axis=0)
        return float(np.sqrt(residual / timing.density.sum()).max())


def _exact_transfer(timing: ReadoutTiming, frequencies: np.ndarray) -> np.ndarray:
    """Off-resonance transfer on the (radius, frequency) grid."""
    cycles = frequencies * timing.duration
    return np.exp(-2j * np.pi * np.outer(timing.times, cycles))


def fit_transfer(
    timing: ReadoutTiming,
    *,
    band: float,
    terms: int = 6,
    frequencies: int = 65,
) -> SpiralTransfer:
    """Fit a separable factorization of the off-resonance transfer.

    The rates are shared across frequency, so a correction costs ``terms``
    separable convolutions however finely the field map is resolved.

    Parameters
    ----------
    timing
        Time map of the readout.
    band
        Half-width of the off-resonance range to cover, in Hz.
    terms
        Number of separable terms. Cost is linear in this and memory is flat.
    frequencies
        Number of tabulated frequencies spanning ``[-band, band]``.

    Returns
    -------
    SpiralTransfer
        The fitted factorization. Check :meth:`SpiralTransfer.error` to see
        whether ``terms`` was enough.

    Examples
    --------
    >>> import numpy as np
    >>> from mrdistortion import ReadoutTiming, fit_transfer
    >>> angle = np.linspace(0, 12 * np.pi, 600)
    >>> arm = np.stack([angle * np.cos(angle), angle * np.sin(angle)], axis=1)
    >>> timing = ReadoutTiming.from_trajectory(arm, duration=5e-3)
    >>> transfer = fit_transfer(timing, band=100.0, terms=5)
    >>> transfer.terms
    5
    >>> bool(transfer.error(timing) < 0.05)
    True
    """
    if terms < 1:
        raise ValueError("terms must be at least 1")
    grid = timing.squared_radius
    tabulated = np.linspace(-band, band, frequencies)
    exact = _exact_transfer(timing, tabulated)

    rates = _shared_rates(grid, exact, terms)
    basis = np.exp(np.outer(grid, rates))
    root = np.sqrt(timing.density)[:, None]
    weights = np.linalg.lstsq(root * basis, root * exact, rcond=None)[0]
    return SpiralTransfer(rates, weights, tabulated)


def _shared_rates(grid: np.ndarray, exact: np.ndarray, terms: int) -> np.ndarray:
    """Rates of the dominant exponentials common to every frequency column.

    Multi-snapshot ESPRIT: the columns share a signal subspace because they are
    all sums of exponentials in the squared radius, so one shift-invariance
    solve recovers rates that serve the whole band.
    """
    window = len(grid) // 2
    blocks = [
        np.lib.stride_tricks.sliding_window_view(column, window)[:window]
        for column in exact.T
    ]
    subspace = np.linalg.svd(np.hstack(blocks), full_matrices=False)[0][:, :terms]
    shift = np.linalg.lstsq(subspace[:-1], subspace[1:], rcond=None)[0]
    poles = np.linalg.eigvals(shift).astype(complex)
    step = grid[1] - grid[0]
    rates = np.log(np.where(np.abs(poles) > 0, poles, 1.0)) / step
    # A growing term would amplify the corners of the Cartesian grid, which lie
    # outside the sampled sphere and carry no measured signal.
    return np.where(rates.real > 0, 1j * rates.imag, rates)


def _axis_factors(
    rate: complex,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """One-dimensional k-space factors whose product is ``exp(rate * u)``.

    Kept as separate axis factors so the full ``u`` volume is never formed.
    """
    factors = []
    for axis, length in enumerate(shape):
        coordinate = torch.fft.fftfreq(length, device=device) * 2.0
        factor = torch.exp(rate * coordinate.to(dtype) ** 2)
        view = [1] * len(shape)
        view[axis] = length
        factors.append(factor.reshape(view))
    return factors


def _term_weights(
    transfer: SpiralTransfer,
    term: int,
    field_map: torch.Tensor,
) -> torch.Tensor:
    """Per-voxel weight of one term, interpolated over the field map."""
    frequencies = transfer.frequencies
    spacing = frequencies[1] - frequencies[0]
    position = (field_map - float(frequencies[0])) / float(spacing)
    position = position.clamp(0, len(frequencies) - 1)
    lower = position.floor()
    fraction = (position - lower).to(field_map.dtype)
    index = lower.long()
    upper = index.clamp(max=len(frequencies) - 2) + 1
    table = torch.as_tensor(transfer.weights[term], device=field_map.device).to(
        _complex_for(field_map.dtype)
    )
    return torch.lerp(
        table[index.clamp(max=len(frequencies) - 2)],
        table[upper],
        fraction.to(table.dtype),
    )


def _complex_for(dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real one."""
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def deblur(
    image: torch.Tensor,
    field_map: torch.Tensor,
    transfer: SpiralTransfer,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Remove off-resonance blur from a reconstructed spiral image.

    Parameters
    ----------
    image
        Complex reconstructed image, shaped ``(*spatial,)`` or
        ``(batch, *spatial)`` for 2D and 3D alike. The spatial rank is taken
        from ``field_map``.
    field_map
        Off-resonance in Hz at each voxel, shaped ``(*spatial,)``.
    transfer
        Factorization from :func:`fit_transfer`.
    backend
        ``"fft"`` applies each term as an exact k-space multiply; ``"conv"``
        applies it as a truncated separable convolution; ``"auto"`` picks
        ``"conv"`` on CUDA when Triton is available and ``"fft"`` otherwise.

    Returns
    -------
    torch.Tensor
        The deblurred image, same shape and dtype as ``image``.

    Notes
    -----
    Peak memory is one accumulator plus one working volume, independent of the
    number of terms.
    """
    if not image.is_complex():
        raise ValueError("image must be complex")
    spatial = field_map.ndim
    axes = tuple(range(image.ndim - spatial, image.ndim))
    if image.shape[-spatial:] != field_map.shape:
        raise ValueError(
            f"image spatial shape {tuple(image.shape[-spatial:])} does not match "
            f"field map {tuple(field_map.shape)}"
        )
    chosen = _resolve_backend(backend, image)
    shape = tuple(image.shape[-spatial:])

    result = torch.zeros_like(image)
    fused = _fused_accumulate(image)
    spectrum = torch.fft.fftn(image, dim=axes) if chosen == "fft" else None
    working = torch.empty_like(image) if chosen == "fft" else None
    for term in range(transfer.terms):
        rate = complex(transfer.rates[term])
        if chosen == "fft":
            factors = _axis_factors(
                rate, shape, device=image.device, dtype=_real_for(image.dtype)
            )
            torch.mul(spectrum, factors[0], out=working)
            for factor in factors[1:]:
                working *= factor
            contribution = torch.fft.ifftn(working, dim=axes)
        else:
            from ._triton_spiral import separable_convolve

            contribution = separable_convolve(image, rate, axes)
        if fused is not None:
            fused(
                result,
                contribution,
                field_map,
                torch.as_tensor(
                    transfer.weights[term], device=image.device, dtype=image.dtype
                ),
                float(transfer.frequencies[0]),
                float(transfer.frequencies[1] - transfer.frequencies[0]),
            )
            continue
        result += _term_weights(transfer, term, field_map) * contribution
        del contribution
    return result


def _fused_accumulate(image: torch.Tensor):
    """Return the Triton weighted accumulator, when device and dtype allow it."""
    if image.device.type != "cuda" or image.dtype != torch.complex64:
        return None
    try:
        from ._triton_spiral import accumulate_weighted
    except ImportError:
        return None
    return accumulate_weighted


def _real_for(dtype: torch.dtype) -> torch.dtype:
    """Real dtype matching a complex one."""
    return torch.float64 if dtype == torch.complex128 else torch.float32


def _resolve_backend(backend: Backend, image: torch.Tensor) -> str:
    """Pick a backend, falling back to the exact one when Triton is absent."""
    if backend != "auto":
        return backend
    if image.device.type != "cuda":
        return "fft"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "fft"
    return "conv"


def _low_pass(shape: tuple[int, ...], device: torch.device, fraction: float):
    """Separable Hann window keeping the middle ``fraction`` of each k-space axis."""
    window = None
    for axis, length in enumerate(shape):
        coordinate = torch.fft.fftfreq(length, device=device) * 2.0
        taper = torch.where(
            coordinate.abs() < fraction,
            0.5 + 0.5 * torch.cos(torch.pi * coordinate / fraction),
            torch.zeros((), device=device),
        )
        view = [1] * len(shape)
        view[axis] = length
        taper = taper.reshape(view)
        window = taper if window is None else window * taper
    return window


def _box_filter(volume: torch.Tensor, window: int) -> torch.Tensor:
    """Mean over a cubic neighbourhood, applied one axis at a time."""
    if window <= 1:
        return volume
    padding = window // 2
    result = volume
    for axis in range(volume.ndim):
        result = result.movedim(axis, -1)
        shape = result.shape
        flat = result.reshape(1, -1, shape[-1])
        flat = torch.nn.functional.avg_pool1d(
            torch.nn.functional.pad(flat, (padding, padding), mode="replicate"),
            window,
            stride=1,
        )
        result = flat.reshape(shape).movedim(-1, axis)
    return result


def _coarse_coordinate(length: int, kept: int, device: torch.device):
    """Normalised k coordinate of the ``kept`` lowest frequencies of a grid."""
    full = torch.fft.fftshift(torch.fft.fftfreq(length, device=device) * 2.0)
    start = (length - kept) // 2
    return torch.fft.ifftshift(full[start : start + kept])


def _crop_spectrum(spectrum: torch.Tensor, kept: tuple[int, ...]) -> torch.Tensor:
    """Central block of an FFT-ordered spectrum, still in FFT order."""
    axes = tuple(range(spectrum.ndim))
    shifted = torch.fft.fftshift(spectrum, dim=axes)
    for axis, size in enumerate(kept):
        start = (spectrum.shape[axis] - size) // 2
        shifted = shifted.narrow(axis, start, size)
    return torch.fft.ifftshift(shifted, dim=axes)


def autofocus(
    image: torch.Tensor,
    transfer: SpiralTransfer,
    *,
    resolution: int = 96,
    window: int = 5,
    smoothing: int = 5,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deblur a spiral image without a measured field map.

    The field map is estimated on a coarse grid, since it is smooth, and the
    correction is then applied at full resolution. Each candidate frequency is a
    weighted sum of the same ``transfer.terms`` term images, so searching the
    whole band costs one transform per term rather than one per candidate. The
    chosen frequency is the one whose correction leaves least signal in
    quadrature with the object's own phase.

    Parameters
    ----------
    image
        Complex reconstructed image, 2D or 3D, with no leading axes.
    transfer
        Factorization from :func:`fit_transfer`.
    resolution
        Size of the grid the field map is estimated on, per axis. Raise it only
        if the field really does vary on a fine scale.
    window
        Width of the neighbourhood the focus metric is pooled over, in coarse
        voxels. Wider is steadier and blunter.
    smoothing
        Width of the neighbourhood the estimated field map is smoothed over,
        weighted by signal magnitude so empty regions follow their neighbours.
    stride
        Take every ``stride``-th tabulated frequency as a candidate.

    Returns
    -------
    corrected : torch.Tensor
        The deblurred image.
    field_map : torch.Tensor
        The estimated off-resonance in Hz, at the resolution of ``image``.

    Examples
    --------
    >>> import numpy as np, torch
    >>> from mrdistortion import ReadoutTiming, autofocus, fit_transfer
    >>> angle = np.linspace(0, 12 * np.pi, 800)
    >>> arm = np.stack([angle * np.cos(angle), angle * np.sin(angle)], axis=1)
    >>> timing = ReadoutTiming.from_trajectory(arm, duration=5e-3)
    >>> transfer = fit_transfer(timing, band=100.0, terms=5)
    >>> corrected, field = autofocus(torch.ones(32, 32, dtype=torch.complex64),
    ...                              transfer, resolution=16)
    >>> field.shape
    torch.Size([32, 32])
    """
    if not image.is_complex():
        raise ValueError("image must be complex")
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got {image.ndim} axes")

    axes = tuple(range(image.ndim))
    shape = tuple(image.shape)
    coarse = tuple(min(resolution, length) for length in shape)
    real_dtype = _real_for(image.dtype)

    spectrum = _crop_spectrum(torch.fft.fftn(image, dim=axes), coarse)
    coordinates = [
        _coarse_coordinate(length, kept, image.device).to(real_dtype)
        for length, kept in zip(shape, coarse, strict=True)
    ]

    terms = []
    for term in range(transfer.terms):
        working = spectrum.clone()
        rate = torch.as_tensor(complex(transfer.rates[term]), device=image.device)
        for axis, coordinate in enumerate(coordinates):
            view = [1] * len(coarse)
            view[axis] = coarse[axis]
            working *= torch.exp(rate * coordinate.reshape(view) ** 2)
        terms.append(torch.fft.ifftn(working, dim=axes))
    del working

    # A centre-out spiral reads the middle of k-space first, so a low-pass image
    # carries almost no off-resonance phase and serves as the object's own
    # reference. Without it a phase-sensitive metric would chase object phase.
    reference = torch.fft.ifftn(
        spectrum * _low_pass(coarse, image.device, 0.25), dim=axes
    )
    unwind = torch.polar(torch.ones_like(reference.real), -reference.angle())
    blurred = torch.fft.ifftn(spectrum, dim=axes)
    del spectrum, reference

    magnitude = _box_filter(blurred.abs().to(real_dtype) ** 2, window)
    floor = 1e-6 * magnitude.mean()
    best_metric = torch.full(
        coarse, float("inf"), device=image.device, dtype=real_dtype
    )
    best_frequency = torch.zeros(coarse, device=image.device, dtype=real_dtype)

    for index in range(0, len(transfer.frequencies), stride):
        candidate = torch.zeros_like(blurred)
        for term, volume in enumerate(terms):
            candidate += complex(transfer.weights[term, index]) * volume
        metric = _box_filter((candidate * unwind).imag ** 2, window) / (
            magnitude + floor
        )
        improved = metric < best_metric
        best_metric = torch.where(improved, metric, best_metric)
        best_frequency = torch.where(
            improved, float(transfer.frequencies[index]), best_frequency
        )
    del terms, unwind, best_metric, candidate

    weight = blurred.abs().to(real_dtype)
    field_map = _box_filter(best_frequency * weight, smoothing) / (
        _box_filter(weight, smoothing) + 1e-6 * weight.mean()
    )
    if coarse != shape:
        field_map = torch.nn.functional.interpolate(
            field_map[None, None],
            size=shape,
            mode="trilinear" if len(shape) == 3 else "bilinear",
            align_corners=False,
        )[0, 0]
    return deblur(image, field_map, transfer), field_map
