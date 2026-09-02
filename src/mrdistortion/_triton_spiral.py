"""Separable convolution of a complex volume, one axis at a time.

Each term of the off-resonance factorization is a k-space factor
``exp(alpha * |k|^2 / kmax^2)``, which factors over the axes. Its image-domain
action is therefore a circular convolution along each axis in turn with a short
kernel, which is cheaper than transforming the whole volume when the kernel is
narrow. The kernel width follows from the term's rate: a rate with a large
imaginary part spreads energy further and needs more taps.

The convolution is circular because the FFT-domain multiply it reproduces is
circular, so the two backends of :func:`mrdistortion.deblur` agree exactly up to
tap truncation.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["accumulate_weighted", "separable_convolve"]


@triton.jit
def _convolve_axis_kernel(
    source,
    destination,
    taps,
    length,
    inner,
    HALF: tl.constexpr,
    BLOCK_LENGTH: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    """Circular convolution along the middle axis of an (outer, length, inner) view."""
    outer_id = tl.program_id(0)
    length_id = tl.program_id(1)
    inner_id = tl.program_id(2)

    along = length_id * BLOCK_LENGTH + tl.arange(0, BLOCK_LENGTH)
    across = inner_id * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    live = (along < length)[:, None] & (across < inner)[None, :]

    real = tl.zeros((BLOCK_LENGTH, BLOCK_INNER), dtype=tl.float32)
    imaginary = tl.zeros((BLOCK_LENGTH, BLOCK_INNER), dtype=tl.float32)
    for tap in tl.static_range(2 * HALF + 1):
        position = (along - tap + HALF + length) % length
        offset = (outer_id * length + position[:, None]) * inner + across[None, :]
        sample_real = tl.load(source + 2 * offset, mask=live, other=0.0)
        sample_imaginary = tl.load(source + 2 * offset + 1, mask=live, other=0.0)
        tap_real = tl.load(taps + 2 * tap)
        tap_imaginary = tl.load(taps + 2 * tap + 1)
        real += sample_real * tap_real - sample_imaginary * tap_imaginary
        imaginary += sample_real * tap_imaginary + sample_imaginary * tap_real

    offset = (outer_id * length + along[:, None]) * inner + across[None, :]
    tl.store(destination + 2 * offset, real, mask=live)
    tl.store(destination + 2 * offset + 1, imaginary, mask=live)


def _taps(rate: complex, length: int, device: torch.device, tolerance: float):
    """Shortest centred kernel holding all but ``tolerance`` of the term's energy."""
    coordinate = torch.fft.fftfreq(length, device=device, dtype=torch.float32) * 2.0
    factor = torch.exp(torch.as_tensor(rate, device=device) * coordinate**2)
    kernel = torch.fft.ifft(factor)
    kernel = torch.roll(kernel, length // 2)
    power = kernel.abs() ** 2
    centre = length // 2
    total = power.sum()
    running = power[centre].clone()
    half = 0
    while running < (1.0 - tolerance) * total and half < centre - 1:
        half += 1
        running = running + power[centre - half] + power[centre + half]
    return kernel[centre - half : centre + half + 1].contiguous(), half


def _convolve_axis(volume: torch.Tensor, rate: complex, axis: int, tolerance: float):
    """Apply one axis factor of ``exp(rate * u)`` to a complex volume.

    A contiguous tensor reshapes to ``(outer, length, inner)`` about any axis
    without moving data, so no axis is ever transposed.
    """
    length = volume.shape[axis]
    kernel, half = _taps(rate, length, volume.device, tolerance)
    if 2 * half + 1 >= length:
        raise ValueError("kernel is not narrower than the axis; use backend='fft'")

    volume = volume.contiguous()
    outer = math.prod(volume.shape[:axis])
    inner = math.prod(volume.shape[axis + 1 :])
    flat = volume.reshape(outer, length, inner)
    result = torch.empty_like(flat)

    if inner == 1:
        block_inner, block_length = 1, 256
    else:
        block_inner = min(triton.next_power_of_2(inner), 128)
        block_length = max(1, 256 // block_inner)
    grid = (outer, triton.cdiv(length, block_length), triton.cdiv(inner, block_inner))
    _convolve_axis_kernel[grid](
        torch.view_as_real(flat),
        torch.view_as_real(result),
        torch.view_as_real(kernel),
        length,
        inner,
        HALF=half,
        BLOCK_LENGTH=block_length,
        BLOCK_INNER=block_inner,
    )
    return result.reshape(volume.shape)


def separable_convolve(
    image: torch.Tensor,
    rate: complex,
    axes: tuple[int, ...],
    *,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Apply ``exp(rate * |k|^2 / kmax^2)`` as a convolution along each axis.

    Parameters
    ----------
    image
        Complex tensor whose ``axes`` span the image grid.
    rate
        Complex rate of one factorization term.
    axes
        Spatial axes to convolve along.
    tolerance
        Fraction of kernel energy the truncation may discard per axis.

    Returns
    -------
    torch.Tensor
        The convolved image, same shape and dtype as ``image``.
    """
    if not image.is_complex():
        raise ValueError("image must be complex")
    working = image
    for axis in axes:
        # The kernel walks the second-to-last axis, so a one-dimensional image
        # needs a trailing axis to stride over.
        if working.ndim < 2:
            working = _convolve_axis(working.unsqueeze(-1), rate, 0, tolerance)
            working = working.squeeze(-1)
        else:
            working = _convolve_axis(working, rate, axis, tolerance)
    return working


@triton.jit
def _accumulate_kernel(
    result,
    working,
    field,
    table,
    tabulated,
    first,
    spacing,
    count,
    BLOCK: tl.constexpr,
):
    """Accumulate ``weight(field) * working``, interpolating the weight in place."""
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offset < count

    position = (tl.load(field + offset, mask=live, other=0.0) - first) / spacing
    position = tl.minimum(tl.maximum(position, 0.0), tabulated - 1.0)
    index = tl.minimum(position.to(tl.int32), tabulated - 2)
    fraction = position - index.to(tl.float32)

    lower_real = tl.load(table + 2 * index)
    lower_imaginary = tl.load(table + 2 * index + 1)
    upper_real = tl.load(table + 2 * index + 2)
    upper_imaginary = tl.load(table + 2 * index + 3)
    weight_real = lower_real + fraction * (upper_real - lower_real)
    weight_imaginary = lower_imaginary + fraction * (upper_imaginary - lower_imaginary)

    value_real = tl.load(working + 2 * offset, mask=live, other=0.0)
    value_imaginary = tl.load(working + 2 * offset + 1, mask=live, other=0.0)
    total_real = tl.load(result + 2 * offset, mask=live, other=0.0)
    total_imaginary = tl.load(result + 2 * offset + 1, mask=live, other=0.0)

    tl.store(
        result + 2 * offset,
        total_real + weight_real * value_real - weight_imaginary * value_imaginary,
        mask=live,
    )
    tl.store(
        result + 2 * offset + 1,
        total_imaginary + weight_real * value_imaginary + weight_imaginary * value_real,
        mask=live,
    )


def accumulate_weighted(
    result: torch.Tensor,
    working: torch.Tensor,
    field_map: torch.Tensor,
    table: torch.Tensor,
    first: float,
    spacing: float,
) -> None:
    """Add one term's contribution, weighting it by the interpolated field map.

    The weight volume is never materialised, so a term costs one pass over the
    working volume rather than three.

    Parameters
    ----------
    result
        Complex accumulator, updated in place.
    working
        Complex contribution of this term, broadcast over any leading axes.
    field_map
        Off-resonance in Hz, shaped like the trailing axes of ``result``.
    table
        This term's weight at each tabulated frequency.
    first, spacing
        First tabulated frequency and the spacing between them, in Hz.
    """
    field = field_map.expand(result.shape).contiguous()
    count = result.numel()
    grid = (triton.cdiv(count, 1024),)
    _accumulate_kernel[grid](
        torch.view_as_real(result),
        torch.view_as_real(working.contiguous()),
        field,
        torch.view_as_real(table.contiguous()),
        table.numel(),
        float(first),
        float(spacing),
        count,
        BLOCK=1024,
    )
