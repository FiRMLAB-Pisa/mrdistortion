"""Off-resonance deblurring of spiral readouts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mrdistortion import ReadoutTiming, autofocus, deblur, fit_transfer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def variable_density_arm(samples: int = 1200, power: float = 0.6) -> np.ndarray:
    """A centre-out arm whose radius grows as a fractional power of time."""
    time = np.linspace(0.0, 1.0, samples)
    radius = time**power
    angle = 40 * np.pi * time
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


@pytest.fixture(scope="module")
def timing() -> ReadoutTiming:
    return ReadoutTiming.from_trajectory(variable_density_arm(), duration=6e-3)


def phantom(size: int) -> np.ndarray:
    """Ellipses on a square grid."""
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    image = np.zeros((size, size))
    for row, column, height, width, value in (
        (0.0, 0.0, 0.70, 0.55, 1.0),
        (0.0, 0.0, 0.65, 0.50, -0.8),
        (-0.2, 0.0, 0.20, 0.30, 0.4),
        (0.3, -0.25, 0.15, 0.10, 0.6),
    ):
        inside = ((rows - row) / height) ** 2 + ((columns - column) / width) ** 2
        image[inside <= 1] += value
    return image


def blur(image: np.ndarray, timing: ReadoutTiming, field: np.ndarray):
    """Blur exactly, by summing over the distinct values of a quantised field map."""
    size = image.shape[0]
    axis = np.fft.fftfreq(size) * 2
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    disc = squared <= 1.0
    times = np.interp(np.clip(squared, 0, 1), timing.squared_radius, timing.times)
    ideal = np.fft.ifft2(np.fft.fft2(image) * disc)
    blurred = np.zeros_like(ideal)
    for value in np.unique(field):
        selected = np.fft.fft2(image * (field == value)) * disc
        phase = np.exp(2j * np.pi * value * timing.duration * times)
        blurred += np.fft.ifft2(selected * phase)
    return ideal, blurred


def relative(estimate, truth) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def test_out_in_arm_is_rejected() -> None:
    outward = variable_density_arm(600)
    out_in = np.concatenate([outward, outward[::-1]])
    with pytest.raises(ValueError, match="not monotonic"):
        ReadoutTiming.from_trajectory(out_in, duration=6e-3)


def test_inward_arm_is_accepted() -> None:
    inward = variable_density_arm(600)[::-1]
    derived = ReadoutTiming.from_trajectory(inward, duration=6e-3)
    assert derived.times[0] > derived.times[-1]


def test_more_terms_reduce_fit_error(timing: ReadoutTiming) -> None:
    errors = [
        fit_transfer(timing, band=120.0, terms=n).error(timing) for n in (2, 4, 6)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_constant_offset_is_corrected_to_the_fit_error(timing: ReadoutTiming) -> None:
    size = 128
    image = phantom(size)
    offset = 90.0
    ideal, blurred = blur(image, timing, np.full((size, size), offset))
    transfer = fit_transfer(timing, band=120.0, terms=6)

    corrected = deblur(
        torch.from_numpy(blurred),
        torch.full((size, size), offset, dtype=torch.float64),
        transfer,
    ).numpy()

    assert relative(blurred, ideal) > 0.2
    # A constant offset is a plain convolution, so nothing but the factorization
    # error stands between the correction and the truth.
    assert relative(corrected, ideal) < 5 * transfer.error(timing) + 1e-3


def test_varying_field_is_mostly_corrected(timing: ReadoutTiming) -> None:
    size = 128
    image = phantom(size)
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    smooth = 70 * (0.6 * columns + 0.4 * rows)
    levels = np.linspace(smooth.min(), smooth.max(), 24)
    field = levels[np.abs(smooth[..., None] - levels).argmin(-1)]

    ideal, blurred = blur(image, timing, field)
    transfer = fit_transfer(timing, band=80.0, terms=6)
    corrected = deblur(
        torch.from_numpy(blurred), torch.from_numpy(field), transfer
    ).numpy()

    assert relative(corrected, ideal) < 0.5 * relative(blurred, ideal)


def test_deblur_rejects_a_real_image(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=4)
    with pytest.raises(ValueError, match="complex"):
        deblur(torch.zeros(8, 8), torch.zeros(8, 8), transfer)


def test_deblur_rejects_a_mismatched_field_map(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=4)
    with pytest.raises(ValueError, match="does not match"):
        deblur(torch.zeros(8, 8, dtype=torch.complex64), torch.zeros(4, 4), transfer)


@requires_cuda
def test_cuda_matches_the_reference(timing: ReadoutTiming) -> None:
    size = 64
    torch.manual_seed(0)
    image = torch.randn(size, size, size, dtype=torch.complex64)
    field = torch.rand(size, size, size) * 200 - 100
    transfer = fit_transfer(timing, band=120.0, terms=6)

    reference = deblur(image, field, transfer, backend="fft")
    fused = deblur(image.cuda(), field.cuda(), transfer, backend="fft").cpu()
    assert (fused - reference).abs().max() < 1e-4 * reference.abs().max()


@requires_cuda
def test_convolution_backend_matches_the_transform(timing: ReadoutTiming) -> None:
    size = 64
    torch.manual_seed(0)
    image = torch.randn(size, size, size, dtype=torch.complex64, device="cuda")
    field = torch.rand(size, size, size, device="cuda") * 200 - 100
    transfer = fit_transfer(timing, band=120.0, terms=6)

    exact = deblur(image, field, transfer, backend="fft")
    truncated = deblur(image, field, transfer, backend="conv")
    assert (truncated - exact).abs().max() < 0.02 * exact.abs().max()


@requires_cuda
def test_peak_memory_does_not_grow_with_terms(timing: ReadoutTiming) -> None:
    size = 96
    image = torch.randn(size, size, size, dtype=torch.complex64, device="cuda")
    field = torch.zeros(size, size, size, device="cuda")

    peaks = []
    for terms in (4, 10):
        transfer = fit_transfer(timing, band=120.0, terms=terms)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        deblur(image, field, transfer, backend="fft")
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    assert peaks[1] <= 1.05 * peaks[0]


def structured_phantom(size: int) -> np.ndarray:
    """Ellipses carrying a smooth object phase, as a real acquisition would."""
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    image = np.zeros((size, size))
    for row, column, height, width, value in (
        (0.0, 0.0, 0.75, 0.60, 1.0),
        (-0.15, 0.0, 0.25, 0.35, 0.55),
        (0.32, -0.28, 0.13, 0.09, 0.8),
        (0.32, 0.28, 0.13, 0.09, 0.8),
    ):
        inside = ((rows - row) / height) ** 2 + ((columns - column) / width) ** 2
        image[inside <= 1] += value
    return image * np.exp(1j * 0.7 * (columns + 0.5 * rows))


def gradient_field(size: int, amplitude: float) -> np.ndarray:
    """A quantised shim-like field map, so the forward blur can be exact."""
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    smooth = amplitude * (0.7 * columns + 0.3 * rows)
    levels = np.linspace(smooth.min(), smooth.max(), 24)
    return levels[np.abs(smooth[..., None] - levels).argmin(-1)]


def test_autofocus_approaches_the_oracle(timing: ReadoutTiming) -> None:
    size = 128
    image = structured_phantom(size)
    field = gradient_field(size, 70.0)
    ideal, blurred = blur(image, timing, field)
    transfer = fit_transfer(timing, band=90.0, terms=6, frequencies=61)

    oracle = deblur(torch.from_numpy(blurred), torch.from_numpy(field), transfer)
    estimated, _ = autofocus(torch.from_numpy(blurred), transfer)

    before = relative(blurred, ideal)
    with_map = relative(oracle.numpy(), ideal)
    without = relative(estimated.numpy(), ideal)
    assert with_map < 0.6 * before
    # Not having a field map should cost little; the metric only has to find the
    # basin, not the exact frequency.
    assert without < 1.4 * with_map


def test_autofocus_rejects_a_batched_image(timing: ReadoutTiming) -> None:
    transfer = fit_transfer(timing, band=80.0, terms=4)
    with pytest.raises(ValueError, match="2D or 3D"):
        autofocus(torch.zeros(2, 8, 8, 8, dtype=torch.complex64), transfer)


@requires_cuda
def test_autofocus_runs_on_cuda(timing: ReadoutTiming) -> None:
    size = 128
    image = structured_phantom(size)
    field = gradient_field(size, 70.0)
    ideal, blurred = blur(image, timing, field)
    transfer = fit_transfer(timing, band=90.0, terms=6, frequencies=41)

    on_host, _ = autofocus(torch.from_numpy(blurred), transfer)
    on_device, _ = autofocus(
        torch.from_numpy(blurred).to(torch.complex64).cuda(), transfer
    )
    assert relative(on_device.cpu().numpy(), ideal) < 1.05 * relative(
        on_host.numpy(), ideal
    )
