"""Off-resonance field maps from single-echo phase."""

from __future__ import annotations

import math

import pytest
import torch

from mrdistortion import field_map_from_phase

ECHO_TIME = 0.7e-3


def acquisition(size: int = 48, peak: float = 150.0, coils: int = 6):
    """Coil images of a phantom sitting in a localised off-resonance bump."""
    axes = [torch.linspace(-1, 1, size) for _ in range(3)]
    rows, columns, planes = torch.meshgrid(*axes, indexing="ij")
    inside = (rows**2 + columns**2 + planes**2) < 0.8**2
    phantom = inside.float() * (1.0 + 0.3 * torch.sin(5 * rows))

    # Finer than the sensitivities, so it survives the calibration low-pass.
    field = peak * torch.exp(
        -(((rows - 0.25) ** 2 + (columns + 0.2) ** 2 + planes**2) / 0.02)
    )
    accrued = torch.polar(torch.ones_like(field), 2 * math.pi * field * ECHO_TIME)

    centres = torch.tensor(
        [
            [1.4, 0, 0],
            [-1.4, 0, 0],
            [0, 1.4, 0],
            [0, -1.4, 0],
            [0, 0, 1.4],
            [0, 0, -1.4],
        ]
    )[:coils]
    images = []
    for centre in centres:
        distance = (
            (rows - centre[0]) ** 2
            + (columns - centre[1]) ** 2
            + (planes - centre[2]) ** 2
        )
        sensitivity = torch.polar(torch.exp(-distance / 4.0), 0.4 * distance)
        images.append(phantom * sensitivity * accrued)
    return torch.stack(images), field, inside


def test_recovers_a_localised_field() -> None:
    coils, field, inside = acquisition()
    estimate = field_map_from_phase(coils, echo_time=ECHO_TIME, smoothing=1.0)

    hot = inside & (field > 0.5 * field.max())
    assert estimate[hot].mean() == pytest.approx(field[hot].mean(), rel=0.35)
    quiet = inside & (field < 1.0)
    assert estimate[quiet].abs().mean() < 0.15 * field.max()


def test_smoothing_trades_detail_for_quiet() -> None:
    coils, field, inside = acquisition()
    hot = inside & (field > 0.5 * field.max())
    sharp = field_map_from_phase(coils, echo_time=ECHO_TIME, smoothing=0.0)
    blunt = field_map_from_phase(coils, echo_time=ECHO_TIME, smoothing=6.0)
    # Over-smoothing spreads the bump out, so its peak comes back lower.
    assert blunt[hot].mean() < sharp[hot].mean()


def test_unambiguous_range_is_set_by_echo_time() -> None:
    coils, _, inside = acquisition(peak=100.0)
    estimate = field_map_from_phase(coils, echo_time=ECHO_TIME, smoothing=1.0)
    assert estimate[inside].abs().max() < 1.0 / (2 * ECHO_TIME)


def test_rejects_a_real_image() -> None:
    with pytest.raises(ValueError, match="complex"):
        field_map_from_phase(torch.zeros(4, 8, 8, 8), echo_time=ECHO_TIME)


def test_rejects_a_missing_coil_axis() -> None:
    with pytest.raises(ValueError, match="three spatial axes"):
        field_map_from_phase(
            torch.zeros(8, 8, 8, dtype=torch.complex64), echo_time=1e-3
        )


def test_rejects_a_nonpositive_echo_time() -> None:
    with pytest.raises(ValueError, match="echo_time"):
        field_map_from_phase(
            torch.zeros(2, 8, 8, 8, dtype=torch.complex64), echo_time=0.0
        )
