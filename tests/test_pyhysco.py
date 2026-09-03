"""Susceptibility correction from an opposite-phase-encoding pair."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from mrdistortion import correct_susceptibility

needs_pyhysco = pytest.mark.skipif(
    importlib.util.find_spec("EPI_MRI") is None,
    reason="PyHySCO is GPL-3.0-only and an optional extra",
)


def displaced_pair(shape=(24, 32, 32), amplitude=3.0):
    """A phantom and the two images a field would displace it into."""
    axes = [torch.linspace(-1, 1, n, dtype=torch.float64) for n in shape]
    rows, columns, planes = torch.meshgrid(*axes, indexing="ij")
    phantom = (
        ((rows / 0.7) ** 2 + (columns / 0.8) ** 2 + (planes / 0.8) ** 2) < 1
    ).double() * (1 + 0.5 * torch.sin(5 * rows) * torch.cos(4 * columns))
    shift = amplitude * torch.exp(-(columns**2 + planes**2) / 0.3)
    index = torch.arange(shape[0], dtype=torch.float64)[:, None, None]

    def warp(image, offset):
        source = (index + offset).clamp(0, shape[0] - 1)
        lower = source.floor().long()
        upper = (lower + 1).clamp(max=shape[0] - 1)
        fraction = source - lower
        return (
            image.gather(0, lower.expand_as(image)) * (1 - fraction)
            + image.gather(0, upper.expand_as(image)) * fraction
        )

    return warp(phantom, shift), warp(phantom, -shift), shift


def test_pair_must_share_a_shape() -> None:
    with pytest.raises(ValueError, match="one shape"):
        correct_susceptibility(
            torch.zeros(4, 4, 4), torch.zeros(4, 4, 5), voxel_size=(1.0, 1.0, 1.0)
        )


def test_voxel_size_must_cover_every_axis() -> None:
    with pytest.raises(ValueError, match="voxel_size has"):
        correct_susceptibility(
            torch.zeros(4, 4, 4), torch.zeros(4, 4, 4), voxel_size=(1.0, 1.0)
        )


def test_phase_encoding_axis_must_be_one_pyhysco_orders() -> None:
    with pytest.raises(ValueError, match="no PyHySCO axis order"):
        correct_susceptibility(
            torch.zeros(4, 4, 4),
            torch.zeros(4, 4, 4),
            voxel_size=(1.0, 1.0, 1.0),
            phase_encoding_direction=3,
        )


@needs_pyhysco
def test_correction_returns_tensors_in_the_callers_orientation() -> None:
    up, down, _ = displaced_pair()
    result = correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=6)
    assert result.blip_up.shape == up.shape
    assert result.blip_down.shape == up.shape
    # The field lives on cell edges, so it is one longer along phase encoding.
    assert result.field_map.shape == (up.shape[0] + 1, *up.shape[1:])


@needs_pyhysco
def test_correction_recovers_the_imposed_displacement() -> None:
    up, down, shift = displaced_pair()
    result = correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=8)
    assert result.field_map.abs().max() == pytest.approx(shift.max(), rel=0.4)
    before = (up - down).abs().mean()
    after = (result.blip_up - result.blip_down).abs().mean()
    assert after < before


@needs_pyhysco
def test_correction_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    up, down, _ = displaced_pair(shape=(16, 24, 24))
    correct_susceptibility(up, down, voxel_size=(1.0, 1.0, 1.0), max_iter=4)
    assert list(tmp_path.rglob("*")) == []


@needs_pyhysco
def test_unknown_optimizer_is_rejected() -> None:
    up, down, _ = displaced_pair(shape=(16, 24, 24))
    with pytest.raises(ValueError, match="unknown optimizer"):
        correct_susceptibility(
            up, down, voxel_size=(1.0, 1.0, 1.0), optimizer="newton-raphson"
        )


@needs_pyhysco
def test_two_dimensional_pair_reports_the_upstream_limit() -> None:
    """PyHySCO 0.0.4's own 2D regulariser builds a 3D transform and fails."""
    up, down, _ = displaced_pair(shape=(24, 32, 32))
    with pytest.raises(NotImplementedError, match="two-dimensional"):
        correct_susceptibility(
            up[:, :, 16].contiguous(),
            down[:, :, 16].contiguous(),
            voxel_size=(1.0, 1.0),
            max_iter=6,
        )
