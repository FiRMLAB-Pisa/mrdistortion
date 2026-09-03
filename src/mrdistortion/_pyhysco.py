"""Susceptibility distortion correction from an opposite-phase-encoding pair.

Two images of the same anatomy acquired with reversed phase encoding are
distorted in opposite directions along that axis. The field that displaced them
is what makes one the mirror of the other, so it can be recovered by finding the
map that brings the pair into register, and the images unwarped with it.

That estimation is PyHySCO's. PyHySCO is torch-based throughout, so this module
hands it tensors and takes tensors back: nothing is written to or read from
disk, and the caller's arrays never leave memory.

PyHySCO is GPL-3.0-only and this package is MIT. MIT is GPL-compatible and the
obligation attaches to distributing the combination, not to writing the import,
so PyHySCO is an optional extra that is never bundled: it is imported inside the
call, and its absence is an informative error rather than an import failure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = ["SusceptibilityCorrection", "correct_susceptibility"]

Optimizer = Literal["gauss-newton", "lbfgs", "admm"]

# How PyHySCO orders axes internally: phase encoding last, and the inverse that
# puts a result back in the caller's order.
_PERMUTATIONS: dict[tuple[int, int], tuple[list[int], list[int]]] = {
    (2, 1): ([1, 0], [1, 0]),
    (2, 2): ([0, 1], [0, 1]),
    (3, 1): ([2, 1, 0], [2, 1, 0]),
    (3, 2): ([2, 0, 1], [1, 2, 0]),
}


@dataclass(frozen=True)
class SusceptibilityCorrection:
    """Result of correcting an opposite-phase-encoding pair.

    Parameters
    ----------
    field_map
        The estimated displacement field, in the caller's axis order. Its
        length along the phase-encoding axis is one greater than the input's,
        because the field is defined on cell edges.
    blip_up, blip_down
        The two corrected images, in the caller's axis order.
    """

    field_map: torch.Tensor
    blip_up: torch.Tensor
    blip_down: torch.Tensor


def _require_pyhysco() -> dict[str, Any]:
    """Import PyHySCO, or explain how to get it."""
    try:
        from EPI_MRI.EPIMRIDistortionCorrection import (
            DataObject,
            EPIMRIDistortionCorrection,
        )
        from EPI_MRI.ImageModels import Interp1D
        from EPI_MRI.LinearOperators import myLaplacian2D, myLaplacian3D
        from EPI_MRI.utils import m_plus, normalize
        from optimization.ADMM import ADMM
        from optimization.GaussNewton import GaussNewton
        from optimization.LBFGS import LBFGS
    except ImportError as error:  # pragma: no cover - exercised only without it
        raise ImportError(
            "susceptibility correction needs PyHySCO, which is GPL-3.0-only and "
            "so is not a dependency of this package: pip install PyHySCO"
        ) from error
    return {
        "DataObject": DataObject,
        "correction": EPIMRIDistortionCorrection,
        "Interp1D": Interp1D,
        "laplacian": {2: myLaplacian2D, 3: myLaplacian3D},
        "m_plus": m_plus,
        "normalize": normalize,
        "optimizers": {"gauss-newton": GaussNewton, "lbfgs": LBFGS, "admm": ADMM},
    }


def _domain(
    shape: tuple[int, ...],
    voxel_size: tuple[float, ...],
    permutation: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """Build the image domain, discretisation and cell size PyHySCO expects."""
    extent = torch.zeros(2 * len(shape), dtype=dtype, device=device)
    extent[1::2] = torch.tensor(
        [size * count for size, count in zip(voxel_size, shape, strict=True)],
        dtype=dtype,
        device=device,
    )
    omega = torch.zeros_like(extent)
    for axis, source in enumerate(permutation):
        omega[2 * axis : 2 * axis + 2] = extent[2 * source : 2 * source + 2]
    counts = torch.tensor(shape, dtype=torch.int, device=device)[permutation]
    spacing = (omega[1::2] - omega[:-1:2]) / counts
    return omega, counts, spacing


def _data_object(
    blip_up: torch.Tensor,
    blip_down: torch.Tensor,
    voxel_size: tuple[float, ...],
    phase_encoding_direction: int,
    parts: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """Populate PyHySCO's ``DataObject`` from tensors instead of file paths.

    ``DataObject.__init__`` reaches straight for ``load_data``, which accepts
    only string paths, so the object is built without running that constructor
    and every attribute it would have set is set here instead.
    """
    key = (blip_up.ndim, phase_encoding_direction)
    if key not in _PERMUTATIONS:
        raise ValueError(
            f"no PyHySCO axis order for a {blip_up.ndim}D image with phase "
            f"encoding along axis {phase_encoding_direction}; images must be 2D "
            "or 3D and the axis must be 1 or 2"
        )
    permutation, inverse = _PERMUTATIONS[key]

    holder = parts["DataObject"].__new__(parts["DataObject"])
    holder.device = device
    holder.dtype = dtype
    holder.omega, holder.m, holder.h = _domain(
        tuple(blip_up.shape), voxel_size, permutation, dtype=dtype, device=device
    )
    holder.p = inverse
    up = blip_up.to(device=device, dtype=dtype).permute(permutation)
    down = blip_down.to(device=device, dtype=dtype).permute(permutation)
    holder.im1, holder.im2 = up, down
    normalised_up, normalised_down = parts["normalize"](up, down)
    holder.I1 = parts["Interp1D"](
        normalised_up, holder.omega, holder.m, dtype=dtype, device=device
    )
    holder.I2 = parts["Interp1D"](
        normalised_down, holder.omega, holder.m, dtype=dtype, device=device
    )
    return holder


def correct_susceptibility(
    blip_up: torch.Tensor,
    blip_down: torch.Tensor,
    *,
    voxel_size: tuple[float, ...],
    phase_encoding_direction: int = 1,
    alpha: float = 300.0,
    beta: float = 1e-4,
    optimizer: Optimizer = "gauss-newton",
    max_iter: int = 10,
    device: torch.device | str | None = None,
) -> SusceptibilityCorrection:
    """Correct susceptibility distortion from an opposite-phase-encoding pair.

    Parameters
    ----------
    blip_up, blip_down
        The two images, same shape, phase encoding reversed between them.
    voxel_size
        Voxel size along each axis, in millimetres, in the images' own axis
        order.
    phase_encoding_direction
        Which axis is the phase-encoding one, counting from 1.
    alpha
        Weight on the smoothness of the estimated field.
    beta
        Weight on the constraint keeping the transformation invertible.
    optimizer
        Which of PyHySCO's optimizers to run.
    max_iter
        Iteration cap for the optimizer.
    device
        Where to compute. Defaults to the images' device.

    Returns
    -------
    SusceptibilityCorrection
        The estimated field and both corrected images, as tensors in the
        caller's axis order.

    Raises
    ------
    ImportError
        If PyHySCO is not installed. It is GPL-3.0-only and therefore an
        optional extra rather than a dependency.
    ValueError
        If the images disagree in shape, or the phase-encoding axis is one
        PyHySCO has no ordering for.
    """
    if blip_up.shape != blip_down.shape:
        raise ValueError(
            f"the pair must have one shape, got {tuple(blip_up.shape)} and "
            f"{tuple(blip_down.shape)}"
        )
    if len(voxel_size) != blip_up.ndim:
        raise ValueError(
            f"voxel_size has {len(voxel_size)} entries for a {blip_up.ndim}D image"
        )
    parts = _require_pyhysco()
    resolved = torch.device(device) if device is not None else blip_up.device

    holder = _data_object(
        blip_up,
        blip_down,
        voxel_size,
        phase_encoding_direction,
        parts,
        dtype=torch.float64,
        device=resolved,
    )
    # Both the smoothness term and the initialiser's own smoothing are written
    # per dimension, and the defaults are the three-dimensional ones.
    try:
        objective = parts["correction"](
            holder, alpha, beta, regularizer=parts["laplacian"][blip_up.ndim]
        )
        initial = objective.initialize(blur_result=blip_up.ndim == 3)
    except IndexError as error:
        if blip_up.ndim == 2:
            raise NotImplementedError(
                "this PyHySCO cannot correct a two-dimensional pair: its own "
                "two-dimensional regulariser builds a three-dimensional "
                "transform and indexes an axis that is not there. Pass the "
                "volume the slice came from instead."
            ) from error
        raise

    try:
        driver = parts["optimizers"][optimizer]
    except KeyError:
        raise ValueError(f"unknown optimizer {optimizer!r}") from None
    solver = driver(objective, max_iter=max_iter, verbose=False, path=None)
    # PyHySCO logs every iteration to a file beside the working directory.
    # The history it also keeps in memory is the part worth having.
    solver.log.log_file = os.devnull
    solver.run_correction(initial)

    # PyHySCO's own apply_correction writes NIfTIs on the way past; the reshape
    # and permute it does around them are all that is wanted here.
    field = solver.Bc.detach().reshape(list(parts["m_plus"](holder.m)))
    shape = list(holder.m)
    return SusceptibilityCorrection(
        field_map=field.permute(holder.p),
        blip_up=objective.corr1.reshape(shape).permute(holder.p),
        blip_down=objective.corr2.reshape(shape).permute(holder.p),
    )
