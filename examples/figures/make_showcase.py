"""Build the figures the README shows.

Run it as ``python examples/figures/make_showcase.py``.

Two of the three columns use acquired data that is not in this repository:

* gradient nonlinearity — a GE body gradient coil's own coefficient table, and
  the same volume corrected by the vendor's Orchestra ``GradwarpCorrector``, so
  the correction here can be put beside the one it has to match. Neither the
  table nor the vendor code may be redistributed; the images are what is shown.
* susceptibility — a spin-echo EPI pair from OpenNeuro ds003653, acquired with
  the phase encoding reversed between them.

Set ``GRADWARP_DATA`` and ``EPI_DATA`` to where those live. Without them the
script falls back to simulating both, and says so.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _brainweb import brain_and_field

import mrdistortion as mrd

ORDER = 3
SHAPE = (48, 64, 64)


def generic_coil(third_order: float = -0.06) -> mrd.GradientCoefficients:
    """A generic coil, given by how far it departs from being linear.

    The coefficients describe the departure, not the whole field: an all-zero
    table is a perfectly linear gradient and corrects to the identity. Each
    axis is stated in its own rotated frame, which is why every axis carries
    the same ``(n, m)``.
    """
    alpha = np.zeros((3, ORDER + 1, ORDER + 1))
    beta = np.zeros_like(alpha)
    for axis in range(3):
        alpha[axis, 3, 0] = third_order
    return mrd.GradientCoefficients(
        basis="normalized", alpha=alpha, beta=beta, reference_radius_mm=250.0
    )


def grid_phantom(shape: tuple[int, int, int], spacing: int = 12) -> np.ndarray:
    """A lattice, so the geometry of the correction is visible."""
    volume = np.zeros(shape)
    for axis in range(3):
        index = np.arange(shape[axis])
        on = (index % spacing) < 2
        volume += np.moveaxis(np.broadcast_to(on.reshape(-1, 1, 1), shape), 0, axis)
    return np.clip(volume, 0, 1)


def variable_density_arm(samples: int = 1500) -> np.ndarray:
    """One centre-out arm whose radius grows as a fractional power of time."""
    time = np.linspace(0.0, 1.0, samples)
    radius = time**0.6
    angle = 40 * np.pi * time
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def displaced_pair(shape=SHAPE, amplitude: float = 4.0):
    """A phantom and the pair a field would displace it into."""
    axes = [torch.linspace(-1, 1, n, dtype=torch.float64) for n in shape]
    rows, columns, planes = torch.meshgrid(*axes, indexing="ij")
    phantom = (
        ((rows / 0.75) ** 2 + (columns / 0.8) ** 2 + (planes / 0.8) ** 2) < 1
    ).double() * (1 + 0.4 * torch.sin(6 * rows) * torch.cos(5 * columns))

    # An air cavity's field: compact, and strongest where the anatomy ends.
    shift = amplitude * torch.exp(-((columns - 0.35) ** 2 + planes**2) / 0.15)
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


GRADWARP_DATA = Path(os.environ.get("GRADWARP_DATA", "/nonexistent"))
EPI_DATA = Path(os.environ.get("EPI_DATA", "/nonexistent"))
CASE = "simple_cartesian_3d"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})


def geometry_from_corners(first, shape, last=None):
    """The image geometry Orchestra states as three corner points."""
    lower_left, upper_left, upper_right = first
    vectors = [upper_left - lower_left, upper_right - upper_left]
    fov = [float(np.linalg.norm(vector)) for vector in vectors]
    directions = [vector / length for vector, length in zip(vectors, fov, strict=True)]
    spacing = [fov[axis] / shape[axis] for axis in range(2)]
    center = lower_left.copy()
    for axis in range(2):
        center = center + directions[axis] * spacing[axis] * (shape[axis] - 1) / 2
    if last is not None:
        through = last[0] - lower_left
        step = np.linalg.norm(through) / (shape[2] - 1)
        normal = through / np.linalg.norm(through)
        directions.append(normal)
        fov.append(float(step * shape[2]))
        center = center + normal * step * (shape[2] - 1) / 2
    return mrd.ImageGeometry(shape, tuple(fov), np.stack(directions, axis=1), center)


def gradwarp_case():
    """The acquired volume, Orchestra's correction, and this package's."""
    archive = np.load(GRADWARP_DATA / f"{CASE}_reference.npz")
    image, reference = archive["image"], archive["reference"]
    geometry = geometry_from_corners(
        archive["first_corners"], image.shape, archive["last_corners"]
    )
    correct = mrd.Gradunwarp.from_file(
        GRADWARP_DATA / f"{CASE}_coefficients.dat", geometry
    )
    return image, reference, correct(image)


def gradient_column():
    if GRADWARP_DATA.exists():
        # A uniform phantom cannot show a geometric distortion, so what the
        # column shows is the displacement the coil's own coefficients imply.
        # How well it matches the vendor is figures/gradwarp_vs_orchestra.png.
        archive = np.load(GRADWARP_DATA / f"{CASE}_reference.npz")
        image = archive["image"]
        geometry = geometry_from_corners(
            archive["first_corners"], image.shape, archive["last_corners"]
        )
        correct = mrd.Gradunwarp.from_file(
            GRADWARP_DATA / f"{CASE}_coefficients.dat", geometry
        )
        grid = correct.sampling_grid()
        index = np.stack(
            np.meshgrid(*[np.arange(n) for n in image.shape], indexing="ij"), axis=-1
        )
        spacing = np.array(geometry.fov_mm) / np.array(image.shape)
        displacement = np.linalg.norm((grid - index) * spacing, axis=-1)
        middle = image.shape[2] // 2
        print(
            f"real coil: displacement median {np.median(displacement):.2f} mm, "
            f"max {displacement.max():.2f} mm"
        )
        return (
            (
                "gradient nonlinearity",
                "acquired",
                f"displacement, 0-{displacement.max():.0f} mm",
            ),
            (np.rot90(image[:, :, middle]), np.rot90(displacement[:, :, middle])),
            None,
        )
    size = 96
    geometry = mrd.ImageGeometry(
        shape=(size,) * 3,
        fov_mm=(400.0,) * 3,
        direction=np.eye(3),
        center_mm=(0.0, 0.0, 0.0),
    )
    warp = mrd.Gradunwarp(generic_coil(-0.16), geometry)
    lattice = grid_phantom((size,) * 3, spacing=10)
    middle = size // 2
    return (
        ("gradient nonlinearity (simulated)", "acquired", "corrected"),
        (np.rot90(warp(lattice)[:, :, middle]), np.rot90(lattice[:, :, middle])),
        {"cmap": "gray"},
    )


def spiral_column():
    """A BrainWeb slice blurred by the field its own tissue makes."""
    size, readout = 256, 20e-3
    truth, field, _ = brain_and_field(slice_index=60, size=size)
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    truth = truth * np.exp(1j * 0.7 * (columns + 0.5 * rows))
    timing = mrd.ReadoutTiming.from_trajectory(variable_density_arm(), duration=readout)
    transfer = mrd.fit_transfer(timing, band=260.0, terms=56)
    axis = np.fft.fftfreq(size) * 2
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    disc = squared <= 1.0
    times = np.interp(np.clip(squared, 0, 1), timing.squared_radius, timing.times)
    levels = np.linspace(field.min(), field.max(), 200)
    quantised = levels[np.abs(field[..., None] - levels).argmin(-1)]
    blurred = np.zeros(truth.shape, complex)
    for value in np.unique(quantised):
        blurred += np.fft.ifft2(
            np.fft.fft2(truth * (quantised == value))
            * disc
            * np.exp(2j * np.pi * value * readout * times)
        )
    corrected = mrd.deblur(
        torch.from_numpy(blurred), torch.from_numpy(quantised), transfer
    ).numpy()
    top = np.percentile(np.abs(blurred), 99.5)
    return (
        ("spiral off-resonance", "off resonance", "deblurred"),
        (np.abs(blurred), np.abs(corrected)),
        {"cmap": "gray", "vmin": 0, "vmax": top},
    )


def epi_column():
    if EPI_DATA.exists():
        archive = np.load(EPI_DATA / "corrected.npz")
        acquired, corrected = archive["ap"], archive["up"]
        middle = acquired.shape[2] // 2
        scale = lambda x: x / np.percentile(x, 99.5)
        return (
            ("susceptibility", "blip up", "corrected"),
            (
                np.rot90(scale(acquired[:, :, middle])),
                np.rot90(scale(corrected[:, :, middle])),
            ),
            {"cmap": "gray", "vmin": 0, "vmax": 1.0},
        )
    up, down, _ = displaced_pair()
    result = mrd.correct_susceptibility(
        up, down, voxel_size=(1.0, 1.0, 1.0), alpha=50.0, max_iter=30
    )
    middle = up.shape[2] // 2
    scale = lambda x: x / np.percentile(x, 99.5)
    return (
        ("susceptibility (simulated)", "blip up", "corrected"),
        (scale(up[:, :, middle].numpy()), scale(result.blip_up[:, :, middle].numpy())),
        {"cmap": "gray", "vmin": 0, "vmax": 1.0},
    )


def showcase() -> None:
    columns = [gradient_column(), spiral_column(), epi_column()]
    figure, axes = plt.subplots(2, len(columns), figsize=(2.6 * len(columns), 5.6))
    for index, ((heading, upper, lower), images, style) in enumerate(columns):
        for row, (image, label) in enumerate(zip(images, (upper, lower), strict=True)):
            axis = axes[row, index]
            if style is None:
                axis.imshow(
                    image,
                    cmap="gray" if row == 0 else "magma",
                    vmax=np.percentile(image, 99.5),
                    vmin=0,
                )
            else:
                axis.imshow(image, **style)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_ylabel(label, fontsize=8)
            if row == 0:
                axis.set_title(heading, fontsize=9, pad=6)
    figure.tight_layout(pad=0.6)
    figure.savefig("figures/showcase.png", dpi=140, bbox_inches="tight")
    print("wrote figures/showcase.png")


def against_orchestra() -> None:
    """The vendor's correction and this one, on the same acquired volume."""
    if not GRADWARP_DATA.exists():
        print("GRADWARP_DATA not set; skipping the Orchestra comparison")
        return
    image, reference, ours = gradwarp_case()
    flat = (reference.astype(np.float64).ravel(), ours.astype(np.float64).ravel())
    print(
        f"vs Orchestra: correlation {np.corrcoef(*flat)[0, 1]:.6f}, "
        f"nrmse {np.linalg.norm(flat[0] - flat[1]) / np.linalg.norm(flat[0]):.5f}"
    )

    middle = image.shape[2] // 2
    top = np.percentile(image[:, :, middle], 99.5)
    panels = (
        (image[:, :, middle], "acquired", {"cmap": "gray", "vmin": 0, "vmax": top}),
        (
            reference[:, :, middle],
            "Orchestra",
            {"cmap": "gray", "vmin": 0, "vmax": top},
        ),
        (ours[:, :, middle], "mrdistortion", {"cmap": "gray", "vmin": 0, "vmax": top}),
        (
            (ours - reference)[:, :, middle],
            "difference, x20",
            {"cmap": "gray", "vmin": -top / 20, "vmax": top / 20},
        ),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for axis, (data, title, style) in zip(axes, panels, strict=True):
        axis.imshow(np.rot90(data), **style)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.tight_layout(pad=0.5)
    figure.savefig("figures/gradwarp_vs_orchestra.png", dpi=140, bbox_inches="tight")
    print("wrote figures/gradwarp_vs_orchestra.png")


if __name__ == "__main__":
    showcase()
    against_orchestra()
