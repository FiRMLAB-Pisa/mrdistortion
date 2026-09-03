"""Build the figures the README shows.

Two of the four columns use acquired data that is not in this repository:

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import field_map_from_phase as phase_example
import gradient_nonlinearity as gradient_example
import mrdistortion as mrd
import spiral_deblurring as spiral_example
import susceptibility as epi_example

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
    correct = mrd.Gradunwarp.from_file(GRADWARP_DATA / f"{CASE}_coefficients.dat", geometry)
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
        print(f"real coil: displacement median {np.median(displacement):.2f} mm, "
              f"max {displacement.max():.2f} mm")
        return (
            (
                "gradient nonlinearity",
                "acquired",
                f"displacement, 0-{displacement.max():.0f} mm",
            ),
            (image[:, :, middle], displacement[:, :, middle]),
            None,
        )
    size = 96
    geometry = mrd.ImageGeometry(
        shape=(size,) * 3, fov_mm=(400.0,) * 3, direction=np.eye(3),
        center_mm=(0.0, 0.0, 0.0),
    )
    warp = mrd.Gradunwarp(gradient_example.generic_coil(-0.16), geometry)
    lattice = gradient_example.grid_phantom((size,) * 3, spacing=10)
    middle = size // 2
    return (
        ("gradient nonlinearity (simulated)", "acquired", "corrected"),
        (warp(lattice)[:, :, middle], lattice[:, :, middle]),
        dict(cmap="gray"),
    )


def spiral_column():
    size = 256
    readout = 12e-3
    timing = mrd.ReadoutTiming.from_trajectory(
        spiral_example.variable_density_arm(), duration=readout
    )
    transfer = mrd.fit_transfer(timing, band=250.0, terms=16)
    axis = np.fft.fftfreq(size) * 2
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    disc = squared <= 1.0
    times = np.interp(np.clip(squared, 0, 1), timing.squared_radius, timing.times)
    truth = spiral_example.phantom(size)
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    smooth = 200 * (0.7 * columns + 0.3 * rows)
    levels = np.linspace(smooth.min(), smooth.max(), 32)
    field = levels[np.abs(smooth[..., None] - levels).argmin(-1)]
    blurred = np.zeros(truth.shape, complex)
    for value in np.unique(field):
        blurred += np.fft.ifft2(
            np.fft.fft2(truth * (field == value)) * disc
            * np.exp(2j * np.pi * value * readout * times)
        )
    corrected = mrd.deblur(
        torch.from_numpy(blurred), torch.from_numpy(field), transfer
    ).numpy()
    top = np.percentile(np.abs(blurred), 99.5)
    return (
        ("spiral off-resonance", "off resonance", "deblurred"),
        (np.abs(blurred), np.abs(corrected)),
        dict(cmap="gray", vmin=0, vmax=top),
    )


def phase_column():
    coils, truth, _ = phase_example.acquisition(64)
    estimate = mrd.field_map_from_phase(coils, echo_time=0.7e-3, smoothing=1.5)
    middle = 32
    return (
        ("field map from phase", "true field", "estimated"),
        (truth[:, :, middle].numpy(), estimate[:, :, middle].numpy()),
        dict(cmap="RdBu_r", vmin=-200, vmax=200),
    )


def epi_column():
    if EPI_DATA.exists():
        archive = np.load(EPI_DATA / "corrected.npz")
        acquired, corrected = archive["ap"], archive["up"]
        middle = acquired.shape[2] // 2
        top = np.percentile(acquired[:, :, middle], 99.5)
        scale = lambda x: x / np.percentile(x, 99.5)
        return (
            ("susceptibility", "blip up", "corrected"),
            (scale(acquired[:, :, middle]), scale(corrected[:, :, middle])),
            dict(cmap="gray", vmin=0, vmax=1.0),
        )
    up, down, _ = epi_example.displaced_pair()
    result = mrd.correct_susceptibility(
        up, down, voxel_size=(1.0, 1.0, 1.0), alpha=50.0, max_iter=30
    )
    middle = up.shape[2] // 2
    scale = lambda x: x / np.percentile(x, 99.5)
    return (
        ("susceptibility (simulated)", "blip up", "corrected"),
        (scale(up[:, :, middle].numpy()), scale(result.blip_up[:, :, middle].numpy())),
        dict(cmap="gray", vmin=0, vmax=1.0),
    )


def showcase() -> None:
    columns = [gradient_column(), spiral_column(), phase_column(), epi_column()]
    figure, axes = plt.subplots(2, len(columns), figsize=(2.6 * len(columns), 5.6))
    for index, ((heading, upper, lower), images, style) in enumerate(columns):
        for row, (image, label) in enumerate(zip(images, (upper, lower), strict=True)):
            axis = axes[row, index]
            if style is None:
                axis.imshow(
                    np.rot90(image),
                    cmap="gray" if row == 0 else "magma",
                    vmax=np.percentile(image, 99.5),
                    vmin=0,
                )
            else:
                axis.imshow(np.rot90(image), **style)
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
    print(f"vs Orchestra: correlation {np.corrcoef(*flat)[0, 1]:.6f}, "
          f"nrmse {np.linalg.norm(flat[0] - flat[1]) / np.linalg.norm(flat[0]):.5f}")

    middle = image.shape[2] // 2
    top = np.percentile(image[:, :, middle], 99.5)
    panels = (
        (image[:, :, middle], "acquired", dict(cmap="gray", vmin=0, vmax=top)),
        (reference[:, :, middle], "Orchestra", dict(cmap="gray", vmin=0, vmax=top)),
        (ours[:, :, middle], "mrdistortion", dict(cmap="gray", vmin=0, vmax=top)),
        (
            (ours - reference)[:, :, middle],
            "difference, x20",
            dict(cmap="gray", vmin=-top / 20, vmax=top / 20),
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
