"""Build the figure the README shows: one column per correction.

Each column is produced by the example of the same name, cut down to the panels
that say what the correction does. Run it from this directory; it writes
``figures/showcase.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import field_map_from_phase as phase_example
import gradient_nonlinearity as gradient_example
import mrdistortion as mrd
import spiral_deblurring as spiral_example
import susceptibility as epi_example

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})


def gradient_column():
    """A lattice bent by a coil, and the lattice it should have been."""
    size = 96
    geometry = mrd.ImageGeometry(
        shape=(size,) * 3,
        fov_mm=(400.0,) * 3,
        direction=np.eye(3),
        center_mm=(0.0, 0.0, 0.0),
    )
    warp = mrd.Gradunwarp(gradient_example.generic_coil(-0.16), geometry)
    lattice = gradient_example.grid_phantom((size,) * 3, spacing=10)
    middle = size // 2
    # Resampling a true lattice at the coil's own coordinates is what the
    # scanner would have measured; correcting returns the lattice.
    return (
        ("gradient nonlinearity", "acquired", "corrected"),
        (warp(lattice)[:, :, middle], lattice[:, :, middle]),
        dict(cmap="gray"),
    )


def spiral_column():
    size = 160
    timing = mrd.ReadoutTiming.from_trajectory(
        spiral_example.variable_density_arm(), duration=8e-3
    )
    transfer = mrd.fit_transfer(timing, band=150.0, terms=8)
    axis = np.fft.fftfreq(size) * 2
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    disc = squared <= 1.0
    times = np.interp(np.clip(squared, 0, 1), timing.squared_radius, timing.times)
    truth = spiral_example.phantom(size)
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    smooth = 220 * (0.7 * columns + 0.3 * rows)
    levels = np.linspace(smooth.min(), smooth.max(), 32)
    field = levels[np.abs(smooth[..., None] - levels).argmin(-1)]
    blurred = np.zeros(truth.shape, complex)
    for value in np.unique(field):
        blurred += np.fft.ifft2(
            np.fft.fft2(truth * (field == value))
            * disc
            * np.exp(2j * np.pi * value * 8e-3 * times)
        )
    corrected = mrd.deblur(
        torch.from_numpy(blurred), torch.from_numpy(field), transfer
    ).numpy()
    top = np.abs(blurred).max()
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
    up, down, _ = epi_example.displaced_pair()
    result = mrd.correct_susceptibility(
        up, down, voxel_size=(1.0, 1.0, 1.0), alpha=50.0, max_iter=30
    )
    middle = up.shape[2] // 2
    acquired = up[:, :, middle].numpy()
    corrected = result.blip_up[:, :, middle].numpy()
    # Jacobian modulation rescales intensity, so each panel gets its own window.
    scale = lambda x: x / np.percentile(x, 99.5)
    return (
        ("susceptibility", "blip up", "corrected"),
        (scale(acquired), scale(corrected)),
        dict(cmap="gray", vmin=0, vmax=1.0),
    )


def main() -> None:
    columns = [gradient_column(), spiral_column(), phase_column(), epi_column()]
    figure, axes = plt.subplots(2, len(columns), figsize=(2.5 * len(columns), 5.4))
    for index, ((heading, upper, lower), images, style) in enumerate(columns):
        for row, (image, label) in enumerate(zip(images, (upper, lower), strict=True)):
            axis = axes[row, index]
            axis.imshow(image, **style)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_ylabel(label, fontsize=8)
            if row == 0:
                axis.set_title(heading, fontsize=9, pad=6)
    figure.tight_layout(pad=0.6)
    figure.savefig("figures/showcase.png", dpi=140, bbox_inches="tight")
    print("wrote figures/showcase.png")


if __name__ == "__main__":
    main()
