"""Deblur a spiral acquisition that ran off resonance.

A spiral reads k-space over milliseconds, so off-resonance accrues phase along
the readout and the image blurs. The transfer is factorised once from the
trajectory's time map into a few separable terms; correcting an image is then a
handful of transforms whose cost does not grow with how finely the field is
resolved.

The blur here is simulated exactly, by summing over the distinct values of a
quantised field map, so the corrected image can be compared against truth.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

import mrdistortion as mrd

SIZE = 192
READOUT_S = 12e-3


def variable_density_arm(samples: int = 1500) -> np.ndarray:
    """One centre-out arm whose radius grows as a fractional power of time."""
    time = np.linspace(0.0, 1.0, samples)
    radius = time**0.6
    angle = 40 * np.pi * time
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def phantom(size: int) -> np.ndarray:
    """Ellipses with a smooth object phase, as an acquisition would have."""
    rows, columns = np.mgrid[0:size, 0:size] / (size / 2) - 1
    image = np.zeros((size, size))
    for row, column, height, width, value in (
        (0.0, 0.0, 0.75, 0.60, 1.0),
        (-0.15, 0.0, 0.25, 0.35, 0.55),
        (0.32, -0.28, 0.13, 0.09, 0.8),
        (0.32, 0.28, 0.13, 0.09, 0.8),
        (-0.45, 0.0, 0.10, 0.10, 0.35),
    ):
        inside = ((rows - row) / height) ** 2 + ((columns - column) / width) ** 2
        image[inside <= 1] += value
    # Fine structure, so the blur has something to smear.
    body = image > 0
    image[body] += 0.25 * (np.sin(22 * rows) * np.cos(19 * columns))[body]
    return image * np.exp(1j * 0.7 * (columns + 0.5 * rows))


def main() -> None:
    timing = mrd.ReadoutTiming.from_trajectory(
        variable_density_arm(), duration=READOUT_S
    )
    transfer = mrd.fit_transfer(timing, band=250.0, terms=16)
    print(f"{transfer.terms} separable terms, "
          f"transfer error {transfer.error(timing):.4f}")

    axis = np.fft.fftfreq(SIZE) * 2
    squared = axis[:, None] ** 2 + axis[None, :] ** 2
    disc = squared <= 1.0
    times = np.interp(
        np.clip(squared, 0, 1), timing.squared_radius, timing.times
    )

    truth = phantom(SIZE)
    ideal = np.fft.ifft2(np.fft.fft2(truth) * disc)

    rows, columns = np.mgrid[0:SIZE, 0:SIZE] / (SIZE / 2) - 1
    # +-250 Hz over a 12 ms readout is three cycles of accrued phase, which is
    # what an air-tissue interface does and what makes the blur unmistakable.
    smooth = 200 * (0.7 * columns + 0.3 * rows) + 90 * np.exp(
        -((columns + 0.3) ** 2 + (rows - 0.3) ** 2) / 0.06
    )
    levels = np.linspace(smooth.min(), smooth.max(), 32)
    field = levels[np.abs(smooth[..., None] - levels).argmin(-1)]

    blurred = np.zeros_like(ideal)
    for value in np.unique(field):
        selected = np.fft.fft2(truth * (field == value)) * disc
        blurred += np.fft.ifft2(
            selected * np.exp(2j * np.pi * value * READOUT_S * times)
        )

    corrected = mrd.deblur(
        torch.from_numpy(blurred), torch.from_numpy(field), transfer
    ).numpy()

    error = lambda x: np.linalg.norm(x - ideal) / np.linalg.norm(ideal)
    print(f"field {field.min():+.0f} .. {field.max():+.0f} Hz over a "
          f"{READOUT_S * 1e3:.0f} ms readout")
    print(f"error: blurred {error(blurred):.4f} -> corrected {error(corrected):.4f}")

    figure, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    top = np.abs(ideal).max()
    for axis_, data, title in (
        (axes[0], np.abs(ideal), "on resonance"),
        (axes[1], np.abs(blurred), "off resonance"),
        (axes[2], np.abs(corrected), "deblurred"),
    ):
        axis_.imshow(data, cmap="gray", vmin=0, vmax=top)
        axis_.set_title(title)
        axis_.set_xticks([])
        axis_.set_yticks([])
    shown = axes[3].imshow(field, cmap="RdBu_r", vmin=-300, vmax=300)
    axes[3].set_title("field map")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    figure.colorbar(shown, ax=axes[3], fraction=0.046, label="Hz")
    figure.tight_layout()
    figure.savefig("spiral_deblurring.png", dpi=110)
    print("wrote spiral_deblurring.png")


if __name__ == "__main__":
    main()
