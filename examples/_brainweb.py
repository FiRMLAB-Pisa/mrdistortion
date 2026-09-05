"""A BrainWeb slice and the field its own tissue would produce.

The anatomy is BrainWeb subject 4. The field map is not invented: it is the
susceptibility forward model applied to BrainWeb's own tissue labels, so the
lobes sit where the air actually is -- the frontal sinus, the ear canals -- and
the brain itself is near zero, which is what a shimmed head looks like.

Susceptibility relative to water, in ppm: air +9.4, cortical bone -2.1, marrow
-1.0, fat +0.6. The field follows from the dipole kernel
``1/3 - kz^2/|k|^2``, and ``127.7 Hz`` is one ppm at 3 T.
"""

from __future__ import annotations

import numpy as np

CHEMICAL_SHIFT_PPM = {"air": 9.4, "bone": -2.1, "marrow": -1.0, "fat": 0.6}
HZ_PER_PPM_AT_3T = 127.7
SHAPE = (181, 256, 256)


def _susceptibility(labels: np.ndarray) -> np.ndarray:
    """Susceptibility in ppm for each BrainWeb tissue label."""
    from brainweb_dl import BrainWebTissuesV2 as tissue

    chi = np.zeros(labels.shape, dtype=np.float32)
    for label, key in (
        (tissue.BACKGROUND, "air"),
        (tissue.SKULL, "bone"),
        (tissue.BONE_MARROW, "marrow"),
        (tissue.FAT, "fat"),
        (tissue.AROUND_FAT, "fat"),
    ):
        chi[labels == label.value] = CHEMICAL_SHIFT_PPM[key]
    return chi


def _field_from_susceptibility(chi: np.ndarray) -> np.ndarray:
    """Off-resonance in Hz from a susceptibility distribution, at 3 T."""
    margin = [size // 4 for size in chi.shape]
    padded = np.pad(chi, [(m, m) for m in margin], mode="edge")
    axes = np.meshgrid(*[np.fft.fftfreq(size) for size in padded.shape], indexing="ij")
    squared = sum(axis**2 for axis in axes)
    squared[0, 0, 0] = 1.0
    kernel = 1.0 / 3.0 - axes[0] ** 2 / squared
    kernel[0, 0, 0] = 0.0
    field = np.real(np.fft.ifftn(np.fft.fftn(padded) * kernel))
    inner = tuple(slice(m, m + size) for m, size in zip(margin, chi.shape, strict=True))
    return HZ_PER_PPM_AT_3T * field[inner]


def brain_and_field(slice_index: int = 60, size: int = 256):
    """One axial slice, the field it sits in, and where the head is.

    Parameters
    ----------
    slice_index
        Which axial slice. Lower is more inferior and closer to the sinuses,
        so the field is stronger there.
    size
        Matrix size to return.

    Returns
    -------
    image, field, mask
        The T1 slice normalised to a peak of one, off-resonance in Hz, and the
        head. All shaped ``(size, size)`` with anterior at the top.
    """
    from brainweb_dl import Segmentation, get_mri
    from scipy.ndimage import gaussian_filter

    labels = np.rint(
        np.asarray(get_mri(sub_id=4, contrast=Segmentation.CRISP, shape=SHAPE))
    ).astype(np.int16)
    volume = np.asarray(get_mri(sub_id=4, contrast="T1"), dtype=float)

    field = _field_from_susceptibility(_susceptibility(labels))
    # Tissue labels are piecewise constant and a measured field never is; and
    # the scanner sets its centre frequency, so the head mean is zero.
    field = gaussian_filter(field, 1.5)
    head = labels != 0
    field = field - field[head].mean()

    turn = lambda plane: np.rot90(plane, 2)  # anterior to the top
    image = turn(volume[slice_index])
    image = image / image.max()
    field = turn(field[slice_index])
    mask = turn(head[slice_index])
    if image.shape[0] != size:
        index = np.linspace(0, image.shape[0] - 1, size).round().astype(int)
        picked = np.ix_(index, index)
        image, field, mask = image[picked], field[picked], mask[picked]
    return image, field, mask
