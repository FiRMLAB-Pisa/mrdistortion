"""Optional external PyHySCO invocation for reverse-polarity EPI correction.

PyHySCO is GPL-3.0-only, so This package neither imports nor vendors it.  This
thin subprocess adapter is usable only after explicitly installing the
``recon-distortion`` extra.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

__all__ = ["run_pyhysco"]


def run_pyhysco(
    positive_phase_encode: str | Path,
    negative_phase_encode: str | Path,
    phase_encode_direction: int,
    *,
    executable: str = "pyhysco",
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the externally installed PyHySCO CLI on an opposite-PE NIfTI pair.

    Parameters match PyHySCO's ``pyhysco --file_1 --file_2 --ped`` command.
    The inputs must be reconstructed NIfTI files; raw MRD data is first
    reconstructed with a regular reconstruction.

    Examples
    --------
    Susceptibility distortion is corrected from a reversed-polarity pair: the two
    acquisitions distort in opposite directions, and the field that explains both
    is what unwarps them::

        import mrdistortion as mrd

        corrected = mrd.run_pyhysco("blip_up.nii", "blip_down.nii", "AP")
    """
    if shutil.which(executable) is None:
        raise ImportError(
            "PyHySCO is not installed. Install mrdistortion[epi] "
            "and ensure its 'pyhysco' executable is on PATH."
        )
    command = [
        executable,
        "--file_1",
        str(positive_phase_encode),
        "--file_2",
        str(negative_phase_encode),
        "--ped",
        str(phase_encode_direction),
        *extra_args,
    ]
    # The executable is the caller's to name and the arguments are paths it
    # supplied, so this is as trusted as the call that reached here. It is a
    # program rather than an import because PyHySCO is GPL-3.0-only.
    return subprocess.run(command, check=True, text=True)  # noqa: S603
