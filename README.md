# mrdistortion

Geometric distortion correction for MRI: gradient nonlinearity from a coil's
own spherical-harmonic coefficients, and susceptibility distortion through
PyHySCO.

[![Tests](https://github.com/FiRMLAB-Pisa/mrdistortion/actions/workflows/test-ci.yml/badge.svg)](https://github.com/FiRMLAB-Pisa/mrdistortion/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/FiRMLAB-Pisa/mrdistortion/branch/main/graph/badge.svg)](https://codecov.io/gh/FiRMLAB-Pisa/mrdistortion)
[![PyPI](https://img.shields.io/pypi/v/mrdistortion.svg)](https://pypi.org/project/mrdistortion/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Two distortions, and they are not the same kind of problem.

A gradient's field departs from linearity away from isocentre, so the image is
sampled on a grid that is not the one it is displayed on. That departure is a
property of the coil, stated once by its manufacturer, and correcting it is
deterministic: evaluate the harmonics over the grid and resample. It is also
the one that grows with field of view — a head-sized acquisition sees almost
none of it, and a whole-spine one is unusable without it.

Susceptibility distortion is a property of the subject rather than the scanner.
It is measured rather than looked up, and what measures it is a second
acquisition with the phase encoding reversed.

- **Vendor-neutral coefficients** — GE `.dat`, Siemens `.grad` and a plain
  `Alpha/Beta` table all parse into one representation, with the documented
  conversion factors rather than a fitted approximation
- **Nothing is bundled or persisted** — no coefficient table ships with this
  package, and a site whose coefficients are not a file satisfies
  `CoefficientAccessor` instead
- **Correction and resampling in one step** — an output geometry different from
  the input one unwarps and reslices together, so the image is interpolated once
- **The Jacobian is optional and physical** — the intensity a displacement
  concentrates or spreads, from SimpleITK's own filter, including for a single
  plane out of a 3D field
- **PyHySCO is called, never imported** — it is GPL-3.0-only, so its licence
  stays its own

## Quick Start

```bash
pip install mrdistortion          # numpy + SimpleITK
pip install mrdistortion[epi]     # susceptibility, through PyHySCO (GPL-3.0)
```

```python
import mrdistortion as mrd

# a coil's own table, in whichever syntax its vendor states it
coefficients = mrd.GradientCoefficients.from_file("coil.dat")  # or .grad, .coef
coefficients = mrd.GradientCoefficients.from_file(system_table)  # or an accessor

# the grid the image is on: extent, orientation and where its centre sits
geometry = mrd.ImageGeometry(shape, fov_mm, direction, center_mm)
geometry = mrd.ImageGeometry.from_mrd(header, reconstructed_shape)

# unwarp, keeping the intensity a displacement concentrates
correct = mrd.Gradunwarp(coefficients, geometry)
unwarped = correct(image)

# unwarp and reslice onto another grid, interpolating once
correct = mrd.Gradunwarp(coefficients, geometry, target_geometry)

# what it moves, before it moves anything: indices in, millimetres of Jacobian
grid = correct.sampling_grid()
intensity = correct.jacobian_grid()

# susceptibility, from a reversed-polarity pair
mrd.run_pyhysco("blip_up.nii", "blip_down.nii", phase_encode_direction=1)
```

## Examples

*Not yet written.*

## Related Works

- **HCP `gradunwarp`** — <https://github.com/Washington-University/gradunwarp>.
  MIT. The spherical-harmonic conversion and the coordinate conventions here
  are adapted from it and from its older `.dat` converter.
- **PyHySCO** — Williams AS, Chung J, et al. *PyHySCO: GPU-enabled
  susceptibility artifact distortion correction in seconds.* Front Neurosci
  2024;18:1406821. GPL-3.0-only, which is why it is invoked rather than
  imported.
- **SimpleITK** — <https://simpleitk.org/>. Its B-spline resampling and its
  `DisplacementFieldJacobianDeterminantFilter` are what move the image and
  weigh it.
- Janke A, Zhao H, Cowin GJ, Galloway GJ, Doddrell DM. *Use of spherical
  harmonic deconvolution methods to compensate for nonlinear gradient effects
  on MRI images.* Magn Reson Med 2004;52:115-122.
- Jovicich J, Czanner S, Greve D, et al. *Reliability in multi-site structural
  MRI studies: effects of gradient non-linearity correction on phantom and
  human data.* Neuroimage 2006;30:436-443. Why this correction is a
  prerequisite for comparing measurements across scanners at all.

## Development

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh
pytest -q
```

The docstring examples run as part of the suite — they are the documentation,
and an example that has drifted is a broken one. See
[CONTRIBUTING.md](CONTRIBUTING.md).
