# Spiral off-resonance deblurring: what has to be decided first

This is a design note, not an implementation. Nothing in the package does this
yet, and the reason is that the two decisions below determine everything else,
and neither is settled by reading code.

## The problem is not the one gradient nonlinearity solves

Off-resonance acts on a spiral acquisition differently from how it acts on EPI,
and the difference is why this needs its own method rather than a displacement
field.

On EPI the phase accrues along one axis, so a constant offset moves the object
along the phase-encode direction. It is a *geometric* distortion: a shift, and
a shift can be undone by resampling — which is what `Gradunwarp` already does
for gradient nonlinearity and what PyHySCO estimates from a reversed pair.

On a spiral the readout is long and the k-space trajectory curls, so the same
offset accrues phase along a path that is not a straight line in any one image
axis. The point spread function *broadens* instead of moving. It is a blur, it
is spatially varying because the field is, and no resampling undoes it.

## Decision one: what the correction operator is

The families differ in where they put the work.

**Frequency-segmented / multifrequency interpolation.** Reconstruct the same
data at a handful of constant demodulation frequencies, then pick or blend per
voxel using a field map. The reconstructions are independent, so it parallelises
trivially and each is an ordinary gridding. What it costs is one full
reconstruction per segment.

**Conjugate-phase.** Apply the conjugate of the accrued phase during gridding,
segment by segment in time. The same arithmetic seen from the k-space side.

**Iterative field-corrected.** Put the off-resonance term inside the forward
operator and solve. Exact in the limit and it composes with everything else a
solve already carries — sensitivities, a subspace, a regulariser — but every
iteration pays for the field term.

`pulserver/recon/physics/_offresonance.py` already implements a segmented
kernel of this kind, and it is not broken: it agrees with `A^H A` to 2.1e-05,
the earlier "wrong" reading having been an array-module violation since fixed.
So the honest starting point for this package is that operator, moved.

**What has to be decided:** whether this package owns a *deblurring* step that
takes an image and a field map and returns a sharper image, or owns the
*operator* that a solve in another package uses. They are different products.
The first is usable without a solver and is what "deblurring" usually means;
the second is what an iterative reconstruction actually wants, and it belongs
next to the physics rather than here. My reading of the family layout is that
the operator belongs in `deepmr`'s physics and the standalone step belongs
here, but that splits one piece of arithmetic across two packages and should be
argued before it is built.

## Decision two: where the field map comes from

Every method above needs a field map, and the package currently has no way to
get one.

- **Measured**, from a multi-echo acquisition. Deterministic, needs the extra
  scan, and the estimator was deliberately dropped from `mrutils` earlier in
  this restructure — so it has no home at present.
- **Autofocus**, choosing the demodulation frequency per voxel or per region
  that maximises a sharpness metric. Needs no extra scan, and the choice of
  metric *is* the method: the same optimisation with a different metric is a
  different algorithm with different failure modes.
- **Estimated jointly** with the image inside a solve.

**What has to be decided:** whether the field map is an input this package
demands or something it estimates. If it estimates, the metric and the
region-growing scheme are the design, not a detail — and picking them is what
the literature below is for.

## What to read, and the question each should answer

Not yet read for this note; each is listed with what it is expected to settle.

| | question it should answer |
|---|---|
| Noll DC, Pauly JM, Meyer CH, Nishimura DG, Macovski A. *Deblurring for non-2D Fourier transform magnetic resonance imaging.* Magn Reson Med 1992;25:319-333. | The autofocus metric, and how it behaves where the object is dark |
| Man LC, Pauly JM, Macovski A. *Multifrequency interpolation for fast off-resonance correction.* Magn Reson Med 1997;37:785-792. | How many frequency segments a given readout length needs |
| Ahunbay E, Pipe JG. *Rapid method for deblurring spiral MR images.* Magn Reson Med 2000;44:491-494. | Whether the cost can be brought near one reconstruction |
| Chen W, Meyer CH. *Semiautomatic off-resonance correction in spiral imaging.* Magn Reson Med 2008;59:1212-1219. | The region-growing scheme, and what it does at tissue boundaries |
| Sutton BP, Noll DC, Fessler JA. *Fast, iterative image reconstruction for MRI in the presence of field inhomogeneities.* IEEE Trans Med Imaging 2003;22:178-188. | What the iterative form costs per iteration against the segmented one |

## What would make this real

A validation that does not beg the question. A field map applied forward to a
sharp phantom, then corrected, is circular — it tests the implementation
against itself. What is not circular: a spiral acquisition of a phantom with a
known susceptibility inclusion, corrected, against the same object acquired
with a readout short enough not to blur.
