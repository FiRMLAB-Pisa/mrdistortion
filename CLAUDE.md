<!-- Generated from AGENTS.md by scripts/sync_agent_docs.sh. Do not edit. -->

# mrdistortion — agent instructions

<!--
This file is the SOURCE. CLAUDE.md and GEMINI.md are generated from it by
scripts/sync_agent_docs.sh, which pre-commit runs. Edit this file, never those.
-->

## What this package is

Geometric distortion correction: gradient nonlinearity from a coil's own
spherical-harmonic coefficients, and susceptibility distortion through PyHySCO.

It sits on `mrutils` and beside `mrtoeplitz`, `torchsolve`, `mrllr` and
`mrmotion`, which never import each other; `deepmr` sits on all of them.

Two distortions, and they are not the same kind of problem. Gradient
nonlinearity is a property of the coil, stated once by its manufacturer, and
correcting it is deterministic. Susceptibility distortion is a property of the
subject, measured from a reversed-polarity pair — that estimation is PyHySCO's.

**No vendor coefficient table is bundled or persisted, ever.** A site whose
coefficients are not a file satisfies `CoefficientAccessor` instead. The MRD
binding that reads them off a scanner header belongs to the server that speaks
MRD, not here: nothing in this package may import an MRD library.

**PyHySCO is GPL-3.0-only.** It is invoked as a program and never imported, so
its licence stays its own. Do not add `import pyhysco` anywhere.

**deepinv is a `deepmr`-only dependency.** Everything below it is plain Torch
with duck-typed operators (`A`, `A_adjoint`, `shape`). Do not import deepinv
here unless this package is `deepmr`.

## Build and test

```bash
pip install -e .[dev]
bash scripts/format_and_lint.sh   # rewrites in place; --check to verify only
pytest -q
```

Build and test steps are mandatory before reporting a change complete. Run them
and report the exact output; do not assume success.

## Tests

pytest with plain functions and fixtures — never `unittest.TestCase`. A test
name states the invariant it protects, so a failure reads as a sentence.

A correction is checked against what it should move, not against a stored
image: a coefficient set with one term has a displacement with a closed form,
and a coil that states no nonlinearity must move nothing at all.

## Comments and docstrings

Write for someone reading the code as it is now, who has no memory of any
earlier version of it. **Never** write text whose subject is the history of the
code. Banned in comments, docstrings and prose alike:

- "used to", "was once", "no longer", "previously", "now that", "this replaces",
  "the old X", "before the fix"
- justifying the present shape by contrast with a shape that is gone
- naming a bug that has been fixed, or the session that fixed it
- restating what the code plainly says

A docstring carries what a caller needs: one line of what, Parameters, Returns,
Raises. A comment earns its place only by explaining a non-obvious algorithm or
a choice a reader would otherwise undo — and even then, prefer a well-named
function or a test whose name states the invariant, because those cannot go
stale silently. When tempted to explain *why not the other way*, write a test.

Stale comments are actively harmful. Deleting an outdated comment is always
correct; rewriting one to describe the change is not.

## Documentation style

The audience is MR scientists. Write in the vocabulary of pulse sequences and
physics, not of software architecture. Never justify a design by describing the
design it replaced.

Do not print a measured constant that is not guaranteed across releases or
hardware. Name the symbol and where it comes from, and let the build supply the
number.