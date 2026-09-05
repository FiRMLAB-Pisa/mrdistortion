# Examples

One example per correction, each against something that knows the answer: a
displacement with a closed form, a field impressed on purpose, a blur simulated
exactly. The `.py` is the source — it runs as a script and lints with the rest
of the package — and the `.ipynb` beside it is generated from it, executed, and
committed with its outputs, so it opens in Colab.

| example | shows | checked against |
|---|---|---|
| [`01-gradient_nonlinearity`](01-gradient_nonlinearity.ipynb) | `Gradunwarp`, `source_grid` | the displacement the coefficients imply, in millimetres |
| [`02-field_map_from_phase`](02-field_map_from_phase.ipynb) | `field_map_from_phase` | two lobes of known amplitude and position |
| [`03-spiral_deblurring`](03-spiral_deblurring.ipynb) | `ReadoutTiming`, `fit_transfer`, `deblur` | a blur applied exactly, then removed |
| [`04-susceptibility`](04-susceptibility.ipynb) | `correct_susceptibility` | the agreement between the two polarities |

[`_brainweb.py`](_brainweb.py) builds the BrainWeb slice and the field its own
tissue would produce, which `03` uses and fetches if it is not beside the
notebook. [`figures/make_showcase.py`](figures/make_showcase.py) draws the
README's figures — from acquired data when `GRADWARP_DATA` and `EPI_DATA` point
at it, simulated otherwise. Neither is one of the examples.

## Rebuilding

```bash
pip install -e .[epi] jupytext nbclient ipykernel
bash scripts/build_examples.sh
```

Every notebook is regenerated from its script and executed against the
interpreter the package is installed into. `--check` verifies the notebooks are
current without running them.
