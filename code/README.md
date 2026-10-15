# DOP-Net TMLCN code

DOP-Net means **Doppler Observation-Preserving Network**. This distribution
contains the original model, waveform, training, validation and evaluation
engines, retained configurations, portable command adapters, and saved-evidence
mapping. Use it with the companion `data/` distribution as sibling directories.
The manuscript snapshot `paper_reference/` is intentionally excluded from this
public code candidate; no manuscript PDF, TeX class, logo or font is required by
the Python runtime. `docs/CURRENT_PAPER.json` retains reference identities only.

## Contents

| Path | Purpose |
| --- | --- |
| `training/main/` | Original DOP-Net, Lite, controls, supervised baseline adapters, waveform generation, losses, training and evaluation. |
| `training/baselines/`, `training/portable/`, `training/runtime/` | Distinct literature-adaptation engines, portable orchestration and original runtime dependencies. |
| `config/training/main/`, `configs/portable/` | 33 main and nine independent method/seed recipes. |
| `config/training/data_fraction/` | Twelve additional 25%/50% DOP-Net/slow-time recipes. |
| `scripts/` | Standard-library plans/checks and the bounded original-PyTorch gate entry. |
| `frozen_base/` | Scientific/reference contracts still required by the retained engines. |
| `docs/`, `environment/` | Runtime scope, evidence mappings and dependency observations. |

## Check inputs and saved evidence

From the directory containing `code/` and `data/`:

```text
python -B code/scripts/tmlcn.py --help
python -B code/scripts/tmlcn.py list
python -B code/scripts/diagnose_gate.py --verify-saved
python -B code/scripts/data_fraction.py verify
python -B code/scripts/verify_release.py
python -B code/scripts/verify_release.py --full
```

These are standard-library checks and plans, not model execution. Both checksum
indices describe this public candidate, not the larger local manuscript archive.
`--full` requires the complete extracted data distribution. A partial transfer
cannot pass a complete-package check. `tests/check_package.py` is an unchanged
archive-wide static checker: it requires the real fixed validation cache and
reference paths even when only a small gate diagnostic is needed. Its success
does not establish full scientific reproduction.

## Bounded original-PyTorch inference

```text
python -B code/scripts/diagnose_gate.py --smoke --output ../gate_smoke_new
python -B code/scripts/diagnose_gate.py --smoke --output ../gate_smoke_new --execute
```

The first command hashes inputs and prints a plan. The second explicitly runs
one DOP-Net model and four calibration frames: eight user-observation pairs,
normal and beta=1. The full calibration command omits `--smoke` and covers six
DOP-Net/CFR-derived states and eight batches, producing 384 model-observation
pairs from 32 distinct physical frames. It is neither training nor a full test
grid. Even smoke preflight verifies all 20 scientific source files and 14 input
files. The adapter disables GPU use, limits CPU threads to two, and uses strict
`torch.load(weights_only=True)` for SHA-bound checkpoints.

New output must be external to code/data. New gate and reduced-data entries
accept `--data-root` or `DOPNET_DATA_ROOT`; default data is the sibling directory.
`scripts/gate_inputs.json` records the public REPORT identity projection while
preserving original scientific input hashes and all Table 5 numeric definitions.

## Training, evaluation and scientific scope

For a short native training check, use a compatible Linux/CUDA environment with
Python 3.11. The bounded dependency route is separate from historical
`requirements.txt` and the CPU gate environment. In an activated environment:

```text
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r code/environment/requirements-training-smoke.txt
python -B code/scripts/train_smoke.py --method dopnet --output ../dopnet_smoke_new
python -B code/scripts/train_smoke.py --method dopnet --output ../dopnet_smoke_new --execute
```

The first smoke command prints a plan without creating output; the second runs
three newly generated training batches of one frame each and one fresh
validation frame. Use `--method dopnet_lite` with a different new output directory
for Lite. Both use FP32, keep the native 40-epoch scheduler horizon and stop after
the three-sample epoch. No full validation-cache download is needed. The script
starts the unmodified native `training/main/train.py` in a separate interpreter,
with no private worker, imports into a private harness, training hooks, archived
training tensors or selected checkpoints. Its output contains native history,
new checkpoints and `SMOKE_REPORT.json`; it does not test checkpoint restoration.
The output directory must not already exist and must be outside code/data.

Read [REPRODUCTION.md](docs/REPRODUCTION.md) before any scientific execution.
The 42-route facade retains original historical contracts: exact full training
requires a genuine fixed-validation reconstruction attestation, derived inputs
and epoch evidence. Those are not proved by selected weights or a cached tensor.
No gate or standard-library smoke establishes complete retraining. A deliberately
small-budget native run through `train_smoke.py` is a functional
optimizer/validation check with changed sample counts; it is not the paper's
40-epoch reproduction. The historical `tmlcn.py train` facade is not the default
short-run entry and remains blocked when genuine reconstruction attestation is
absent. The native short route does not create or waive that attestation.

Full main evaluation requires all 33 completed trained routes and original epoch
proofs. The retained 38-route score view excludes twelve reduced-data routes.
The additional joint ICI-aware OCE/DPSS baselines use published basis
constructions with this task's observation model, crop and calibration; they
are task adaptations, not claims of exact original-estimator reproduction.
WiMAE SSL+CE is the recorded ContraWiMAE-derived task adapter with clean-CFR
pretraining and pilot-LS CE inference; it is distinct from the supervised WiMAE
CE control. Check recorded variant/configuration/commit before changing its
upstream citation. DOP-Net-Lite reduces model width/capacity and has no
knowledge-distillation training stage.

`tmlcn.py plot` produces two summary plots from saved `macro_summary.csv`; it
does not regenerate every manuscript figure. The evidence matrix names original
manuscript assets as archival references; those assets are not bundled here.

## Dependencies and terms

`requirements.txt` records the historical observed training environment, not a
validated universal lockfile. `environment/requirements-gate.txt` records the
bounded CPU gate environment. Full generation/training engines require compatible
Linux/CUDA/Sionna dependencies and their real prerequisite proofs.

The author's own software is provided under the existing [MIT license](LICENSE),
as authorized for this candidate. Third-party notices and original license
headers remain applicable; MIT does not replace them. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[license scope](docs/LICENSING_AND_PROVENANCE.md), and the companion data terms.
No repository URL, publication DOI or manuscript identifier is invented. This
candidate preparation does not itself upload or publish anything.
