# Reproduction routes and their limits

This public code candidate keeps the original training, evaluation and data-generation engines.
The manuscript snapshot, TeX assets and fonts are omitted; paper-to-evidence
metadata retains their original archive identities without claiming those paths
exist here. `tests/check_package.py` and scientific contracts are unchanged;
the full archive checker needs real validation/reference files and is not a
thin-transfer runtime test.
The current manuscript is `technical_fixes_20261014`, including the short GPT/Claude
acknowledgment revision. Paths below are relative to the directory containing
`code/` and `data/`; new run directories must be outside both directories.

## Start with existing evidence

```text
python -B code/scripts/diagnose_gate.py --help
python -B code/scripts/diagnose_gate.py --verify-saved
python -B code/scripts/tmlcn.py list
python -B code/tests/check_package.py
```

These use only Python's standard library. `--verify-saved` verifies the SHA256
identities of the accepted original-PyTorch report and records, the complete
384-pair matrix, and all 15 displayed Table 5 cells after the current aggregation.
It does not run a model or prove that the full test grid has been regenerated.
See `REPRODUCTION_MATRIX.md` for table/figure and evidence coverage.

## Original PyTorch gate diagnostic

```text
python -B code/scripts/diagnose_gate.py --smoke --output ../DOPNet_gate_smoke
python -B code/scripts/diagnose_gate.py --smoke --output ../DOPNet_gate_smoke --execute
python -B code/scripts/diagnose_gate.py --output ../DOPNet_gate_full --execute
```

Without `--execute`, the first command verifies input/source identities and prints
a plan without importing scientific packages or creating the output. The smoke
runs only `dopnet_seed42` and `batch_0000.npz`: four frames, two users, eight
model–observation pairs, normal and beta=1. The full route uses the original six
selected DOP-Net/CFR-derived checkpoints (seeds 42/43/44) and eight common batches
numbered 0000 through 0112 in steps of 16. This is 32 distinct physical frames,
64 distinct user observations, shared by six models, hence 384 pairs. It is small
calibration inference, not training or full evaluation.

Data root resolution is `--data-root`, then `DOPNET_DATA_ROOT`, then the sibling
`../data`. `--code-root` defaults to the code directory containing the script;
it allows a separately relocated code tree. All source, weight and calibration
identities in `scripts/gate_inputs.json` are checked before new inference.
The script loads only SHA-bound checkpoints with
`torch.load(weights_only=True, map_location='cpu')` and strict state matching;
it never falls back to unrestricted pickle. Scientific files remain unchanged.
The CPU route fixes two threads and disables GPU visibility. Its bounded
dependency versions are recorded in `environment/requirements-gate.txt`.

New outputs are `REPORT.json`, `PER_OBSERVATION.jsonl`, and
`TABLE5_RECOMPUTED.json` in the requested, previously absent external directory.
A smoke report explicitly says it cannot reproduce the full Table 5. The full
report displays NMSE by averaging observation-level linear values over all
observations and seeds before converting to dB. The beta and branch-power
statistics are means of observation-level statistics. The branch denominator is
the sum of separate weighted-branch powers, excluding the fusion cross term.
Forcing beta=1 retains the ADD-driven offset and downstream gate; this diagnostic
does not identify an exclusive causal mechanism. New floating-point outputs are
reported honestly; original accepted results are never overwritten.

The adapter copies `arr`, `complex_ri`, `trusted_inputs`, and `torch_forward`
without changing their numerical operations from the reviewed original-PyTorch
verification harness. Their source SHA256, the original gate script SHA256 and
the exact scientific import closure are recorded in `scripts/gate_inputs.json`.
The new adapter imports the original `training/main` model and waveform routines;
it does not import or execute the historical NumPy model, custom `ptload`, or the
comparison harness. NumPy is used only for array input and scalar aggregation.
Historical NumPy/PyTorch equivalence failures and operator traces are development
evidence and are outside the formal reproduction route.

## Complete training and evaluation engines

The existing standard-library facade retains all 42 declared training routes:
33 main routes (11 methods, seeds 42/43/44) and nine independent routes
(FortiTran, WiMAE, alternative Lee meta-learning). Inspect their real options with
`python -B code/scripts/tmlcn.py --help` and the relevant subcommand `--help`.
Help/list/plans run without science dependencies; actual scientific execution
requires an explicit `--execute` and compatible research hardware/dependencies.

The 11 main methods are `dopnet`, `cfr_derived`, `slow_time`,
`pilot_only_fast_time`, `aft_only`, `no_fusion`, `no_offgrid`, `lee3ddual`,
`wimae_ce`, `dopnet_lite`, and `fortitran`. Their retained configurations are in
`config/training/main/`. Independent engines and recorded configurations remain
under `training/baselines/`; adapters are under `training/portable/` and original
runtime dependencies under `training/runtime/`. The matched supervised adapters
and independently trained literature methods are distinct experiments.

```text
python -B code/scripts/tmlcn.py plan --method dopnet --seed 42 --output ../DOPNet_new_run
python -B code/scripts/tmlcn.py generate --method dopnet --seed 42 --epoch 1 --output ../DOPNet_new_run
python -B code/scripts/tmlcn.py verify-validation --output ../DOPNet_validation
python -B code/scripts/tmlcn.py evaluate --method main --output ../DOPNet_new_run
python -B code/scripts/tmlcn.py plot --output ../DOPNet_new_run
```

These are plans, not execution. Main training uses the unchanged optimizer,
scheduler, loss, epoch cache and fixed-validation routines: 40 epochs, 12,000
frames per epoch, and 2,000 validation frames. It does not initialize from the
packaged inference weights. Exact retained-run checks require the genuine
fixed-validation reconstruction attestation (not currently archived) and exact
epoch hashes; an input tensor or generator alone cannot replace that attestation.
Main evaluation additionally requires all retained trained routes and the shared
epoch proofs. Full CUDA/Sionna training and test-grid replay were not performed
during this release organization. The original observed dependency lists remain
historical environment observations, not a tested complete training lockfile.

Independent FortiTran/WiMAE preparation uses the existing local 38-row public
reference and an explicit fixed-validation working cache. Evaluation retains
original scientific equations and provenance checks. WiMAE requires a real newly
verified FortiTran evaluation and the appropriate lineage archive; it cannot be
unlocked with an unrelated historical success receipt. The alternative Lee route
has a separate new-run namespace, explicit authorization/acknowledgment and stop,
deadline and resource checks. Retained historical authorization is evidence only.
The current reproduction matrix records these prerequisites and the retained
implementation paths. Historical adapter-development documents are archived
separately and do not define a second current route.

## Plots and remaining coverage

`tmlcn.py plot` can create two summary plots from a newly saved
`summary/macro_summary.csv`; it does not recreate every manuscript figure's exact
layout. Frozen table/figure assets, source rows and producer mappings distinguish
display verification from generator replay. Reduced-data (25%/50%) evidence is
separately exposed by `scripts/data_fraction.py`; it does not change the original
42-route facade. Some historical raw inputs and exact reconstruction
attestations are missing. The retained additional joint-BEM reporting chain still
expects its original reference view and is not a ready canonical quick command.
Shared original receiver modules may retain additional diagnostics; their
presence is not a claim that an unrelated TWC coding study is part of this release.

No complete training, GPU experiment, missing-input reconstruction, full test-grid
replay or complete figure regeneration is claimed by this local organization.
The new gate entry has a separately recorded static/plan/saved-record check and
bounded CPU smoke receipt. License and upstream notices apply as recorded in the
release; local organization itself grants no new permission.

## Additional 25% and 50% recipes

```text
python -B code/scripts/data_fraction.py list
python -B code/scripts/data_fraction.py verify
python -B code/scripts/data_fraction.py plan --method dopnet_25pct --seed 42 --stage generate --epoch 1 --output ../DOPNet_fraction_new
python -B code/scripts/data_fraction.py plan --method slow_time_50pct --seed 42 --stage train --epoch 1 --output ../DOPNet_fraction_new
```

These standard-library commands inspect the twelve retained reduced-data
configurations at `config/training/data_fraction/` and twelve original selected
weights at `data/weights/selected/`. `verify` checks their SHA256 identities and
the unchanged original engine's budget fields; it does not deserialize weights.
`plan` never executes a subprocess or writes a run directory. Its original
generator/trainer CLI options have been read from the retained source.

The original `experiment_protocol` and `epoch_cache.read_batches` implement the
fixed-update design: generate a common 12,000-frame epoch, select its first 3,000
or 6,000 unique frames, and repeat that prefix to emit 12,000 training samples.
With batch size 32 this retains 375 optimizer updates per epoch for 40 epochs.
The configuration and implementation establish the intended procedure; the
missing complete historical epoch/validation evidence is not manufactured by
this organization. Plans expose the genuine validation attestation and epoch
cache prerequisites. They are not a shortcut around exact historical checks.

The existing public fixed-test view contains 38 routes and excludes these twelve
reduced-data routes. The original broader evaluator and training engines are
preserved, but this additive entry does not manufacture the original 53-route
completion signatures or advertise a full-grid evaluation command against the
38-row view. Saved source tables can verify the reported reduced-data summaries;
complete historical training and evaluation have not been rerun here.

## Public candidate scope and small functional tests

The current candidate does not supply a newly completed historical validation
attestation or epoch replay. Do not change signed metadata to bypass these gates.
A small-budget direct `training/main/train.py` invocation may test one optimizer
step and a separate small validation batch when deliberately using one epoch,
one train sample, one validation sample and batch size one. This changes the
recipe, starts from fresh weights and must write to a new external absolute
`--run-name`; it cannot be presented as retraining or a full scientific score.
The original `--smoke` flag instead selects two epochs/64 train/32 validation.
No new model, optimizer, GPU or server validation is performed by this packaging.

Joint OCE/DPSS-BEM references reuse published basis constructions with this task's
joint observation/crop/calibration choices. WiMAE SSL+CE is the recorded
ContraWiMAE-derived clean-CFR/pilot-LS adaptation, distinct from the supervised
WiMAE CE control; source variant and commit remain explicit. Lite is a smaller
architecture trained directly and has no distillation stage.

The public `gate_inputs.json` binds the projected REPORT bytes; the projection
redacts local path strings only. Original weights, NPZ observations, scientific
source hashes, scalar result rows and Table 5 definitions remain unchanged.
The old local smoke receipt is historical eight-pair evidence, not a new
scientific run of this public candidate.
