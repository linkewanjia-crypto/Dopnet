# DOP-Net submission experiment

This package supports an **incremental 21-training-run experiment** that reuses
the 27 archived checkpoints for nine existing configurations, and a separate
**48-training-run fresh replication**. Both test DOP-Net on the fixed 636-cell
evaluation grid from the
supplied `DOPNet_Results.zip`. The prior ZIP contains checkpoints, saved scores,
and waveform fingerprints, but not received waveforms. The CUDA server must regenerate each
test cell and match its archived waveform SHA-256 before saving any new score.

The experiment is designed to address the paper's most consequential review
gaps: a pilot-support-matched fast-time control, removal of the
simulator-derived ADD coefficient target, a task-matched FortiTran architecture
adapter, and paired 25%/50%/100% training-data curves. The incremental route
first verifies the archived checkpoints and then compares every new training
epoch's original tensor hashes against the 120 historical data manifests.
It refuses mixed-data comparisons. Neither route guarantees acceptance at TWC
or any other journal.

## Server requirements

- Python 3.11 or later and a CUDA-capable GPU; Sionna 2.0.1.
- Packages in `requirements.txt`; use a CUDA-compatible PyTorch installation.
- The supplied `reference/DOPNet_Results.zip` (included here).
- The **original** `DOPNet.zip` from the project's training folder, supplied
  separately. Its 4.68 GB historical validation tensor cache is intentionally
  not duplicated in this release.
- Sufficient disk space for the 4.68 GB extracted validation cache, a derived
  pilot-only validation feature, one 12,000-frame shared training epoch, and
  checkpoints. The epoch cache is deleted only after every model completes
  that epoch; interruption is resumable.

To **train only the seven added configurations** (three seeds each), run:

```bash
bash run_additions.sh /absolute/server/path/to/DOPNet.zip
```

The incremental command also accepts the extracted historical project
directory if it contains `cache/validation_895338a79e73ad3f.pt` and the
matching `data_manifests` files. The cache is read directly in that case;
the original ZIP need not be uploaded again.

For a full 48-checkpoint retraining from scratch instead, run:

```bash
bash run_submission.sh /absolute/server/path/to/DOPNet.zip
```

This extracts and SHA-checks the original 2,000-frame validation cache, checks
its configuration and tensor schema, and compares all 29 original tensors to
a server regeneration. Only after those comparisons pass does it save the
pilot-only feature computed directly from that regenerated received waveform;
reconstruction from the archived sparse LS grid is recorded as a float32
roundtrip diagnostic. The immutable original cache is unchanged. It then
replays three complete archived test cells as a
preflight. If any identity, waveform, or numerical comparison fails, the
process stops before training. The public result ZIP remains unchanged.
An existing verified extraction and completed epochs are reused on
restart.

The incremental route verifies and imports 27 saved models from the immutable
reference ZIP, then trains seven added configurations at seeds 42, 43, 44 (21
runs). The other route trains all 16 configurations (48 runs). For a given seed
and epoch, it generates the training realization once, trains each requested
variant for that epoch, saves their state, and only then advances to the next
epoch. For incremental reuse, each generated cache must match all original
tensor hashes from the historical core and Lite training manifests before
that epoch's new models may train. Every route makes 375 optimizer steps per epoch.
Full-data routes use 12,000 unique frames; 25% and 50% routes repeat only the
first 3,000 and 6,000 frames of that same shared epoch to reach 12,000 training
exposures. All run for 40 epochs, with the same fixed validation set and
checkpoint-selection rule. The reduced-data comparison therefore isolates
unique-frame diversity at a matched update budget, rather than saving compute.

After training, the server evaluates all 48 learned routes and five fixed
references on the 636 original test conditions, with 512 frames (64 clusters
of eight) per cell. It verifies **every** cell's six-tensor realization SHA
against the supplied ZIP before accepting its new scores. Finally it writes
three-seed statistics, model-stage and receiver-stage profiles, eight LaTeX
result tables, and four focused PDF/PNG figures under `results/dopnet_additions/`
or `results/dopnet_submission/` according to the chosen route.

To rerun only the CPU report after a *completed* server evaluation:

```bash
python submission_report.py \
  --output results/dopnet_additions \
  --reference-results reference/DOPNet_Results.zip
```

The report rejects partial runs, stale aggregate signatures, missing routes,
invalid scores, or mismatched test fingerprints. It uses the original ZIP as
test-realization provenance only; all new table and figure values come from
the new run. `results/dopnet_submission/submission_report/` contains the report
manifest, tables, and figures.

## Experiment identities

The 16 learned configurations are DOP-Net, Slow-time, CFR-derived, AFT-only,
no fusion, no off-grid refinement, 3D dual-CNN adapter, WiMAE-style adapter,
DOP-Net-Lite, pilot-only fast-time, no ADD auxiliary target, FortiTran adapter,
and two reduced-data configurations each for DOP-Net and Slow-time.

The pilot-only control changes only the received-bin support of the PWB view:
it masks bins outside the joint transmitted pilot comb before fast-time
de-rotation. Its architecture, clean ADD target, and losses are unchanged.
The no-ADD-auxiliary control keeps DOP-Net's architecture, CFR objective,
sparsity term and other settings, but removes the simulator-derived
coefficient-label loss and monitoring. FortiTran uses only public pilot LS and
mask observations, without true channel statistics. It is a **task-matched
architecture adapter**, not a reproduction of the original paper's full
training pipeline or numerology. The source architecture is described in
[AdaFortiTran/FortiTran (ICC 2025)](https://doi.org/10.1109/ICC52391.2025.11160810);
the published authors' implementation uses a different SISO grid and pilot
layout.

All reported confidence intervals are pointwise and conditional on this fixed
test realization. The test set has already been inspected in earlier work, so
it is not a fresh blind holdout. Receiver timing is reported with its actual
included stages; model-stage results must not be described as end-to-end
real-time deployment. Read `SUBMISSION_GATE.md` before interpreting or
submitting the new figures.

No training, GPU simulation, or checkpoint inference was run on the author's
laptop while preparing this package. The server must strictly load the 27 old
models and pass all old/new training-data comparisons before a reused-model
result can enter the manuscript. Select a code/data license and archive the
exact run before public release.
