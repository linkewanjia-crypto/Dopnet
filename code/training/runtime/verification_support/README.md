# DOP-Net

For the current GPU-only verification continuation, use
[GPU_RESUME_zh.md](GPU_RESUME_zh.md) and `resume_gpu_verification.py`.
It reuses completed calibration, selects the original GPU data generator,
and measures receiver latency only on GPU. It performs no CPU/GPU speed experiment.

Doppler-aware Occupied-band Pilot Network for doubly selective multi-user
MIMO-OFDM channel estimation. **DOP-Net-Lite** is the directly supervised compact
architecture with both physical input views, fewer parameters and lower measured
model-stage latency.

## Rebuild the completed study without training

The project has three sibling directories: `论文/` (manuscript), `开源代码/`
(this source), and `实验结果及数据/` (completed experiments). From this directory, run:

```bash
python -m pip install -r classical_extension/requirements_cpu.txt
python reproduce_results.py --results "../实验结果及数据" --manuscript "../论文"
```

This CPU-only command rebuilds the 636-condition study aggregates, the matched
99-condition joint-BEM statistics, and the manuscript figures/tables from saved
per-frame scores. It performs no training, inference or channel generation.
Omit `--manuscript` to rebuild only statistics. Compile the two paper sources
separately with XeLaTeX if updated PDFs are needed.

The original paper tables use three-training-seed Student-t intervals. Joint-BEM
tables resample shared eight-frame test-generation clusters and, for learned
methods, the three training seeds. Their grids and interval definitions remain
separate. The preserved `classical_core/server_report/` records the server report;
rebuilding writes `classical_core/summary/` without replacing that evidence.

## Fresh training and neural evaluation

For corrected classical references, streamed replay of deleted observations,
and CPU/GPU receiver timing using existing weights, see
[SERVER_VERIFICATION_zh.md](SERVER_VERIFICATION_zh.md). The server release
`发布包/dopnet_no_training_verification_20261001.zip` includes the exact frozen
simulation source and the new `run_verification.py` pipeline. It does not include
old results or checkpoints. It reuses the completed server result directory.

Python 3.11+, a CUDA GPU and the dependencies in `requirements.txt` are required.
Run on the training server:

```bash
python -m pip install -r requirements.txt
python run_experiments.py --output "../实验结果及数据/dopnet_new"
```

This trains all 16 learned configurations with seeds 42, 43 and 44, then evaluates,
aggregates and profiles them. Within each seed and epoch, all applicable models
consume the same generated batches before the next epoch begins. Batch size is
32; the 3GPP link chunk remains 8. The CUDA allocator policy is applied before
PyTorch import. DOP-Net-Lite uses direct supervision.

Stages are `preflight`, `train`, `evaluate`, and `aggregate`. A completed output
can resume under its recorded protocol; use a new output after changing code,
configuration, sample count or generation batch size. The accompanying completed
results do not require neural retraining to rebuild the paper.

## Joint ICI-aware classical extension

```bash
python run_classical.py --project "../实验结果及数据/reference_project" --reference-results "../实验结果及数据/dopnet_additions" --data /path/to/frozen_observations --output "../实验结果及数据/classical_new"
```

The unmodified simulation project must belong to the reference results. Export
runs on its original CUDA/Sionna server; calibration and estimation use the CPU
reference implementation. Add `--skip-export` when the verified raw observations
already exist. See `classical_extension/README.md` for the basis, ridge and solver
settings. Current archived results were computed with that CPU implementation;
unmeasured GPU acceleration is not substituted into their provenance.

`reference_project/` preserves the exact historical simulation source used by the
completed study. Use it for frozen-waveform replay. This public code edition is
for new runs and saved-score reproduction; use a new output for fresh training.
Fixed validation files and full training histories/states are kept in the data
directory. They are separate from the compact inference checkpoints.

## Layout

- `models/`, `data/`, `utils/`: estimators, waveform generation and shared operators.
- `configs/dopnet.yaml`: base simulation and training configuration.
- `run_experiments.py`: full three-seed server pipeline.
- `reproduce_results.py`: completed-score statistics and paper assets.
- `run_classical.py`, `classical_extension/`: frozen-frame joint-BEM comparison.
- `tests/`: CPU configuration and statistical checks.

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s classical_extension/tests -v
```

Stable identifiers are `dopnet` and `dopnet_lite`; display labels are DOP-Net and
DOP-Net-Lite. Training progress prints the total loss. Checkpoints, scores and
raw waveform caches are separate from this source package. Select the source
license before publishing; third-party notices remain with their components.
