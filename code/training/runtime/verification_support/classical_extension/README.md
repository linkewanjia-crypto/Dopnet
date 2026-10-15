# Joint occupied-band BEM ridge estimators

The completed comparison uses joint OCE-BEM and DPSS-BEM ridge fits to the same
pilot-only two-user waveform as DOP-Net. All 128 occupied received bins of the
seven pilot blocks are used, including received bins off the transmitted combs.
Within-symbol ICI mixing uses the physical 1024-point FFT; the target CFR is the
useful-window average. Known pilots, public numerology and noise variance enter
the estimator. Targets and payload labels enter calibration/scoring only.

## Completed score reproduction

From the parent source directory:

```bash
python reproduce_results.py --results "../实验结果及数据" --manuscript "../论文"
```

Alternatively, from this directory:

```bash
python report_classical.py --classical-output "../../实验结果及数据/classical_core" --reference-results "../../实验结果及数据/dopnet_additions" --archived-scores --suite core
```

Archive mode verifies the recorded completed server report, score hashes,
calibration lock, original frame identities, bit totals and solver diagnostics.
It does not reread raw observations. `server_report/` stays unchanged; `summary/`
contains regenerated statistics. Ordinary reporting without `--archived-scores`
requires the original raw export and verifies its batches directly.

## Server export and evaluation

```bash
python export_frozen_observations.py --project /path/to/reference_project --reference-results /path/to/reference_results --output /path/to/frozen_observations --suite core
python run_classical_cpu.py --data /path/to/frozen_observations --output /path/to/classical_core --calibrate
python run_classical_cpu.py --data /path/to/frozen_observations --output /path/to/classical_core --evaluate
python report_classical.py --classical-output /path/to/classical_core --reference-results /path/to/reference_results --export-manifest /path/to/frozen_observations/export_manifest.json --suite core
```

The exporter requires the unmodified simulation runtime of the reference study.
It uses the existing allocator policy and fixed link chunks, and checks each
replayed cell against its archived waveform fingerprint. No neural model is
trained or loaded. `--suite core` covers 99 conditions: eleven dynamic channels,
Dopplers 800/1200/1600 Hz, QPSK SNR 15/25 dB and 16-QAM SNR 20 dB, CP 16,
pilot spacing 4, delay spread 300 ns. Each cell has 512 frames in batches of eight.
`--suite all` covers the original 636 cells.

## Recorded estimator settings

Both fits use Q=15 temporal modes and 48 signed delays [-8,39]. Their two-user
coefficient count is 1440 per receive branch, versus 896 pilot observations;
ridge regularization is part of the estimator. OCE frequencies are q/Tframe,
q=-7,...,7. DPSS uses the physical sample grid and training bandwidth 1620 Hz.

The ridge rule is `lambda = rho * noise_variance * mean(diag(D^H D))`. A global
rho is selected on 128 independent training-mixture frames, seed 810000001,
from [0.01,0.1,1,10,100]. The completed study chose rho=10 for both bases among
candidates completing all calibration frames. Smaller candidates hit the
256-iteration fit budget; those failures remain in `calibration_lock.json`.
Complex128 PCG requires relative residual <=1e-6; native detection allows 128
iterations. All recorded test solves converged. These are waveform-specific,
iteration-budgeted regularized BEM references.

Common diagonal MMSE is the shared-detector comparison. The classical native
receiver uses a full occupied-band ICI operator, while learned native receivers
use five-band interfaces. Their native BERs do not represent equal-cost solvers.

## Outputs and checks

`cells/*.npz` contains per-frame linear NMSE, integer bit/error counts, frame and
cluster indices, and solver records. `summary/` supplies cell/macro metrics,
paired differences, 95% bootstrap intervals, CSVs, plots and LaTeX tables.
Classical intervals resample test clusters; learned intervals also resample the
three training seeds. Small mathematical and file-contract tests run with:

```bash
python -m unittest discover -s tests -v
python check_reporting_math.py
```

Basis-expansion references: [Tang and Leus (2011)](https://doi.org/10.1186/1687-6180-2011-74)
and [Zemen and Mecklenbraeuker (2005)](https://doi.org/10.1109/TSP.2005.853104).
The implementation adapts these modeling principles to this waveform.
