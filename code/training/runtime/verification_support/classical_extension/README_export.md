# Frozen observation export

This adapter adds traditional baselines to the completed experiment without
training or loading a neural model. Run the export on the same CUDA/Sionna
server as the completed experiment. The resulting observations can be evaluated on the server or transferred for
CPU estimation and BER computation.

Example from the extension directory on the server:

```bash
python export_frozen_observations.py --project /path/to/reference_project --output /path/to/frozen_observations --suite core --list
python export_frozen_observations.py --project /path/to/reference_project --output /path/to/frozen_observations --suite core
```

The default reference is `PROJECT/results/dopnet_additions`. Override it with
`--reference-results` if the completed results directory has another name. A
results ZIP is also supported. `--list` reads metadata and saved scores only;
it does not import Torch/Sionna or generate observations.

`core` is a predefined 99-cell extension: 11 dynamic channels, Dopplers
800/1200/1600 Hz, QPSK at 15/25 dB for NMSE, and 16-QAM at 20 dB for BER;
CP=16, pilot spacing=4, and delay spread=300 ns. It covers 50,688 test frames,
with 512 frames per cell and the original batch size of 8. Its conservative
uncompressed storage budget is about 26.3 GiB; NPZ compression normally makes
the transferred directory smaller. `--suite all` exports all 636 historical
cells. `--cell-id ID` can be repeated to export a custom subset. Custom subsets
are labelled as such and do not stand in for a complete core suite.

An independent 128-frame calibration cache uses the frozen training
distribution and seed 810000001. Its chunk size follows the original training
sampler and may be 4 rather than 8. Calibration records have role `calibration`
and must not enter test averages. `--calibration-only` exports just this cache;
`--skip-calibration` is useful when it is already available.

The export retains the original configuration, generation batch sizes and
test seeds. It captures known transmitted pilots and the full occupied-band
received pilot symbols using temporary wrappers restored after export. The
wrappers do not alter observations or consume random draws. For every test
cell, the original six tensors are hashed in the same order as the completed
experiment. The cell is usable only after the hash exactly matches the saved
`realization_sha256` and `cells/ID/COMPLETE.json` is written. An interrupted or
mismatched cell is not released for CPU scoring. Resume with the same command;
completed cells are checked against their batch-file hashes and skipped.

Each `batch_OFFSET.npz` contains:

| Field | Shape / meaning |
|---|---|
| `y_pilot` | complex64 `[B,P,R,N]`; complete received occupied band |
| `x_pilot` | complex64 `[B,U,P,N]`; known pilot grid, zeros off each user's comb |
| `target_h` | complex64 `[B,U,D,R,N]`; original diagonal CFR at data symbols |
| `rx_data` | complex64 `[B,D,R,N]` |
| `tx_label` | int64 `[B,U,D,N]`; original Gray-coded symbol labels |
| `noise_var` | float32 `[B]`; noise variance per received resource element |
| `pilot_visible`, `data_visible` | int64 symbol indices within the visible frame |
| `cp_wave_samples`, `waveform_fft_size`, `subcarrier_spacing_hz` | public waveform geometry |
| `metadata` | scalar JSON containing role, cell/condition, offset and sample count |

`export_manifest.json` binds the configuration, simulator source hashes and
reference provenance. `suites` lists each suite's complete expected cell IDs;
`cells` supplies their original condition metadata. For a visible symbol index
`v`, the useful interval begins at
`v * (waveform_fft_size + cp_wave_samples) / sample_rate_hz`, relative to the
first visible useful sample. The target is averaged over useful sample indices
0 through `waveform_fft_size - 1`. Occupied bins run from `-N/2` through `N/2-1`.

The laptop can validate the exporter contract without channel generation:

```bash
python -m unittest discover -s tests -p test_export_contract.py
```

The supplied completed core results passed the original-frame fingerprint checks
on the server. Reproduction of estimates requires the raw export; rebuilding
statistics and paper assets requires only the supplied score archives.
