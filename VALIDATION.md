# Runtime validation

Bounded acceptance completed on 2026-10-15. The tested code performs forward computation, a few training updates, checkpoint restoration and saved-score plotting. **Strict per-observation comparison with the archived gate results FAILED at the original tolerances. Its cause is unresolved. These limitations are retained in version 1.0.0.**

## Environment and scope

The isolated Linux environment used Python 3.11.8, PyTorch 2.10.0+cu128, CUDA 12.8, NumPy 2.2.6 and Sionna 2.0.1. The observed GPU was NVIDIA GeForce RTX 4090 with 49,140 MiB reported total memory. All three dependency-install commands returned zero, taking 202.64 seconds in total. The historical gate comparison ran on CPU with two threads; the short training checks used CUDA and explicit FP32 (`--amp none`).

This acceptance used a small runtime data subset. It did not download the full historical validation caches. The 4.68 GB cache is the original fixed 2,000-frame validation set used for checkpoint selection. It is not training data or a test set. Exact reconstruction of that historical cache is unverified.

## Actual outcomes

| Check | Result | Tested scope and limitation |
| --- | --- | --- |
| Public native DOP-Net training | PASS | Independent public command from the verified clean code directory; three new one-frame batches and three updates. Return code zero, finite metrics and parameters. Elapsed 4.4418 seconds. All 130 optimizer parameter states have step three. |
| Public native DOP-Net-Lite training | PASS | Same independent route and budget. Return code zero, finite metrics and parameters. Elapsed 4.2763 seconds. All 50 optimizer parameter states have step three. |
| Validation during short training | PASS within scope | One separately generated validation frame, seed 610900002, gradients disabled. No archived calibration/validation input was used for training gradients. |
| Earlier private tensor-only checkpoint round trip | PASS | Strict state equality and restored forward maximum absolute difference zero for both models. This is separate from the public command, which only saves new checkpoints. |
| Earlier private complete checkpoint state restoration | PASS within scope | Original saved model, optimizer, scheduler, scaler and Python/NumPy/Torch CPU/CUDA RNG states restored exactly; forward maximum absolute difference zero. No continued optimizer step or native resume CLI execution was tested. Earlier safe-loader rejection records remain retained. |
| Historical `scripts/tmlcn.py train ... --execute` facade | FAIL / BLOCKED | Returned one because genuine historical input-reconstruction attestation was absent. The bounded native training success does not override this failure. |
| Selected model states | PASS within scope | Eighteen method/ablation/reduced-data representatives at seed 42 were constructed and strictly loaded; tensor dtype, shape and raw SHA were unchanged. Tested on one shared fresh validation frame. |
| Repeated forward under default backend | 17/18 PASS | Lee3DDual failed, maximum absolute repeat difference 0.00028136372566223145. That failed record is retained. |
| Lee3DDual deterministic backend | PASS | Repeated forward difference zero with `cudnn.deterministic=True`, `cudnn.benchmark=False` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. |
| Original plotting entry | PASS within scope | Two PNG and two PDF files generated from retained `macro_summary.csv`: NMSE and diagonal BER. No fresh full BER grid or complete paper figure reconstruction. |
| Public runtime staging and entry checks | PASS within scope | 799 code files verified against the tested staging manifest; 41 actual data files, not the full 2,704-file distribution. Help/list/verify-saved returned zero. Later local documentation/index updates do not alter scientific executable sources. |
| Gate inference | COMPLETED | 384 model-observation pairs from six states, eight batches, four physical frames and two users; 294.60 seconds. These are 32 physical frames and 64 user observations, not 384 independent samples. |
| Gate Table 5 display | MATCH | All 15 rounded/displayed cells match the archived table. |
| Gate per-observation numerical comparison | FAIL | Original tolerances retained: `atol=1e-6`, `rtol=1e-4`, NMSE difference limit `0.001 dB`. No expected values, inputs, algorithms or thresholds were changed to obtain a pass. |

Training elapsed times include the bounded generation/training/save checks and are not throughput benchmarks. The native scheduler horizon remained 40 epochs; the check stopped after the three-sample epoch. No full 12,000-sample epoch or convergence claim follows from it.

## Gate discrepancy and retained counterevidence

For output NMSE, the maximum absolute linear difference was `0.0006491243839263916`, maximum relative difference `0.005335745390686399` (about 0.534%), and maximum dB difference `0.023111244627641216`. The beta-one diagnostic had maximum dB difference `0.024012321341840108`. Matching rounded table cells does not establish a pass under these stricter per-record limits.

The comparator also emitted 1,536 missing-field entries for four older auxiliary norm fields absent from the new report schema. Those entries are schema differences and are distinguished from actual numerical failures; they do not establish nonfinite model outputs. The original comparator report, true numerical failures and initial Lee repeat failure remain retained.

Repeating the same original eight-row batch in the same environment matched exactly. Running that batch on CPU with the CPU-only wheel versus the CUDA wheel also matched exactly on all 12 common metrics. These checks do not explain the remaining difference from the archived Windows/Python 3.10.20/PyTorch 2.10.0+cpu reference. No first divergent intermediate tensor or causal explanation was established.

The Windows reference report itemizes five key scientific sources, all six
weights and eight NPZ inputs; their SHA-256 identities match the retained
manifest. Current byte checks also match all 20 scientific sources and 14 inputs
in that manifest, and the public candidate preserves the same scientific source
and input entries. The Linux reports record successful manifest verification
before execution. The Windows report does not individually lock all 20 source
dependencies, so a stronger claim about their historical identities is not made.
For the minimum B8, `dopnet_seed43.pt` has SHA-256
`1faa348d2585b16787b81c93463b6a4221e2a06dbb4241cc4341ee6d399089e2`
and `batch_0000.npz` has
`c89b71d4d1d109659a6aa793d4616386f7076f4c5deb160c975921b9b7328288`.

| CPU comparison environment | Windows reference | Linux recomputation |
| --- | --- | --- |
| Python | 3.10.20 | 3.11.8 |
| PyTorch | 2.10.0+cpu | 2.10.0 CPU-only or cu128 wheel, both forced CPU |
| NumPy / CPU threads | 2.2.6 / 2 | 2.2.6 / 2 |
| Compiler | MSVC 194234444 | GCC 13.3 |
| MKL / OpenMP | 2025.3 / 2019 | 2024.2 / 201511 |
| CPU capability | AVX2 | AVX512 |

Both use the same recorded Torch commit, MKLDNN v3.7.1, enabled MKLDNN and
non-deterministic-algorithm mode. The reference and current inference helpers
use FP32 real tensors, complex64 signals, evaluation/inference mode and no
autocast; their four input/forward/conversion helper ASTs match. The compiler,
MKL and instruction-set differences are observed environment differences,
not established causes. This is a disclosed cross-platform recomputation
limitation; exact reproduction of the archived per-observation values is not
claimed.

## Independent public training invocation

From a clean public directory containing `code/`, with the separate CUDA smoke
dependencies in `code/README.md` installed, these commands were actually run:

```text
python -B code/scripts/train_smoke.py --method dopnet --output ../dopnet_smoke_new --execute
python -B code/scripts/train_smoke.py --method dopnet_lite --output ../lite_smoke_new --execute
```

Absolute paths and fresh external output names were used in the recorded server
commands; the commands above normalize those locations. The clean code directory
contained 801 files, with all 800 non-index files verified and no symlinks or
private workers. Its `SHA256SUMS.json` has SHA-256
`daf214e0a5091470dfe10a8b4f17e1c71903ba4fee3e265dc0fe5c4528a15dab`.
Both the launcher and native trainer returned zero for each model. The launcher
uses only the standard library and starts unmodified `training/main/train.py`
in a separate interpreter, clearing inherited `PYTHONPATH`/`PYTHONHOME`. There
are no training imports into a private worker, monkeypatches or hooks.

Native history records one completed three-sample epoch and finite training and
validation metrics. The new checkpoints were inspected read-only with
`weights_only=True`, a fixed NumPy type context and no unsafe pickle fallback:
all 130 DOP-Net and 50 Lite optimizer states had step three, with no missing
parameter states. Validation caching was disabled, the output directories were
new and the logs recorded cache misses; native generation therefore produced
one separate fresh validation frame at seed 610900002 for each command. Lite's
fast progress bar did not retain an explicit final `3/3` line; checkpoint step
counts establish its three actual updates. These public commands do not test
checkpoint restoration or historical reconstruction.

## Earlier private training evidence

The private acceptance worker invoked the original `training/main/train.py` argument parser and `main()`, with bounded fresh inputs and observation hooks. It was not a successful execution of the blocked historical facade. The recorded native arguments for DOP-Net were:

```text
--config code/config/training/main/dopnet_seed42.json
--run-name dopnet_private_three_batches --seed 42 --device cuda
--train-samples 3 --val-samples 1 --batch-size 1
--generation-batch-size 1 --microbatch-size 1 --stop-after-epoch 1
--amp none
--set train.validation_cache=false
--set system.sionna_device=cuda
--set data_pipeline.require_cuda=true
--set data_pipeline.emit_pilot_only_fast_time_add=true
--set data_pipeline.synthetic_microbatch_size=1
--set data_pipeline.threegpp_microbatch_size=1
```

The Lite route used `dopnet_lite_seed42.json` and run name `dopnet_lite_private_three_batches`, with the same bounded arguments. The tested `training/main/train.py` was 42,895 bytes, SHA-256 `fdcac9603a887f78087b58fb2c5393ed5e9f87ae72ce366cc8b6ad260a56a85c`. Paths above are normalized relative arguments, not a claim that an additional standalone convenience command was tested.

## Evidence and uncovered scope

The independent public-entry evidence includes `NATIVE_PUBLIC_SMOKE_SUMMARY.json`, `STAGING_RECEIPT.json`, `NATIVE_COMMANDS.json`, both native logs and `SMOKE_REPORT.json` files, `CHECKPOINT_READONLY_INSPECTION.json`, `DEPENDENCY_CHECK.json` and `FINAL_PROCESS_AND_INTEGRITY.json`. All native processes finished; the final check observed no GPU compute process and utilization zero.

Earlier private acceptance receipts include `BOUNDED_RUNTIME_SUMMARY.json`, `training_v2/REPORT.json`, the two training worker reports, `resume_probe/REPORT.json`, `models_v4/ACCEPTANCE_MODELS.json`, `lee_deterministic/cases/lee3ddual_seed42.json`, `plots/REPORT.json`, `GATE_FULL_SAVED_COMPARISON.json`, `GATE_MIN_CU_REPEAT_COMPARISON.json`, `GATE_MIN_WHEEL_COMPARISON.json` and `public_runtime/STAGING_RECEIPT.json`. Machine-specific receipts remain private; no public receipt URL is claimed.

Not covered: all 54 selected states, full historical training or validation reconstruction, mixed-precision training, training continuation after restoration, full paper evaluation grids, full paper figure regeneration, full-cache runtime loading, cross-platform bitwise equality or submission readiness. Saved-data integrity checks and archive readback are separate from this runtime scope. Supplied scientific originals and unfavorable evidence were preserved. This runtime validation did not submit the paper or itself publish a release. Version 1.0.0 release metadata was prepared separately; citation metadata and its checksum index were updated after the tested native snapshot, without changing scientific engines, inputs, selected states or the public training launcher.
