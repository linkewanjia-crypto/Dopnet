# DOP-Net

Code and retained numerical data for *DOP-Net: A Doppler Observation-Preserving Network for Doubly Selective MIMO-OFDM Channel Estimation*.

**Version 1.0.0.** Independent public native commands completed three new training updates and one fresh validation frame for each of DOP-Net and Lite. Selected-state loading and saved-score plotting succeeded in the bounded acceptance, and the 384-pair gate reproduced all 15 displayed Table 5 cells. Strict per-observation numerical comparison failed at the original tolerances: maximum normal NMSE difference 0.023111 dB and beta-one difference 0.024012 dB; the cause remains unresolved. Exact cross-platform numerical reproduction is not claimed. Companion numerical data: [Zenodo DOI 10.5281/zenodo.23392928](https://doi.org/10.5281/zenodo.23392928). Source version: [v1.0.0](https://github.com/linkewanjia-crypto/Dopnet/tree/v1.0.0).

## Code and data

Start with [code/README.md](code/README.md) and [code/docs/REPRODUCTION.md](code/docs/REPRODUCTION.md). Restore the separately distributed dataset into `data/` beside `code/`; do not put another `data/` level inside it.

The numerical distribution contains calibration observations, retained test results and source tables, selected trained states, validation caches and provenance. It uses four ZIP archives and two original PT cache files. The original 2,701 data paths are retained; public license and path-projection documentation are added separately. The 240-file epoch source/hash metadata subset is not a substitute for this numerical dataset.

The four ZIPs form the ordinary retained-data download (about 610 MB). The two PT files are **optional training-validation caches**: the 4.68 GB file is the original fixed 2,000-frame validation set used for model selection, and the smaller file provides its pilot-only validation inputs. They are not training samples or test results. Ordinary gate/metric checks and the bounded generator-based training check do not require downloading the full validation cache. Exact reconstruction of the historical full cache has not been verified from the available generator/configuration/seed records; the missing reconstruction evidence remains disclosed.

This release is not the full historical training corpus or complete optimizer/RNG resume history. Missing original input/validation reconstruction evidence and untested routes remain disclosed in the reproduction guide. It does not establish a fresh full-paper retraining or evaluation replay.

## Running and validation

Install the environment for the selected route; the main, gate and checker requirements represent different environments and should not be merged into one pip input. Use the route documented in the code README. Write new outputs outside the supplied code and data trees.

For a short CUDA FP32 training run, follow the bounded dependency instructions in
`code/README.md`, then run
`python -B code/scripts/train_smoke.py --method dopnet --output ../dopnet_smoke_new --execute`.
It starts the native trainer directly for three new batches and one new validation
frame, using a fresh external output directory. Use `dopnet_lite` and a new path
for Lite. This is the ordinary functional training entry; the historical
`tmlcn.py train` facade retains its reconstruction requirements.

`verify_release.py` (both default and `--full`) and `check_package.py` check the complete archived collection and therefore require the optional validation caches. They are not prerequisites for ordinary core inference, saved-metric checks or short generator-based training. Core users can use the documented `diagnose_gate.py` route without those caches; do not interpret an intentionally incomplete archive as a successful full-package check.

[VALIDATION.md](VALIDATION.md) records actual tested arguments, environment, results and uncovered scope. The public native commands ran from a verified clean code directory without private workers or hooks; checkpoint step counts confirmed three updates per model. An earlier private checkpoint-state probe restored complete states with zero forward difference; restoration and continued training are not tested by the public short command. The historical training facade remains blocked by missing reconstruction attestation. Supplied selected states were preserved, and no full-size training epoch was run.

External estimators are the documented task adaptations. WiMAE retains the recorded implementation/preprint variant; it is not silently replaced with a later variant or represented as an exact replay of the authors' full protocol. DPSS-JointBEM uses a DPSS basis in this study's adapted estimator. DOP-Net-Lite is a smaller independently trained architecture with the same objective, not a distilled model.

The saved gate study uses 32 physical calibration frames, 64 user observations and 384 model-observation pairs across six models. The pairs are not 384 independent samples. Original unfavorable results and qualifications are retained. Anonymous public metadata distinguishes original identities from identities of path-only derivatives; it does not pretend that a historical signature covers edited metadata.

## Licenses

Author-controlled software retains the existing [MIT notice](LICENSE); third-party components retain their own notices. The author has authorized CC BY 4.0 for this project's simulated numerical data, results and selected trained states, as scoped in the dataset license files. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md). This is not a license for unrelated fonts, publisher marks or third-party source.
