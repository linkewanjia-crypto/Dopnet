# Portable entries in the public candidate

Use sibling `code/` and `data/` trees. Scientific outputs belong in a new external
run directory. Help, list, verification and plans do not start model execution.

```text
python -B code/scripts/tmlcn.py list
python -B code/scripts/diagnose_gate.py --verify-saved
python -B code/scripts/data_fraction.py verify
python -B code/scripts/diagnose_gate.py --smoke --output ../gate_smoke_new
python -B code/scripts/tmlcn.py plan --method dopnet_lite --seed 42 --output ../main_new
python -B code/scripts/verify_release.py --full
python -B code/scripts/train_smoke.py --method dopnet --output ../dopnet_smoke_new
python -B code/scripts/train_smoke.py --method dopnet --output ../dopnet_smoke_new --execute
```

Only an explicit `--execute` enables the facade/gate scientific routes. A gate
smoke covers one model/eight pairs, not full training, all baselines/Lite, or a
full test grid. Gate preflight still checks six weights and eight NPZ inputs.
The reduced-data entry only lists, verifies and plans; it never runs a trainer.

`tests/check_package.py` is the original archive-wide static checker. It needs
the real fixed validation cache and retained reference paths. The public code's
omitted manuscript snapshot is not a runtime prerequisite, but a thin data
transfer cannot satisfy all archive checks. Do not create dummy inputs or weaken
signatures to make one pass. The public checksum indices require complete
extracted code/data; a separate projection/inventory receipt can identify a thin
transfer without claiming it is the full package.

Full historical training still requires genuine validation reconstruction and
epoch proofs. The public `train_smoke.py` launcher directly starts the unmodified
native trainer in a new external working directory: three new one-frame batches,
one fresh validation frame, CUDA FP32, no private worker or hooks. It needs no
historical cache or attestation and does not establish historical reproduction.
Use the separate `environment/requirements-training-smoke.txt` dependency route
documented in the code README. The native trainer's own `--smoke` flag changes
the budget to two larger epochs and is not used by this three-batch entry. Complete
original settings, independent-method lineage/authorization gates and known gaps
are explained in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

The public code excludes manuscript PDFs, TeX assets, logos and fonts. Stored
paper identities and display mappings are reference metadata only. Two summary
plots are supported from saved CSV; full manuscript rendering is separate.
