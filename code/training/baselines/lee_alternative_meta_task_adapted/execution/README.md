Historical documentation with machine paths redacted. Original bytes remain in the scientific archive.
This document is not a valid signed execution plan; use the portable entry and a fresh prepare.

# Lee execution extension 1.0.0

This is an explicit administrative extension of frozen Lee0.3.1. The user has
authorized continuing until all work finishes or the user explicitly requests
stopping. The former13:46UTC limit remains in the legacy scientific recipe and
old evidence. It is superseded only by signed EXECUTION_CONTROL for new sessions.
No purchase, renewal, plan change, early return or scientific-budget reduction is
authorized. The40 epochs,12000 fresh frames,1meta+39online, original geometry,
source19, runtime31, validation2000, model/optimizer/RNG/binding stay unchanged.

CPU waiting controller may start while the old worker runs. It performs no
checkpoint reads/copies/model imports/GPU jobs until the old queue naturally
records exit75, all recorded/process-scan Lee workers exit, and GPU is empty.
It then copies manifest/config/qualification and immutable full-state commits,
checking every source/destination SHA and byte count before and after. Original
results and diagnostics are retained. A new root receives the prefix; partial
epochs do not count and restart from the next full global epoch.

The explicit in-process overlay changes administrative deadline checks, the
online_step default deadline argument, all three scanner references and durable
external commit-control receipts. It never writes legacy source or reserializes
old checkpoint bytes. New state/commit schemas and scientific bindings retain
the original identity; external receipts bind the extension source and user
execution control. Existing actual micro8 hardware qualification is reused; no
probe is repeated. A pre-existing full40 train task is reused only with its actual
legacy exit0 and submitted marker, then receives an actual restricted verifier.

## Launch once from authenticated Jupyter terminal

Run with cwd __HISTORICAL_MACHINE_PATH_REDACTED__ The surrounding
launch adapter must exclusively create its launch request/log (open x), preserve
the child PID and source proof, and must never repeat this command. Source-only
installation and metadata tests do not qualify actual migration or training.

```text
python -u -B -m lee_execution_extension.controller --legacy-code-root __HISTORICAL_MACHINE_PATH_REDACTED__ --legacy-results __HISTORICAL_MACHINE_PATH_REDACTED__ --legacy-launch __HISTORICAL_MACHINE_PATH_REDACTED__ --output __HISTORICAL_MACHINE_PATH_REDACTED__ --control __HISTORICAL_MACHINE_PATH_REDACTED__ --project __HISTORICAL_MACHINE_PATH_REDACTED__ --parent-receipt __HISTORICAL_MACHINE_PATH_REDACTED__ --forti-code-root __HISTORICAL_MACHINE_PATH_REDACTED__ --authorization-text "你可以一直运行，直到完成或者我让你完成" --authorization-utc <actual-user-authorization-UTC>
```

The CLI authorization-utc is the real clock time when the agent records the
already-received user authorization, not an invented user-message send time.
The signed control explicitly labels authorization_time_basis as
agent_recorded_after_actual_user_message. Do not invent an earlier timestamp.
Metadata tests: `python -B -m unittest lee_execution_extension.test_metadata`.
This imports only standard-library administrative code and toy metadata.

## Schemas and read-only API

Package exports activate_for_read(control_path,migration_path,legacy_code_root,
evidence_root=None), verify_source(root). It returns Context with control,
migration,migration_sha,evidence_root. No Torch/NumPy is imported. Call before
importing legacy scanner/verifier function references. Repeat activation can
relocate the same exact control/migration identity between live and archive
copies; changing the identity in a process is rejected.

control/EXECUTION_CONTROL.json: signed schema lee_execution_control_v1,
mode until_complete_or_explicit_user_stop, deadline_utc null, old deadline,
user authorization, scientific-source manifest SHA and extension signature,
absolute server paths stored as historical identities. Local Windows checkers
use POSIX path syntax validation without dereferencing these historical paths.

new_output/MIGRATION_RECEIPT.json: signed schema lee_execution_migration_v1,
legacy_queue_sha256,legacy_launch_sha256,training_manifest_sha256, all exact
copied_files and prefix_commits keyed42/43/44, copied_utc. Queue/launch snapshots
live beside it. New output/candidate_micro8 retains the exact old manifest,
configs and original scientific binding; no new recipe/config is generated.

New commit external receipts are under
new_output/execution_commits/lee_meta_seedNN/epochNN.json. An fsynced publication
intent binds state/commit SHA before directory publication, then becomes
committed. Crash recovery only reconciles exact already-published bytes; an
unpublished intent/staging remains incomplete for explicit review. No tensor
deserialization is used for migration/scanning/recovery.

The new queue has schema lee_execution_extension_queue_v1 and exactly six
ordered tasks train_seed42,verify_seed42,train_seed43,verify_seed43,train_seed44,
verify_seed44. Complete requires true integer0 for each, proof_sha256 for each
actual verifier, and final status three_seed_training_metadata_complete_scoring_pending.
It does not qualify scoring, latency or local handoff. Each training proof adds
execution_control_signature,extension_source_signature,migration_receipt_sha256,
execution_mode,legacy_deadline_superseded to the original signed proof. RNG
presence is true; independent bitwise boundary proof remains false.

For archives place control,MIGRATION_RECEIPT,legacy_queue_state,
legacy_launch_receipt and execution_commits in evidence/execution. Pass that
directory as evidence_root. Bundle the complete extension source manifest and
original scientific source19 and all120 full-state commits. New downstream and
Lee latency versions must bind this extension explicitly and recognize both
new controller/worker names in their no-live process gates.

## Explicit user stop

Only an actual user request authorizes creating control/USER_STOP.json. It is
signed schema lee_execution_user_stop_v1, status user_requested_stop,
execution_control_signature and requested_utc. Workers observe it before model
microsteps/validation/commit, preserve all immutable committed states and return75.
The controller records incomplete_user_stop; no other process is signalled.
Invalid or foreign stop files fail closed. They are never silently ignored.

Actual completion still requires all3x40 training and120-state metadata checks,
269 frozen scores, CI/assets/fullZIP, Lee actual36-row latency, every completed
upstream checker, uniform browser download and local full verification before
the authorized instance return.
