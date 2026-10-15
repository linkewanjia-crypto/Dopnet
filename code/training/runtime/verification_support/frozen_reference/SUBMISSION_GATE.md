# TWC submission decision gate

This package makes the missing experiments runnable and auditable. It does not
by itself make the manuscript likely to be accepted or guarantee a minor
revision. The new results do not exist until the CUDA server completes either
21 new trainings plus strict reuse verification of 27 historical checkpoints,
or all 48 trainings from scratch, followed by fixed-data preflights, all 636
evaluation cells, profiling, and report generation.

The previously archived three-seed comparison showed a 0.315 dB overall
DOP-Net advantage over the matched Slow-time control, but a reversal on static
channels. The paper should center its claim on time-selective regimes and
report the static limitation. A small positive overall gain alone is not a
strong TWC novelty argument.

## Scientific gates after the new run

1. **Same observations.** The historical validation cache must pass its
   content-hash, schema, and 2,000-sample replay. The new pilot-only feature
   is computed from the matching regenerated received waveform; sparse-LS
   inversion error is recorded separately and is not treated as a new sample.
   Three representative test cells must pass the preflight fingerprint gate,
   followed by all 636 cells in the final evaluator. A failure prevents
   cross-run pairing; do not bypass it by changing seeds or accepting new
   test realizations under the old result name. If reusing old checkpoints,
   all 120 historical epoch manifests and every new cache batch must match on
   the original tensors; a failure requires fresh training.
2. **Matched mechanism.** Compare DOP-Net, Slow-time, and pilot-only fast-time
   with identical architecture, targets, shared epoch data, and test frames.
   Interpret DOP-Net versus pilot-only as access to received off-comb bins;
   pilot-only versus Slow-time additionally changes fast-time de-rotation.
   Do not attribute an advantage to a factor whose matched control does not
   support it.
3. **Auxiliary-target dependence.** Compare DOP-Net with `no_add_aux`. If
   removing simulator-derived coefficient labels eliminates the relevant
   advantage, present the result as a joint representation-and-supervision
   method rather than an input-representation effect alone.
4. **External comparator.** Report FortiTran as a task-matched, no-prior
   architecture adapter with its parameter count, own input access, and the
   same training and receiver protocol. The original FortiTran numerology and
   full training workflow are not reproduced here; do not call this an exact
   reproduction or claim universal superiority over published systems.
5. **Sample efficiency.** Show the two six-point curves (DOP-Net and Slow-time
   at 25%, 50%, and 100%) with three-seed intervals, on the same fixed test
   frames. All budgets make the same 375 optimizer steps per epoch; the 25%
   and 50% routes replay their unique-frame prefix. A data-efficiency claim
   requires a consistent advantage across budgets; otherwise report the
   observed interaction without extrapolation.
6. **Receiver and cost.** Common-diagonal BER uses the same detector for every
   estimator. Native ICI results are separate estimator–receiver outcomes.
   Model-stage latency and any model-plus-equalizer timing must name every
   excluded stage. Do not infer whole-system real-time behavior from them.

The report script creates tables and figures, not manuscript conclusions.
Rewrite the abstract, contributions, mechanism discussion, and limitations
only after inspecting those outputs. Include unfavorable slices. Three-seed
intervals are pointwise and conditional on the already inspected fixed test
set; they are not a fresh blind replication, a simultaneous confidence band,
or proof of real-channel generalization. Another independently frozen test
source or measured-channel transfer would materially strengthen TWC review.

Before submission, reconcile every manuscript number with the final report,
replace author placeholders, verify references and figure labels, compile a
clean root-level source package, record the exact code/data release and its
license, and check the current [IEEE TWC author instructions](https://www.comsoc.org/publications/journals/ieee-twc/policies-guidelines).
