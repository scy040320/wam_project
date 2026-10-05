# Fixed-budget mechanism study

This study answers a narrower question than generalization: on trained tasks,
does attribution-conditioned belief and routing improve the same frozen
candidate-effect ranker on previously unused states?

| Arm | Implementation difference | Question |
|---|---|---|
| value_only | official highest value | Official selection control |
| candidate_only | frozen learned candidate-effect backbone | Candidate learning benefit |
| no_dependency | same backbone plus attribution, belief and routing; empty dependency graph | Increment of attribution/belief/routing bundle |
| full | identical to previous arm, with frozen DAG edges | Increment of dependency propagation |

The no-dependency contrast does not isolate classification from belief or
routing individually. Their bundle is the intervention. Full and
no-dependency share weights, thresholds, candidate effects and fallback.

## Frozen membership and execution

- LIBERO-90 task0 states35–36; task9/task46/task57 states14–15.
- Four conditions: clean, visual occlusion, object shift, execution deviation.
- 32 scenes, 128 arm trajectories, K=4, native 16-action blocks.
- One public prefix and runtime snapshot per task/state; all condition
  branches restore that source. Clean-A/B retain existing pairing tolerances.
- Every arm is genuinely re-executed from the same condition root, observation
  and candidates. Identical later requests share immutable generated arrays;
  logical calls remain charged separately per arm.
- 400 total policy steps, including 48 prefix steps; at most 92 postfork
  logical candidate queries, including an actually executed bounded fallback.
- Two task-exclusive lanes. The first 16 scenes are a quality-only pilot.
  A scientific success/failure never changes which remaining scenes run.
- Both attribution arms update attribution and belief at every later block;
  this is not a one-shot candidate substitution.
- Early terminal prefixes remain in the denominator and are separately
  flagged as not reaching the intervention. No successful-scene filtering.

## What gain zero does and does not mean

The current score is a candidate-effect backbone plus a bounded learned cause
residual. Its cause gain is zero, so that learned correction contributes
nothing. Attribution still changes belief, gates and evidence routing.
Training calibration at gain0.125 gave no improvement over gain0; gain0.25
introduced a success harm. Neither the learned cause-score contribution nor
the necessity of attribution is currently proven. A future repair should
learn cause-by-candidate-recovery interactions from paired outcomes and compare
against the exact same backbone using development data only. It must not
merely turn on a nonzero gain to manufacture a component claim.

## Reporting and stopping

Report all-scene success rate, paired gains/harms, actual fallback and costs.
For efficiency, compare total steps and logical WAM calls on the exact same
jointly successful scenes, with sample count, mean and median. Cache savings
are not policy efficiency. Success harms are baseline success/full failure,
not physical injuries; risk proxies are not safety guarantees.

Separately assess full versus value-only, no-dependency versus candidate-only,
and full versus no-dependency. A tie does not establish component necessity.
This small study is descriptive, not a statistical-significance claim.
No root oracle rollouts are added: candidate-pool coverage is unmeasured in
this study and must not be substituted by the union of method successes.

The queue finishes fixed membership, audits quality and reports outcomes;
it does not tune policies, retrain models or start the 20-task main experiment.
Engineering errors or disk free space below15GiB stop the queue and preserve
evidence. No automatic retry of a scientific failure is permitted.

## Reproduction boundary

`scripts/run_known_task_ablation.py` uses the frozen execution adapter closure
listed and hashed in the cloud protocol. The public mechanism modules and
tests are included here; upstream checkpoints/assets and machine-specific
historical adapter files are not bundled. The full cloud protocol and
state-inventory audit are retained with the experiment. Public code alone is
not yet a turnkey reproduction of the physics execution environment.
`baseline_consensus.py` is retained only to reproduce a failed historical
route in regression tests, not as the current deployed selector.
