# Native-16 paired A/B/C/D recovery comparison

## Scope

This study tests whether cause-aware, dependency-aware recovery reduces unnecessary
state invalidation and global refreshes while preserving task success. It is a
small-scale controlled comparison, not the final paper matrix.

The compared methods are:

- **A — binary global:** image mismatch triggers a global refresh;
- **B — uniform subgraph:** mismatch invalidates a fixed large subgraph;
- **C — attribution global:** predicted cause selects a response, but state
  invalidation remains global;
- **D — dependency aware:** predicted cause routes to the minimal reviewed
  dependency closure and the frozen V6 unknown-safety guard.

## Preregistered protocol

- frozen Cosmos Policy checkpoint and native 16-action execution horizon;
- validation-frozen attributor, thresholds, and recovery rules;
- same task, initial state, condition, intervention block, and budget for all
  four methods in a matched comparison cell;
- cyclic method-order rotation to balance run-order effects;
- one deterministic condition per task/seed cell for the screening run;
- no adaptation from comparison outcomes.

LIBERO exposes only 50 initial states for each selected task. States 0–23 were
used for training and 24–31 for validation/calibration. Consequently, only
states 32–49 are eligible as fully held-out primary states. The runner executes
the planned 20 states per task, but states 30–31 are tagged
`supplemental_validation_overlap` and permanently excluded from primary
inference.

| Population | Jobs | Statistical use |
|---|---:|---|
| Tasks 0–3, seeds 32–49, four methods | 288 | primary held-out analysis |
| Tasks 0–3, seeds 30–31, four methods | 32 | supplemental diagnostics only |
| Total completed | 320 | integrity accounting |

## Primary results

All 320 decision JSON files and 640 camera videos were non-empty. The runner
reported no OOM, dead loop, or runtime-fatal traceback. MuJoCo/EGL destructor
messages emitted after completed episodes are cleanup warnings.

| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Episodes | 72 | 72 | 72 | 72 |
| Task success | 59.7% | 59.7% | 55.6% | 59.7% |
| Correct intervention-block recovery | 5.6% | 5.6% | 31.9% | 90.3% |
| Safety-stop rate | 0.0% | 0.0% | 38.9% | 23.6% |
| Mean global refreshes | 15.07 | 0.00 | 6.88 | 0.24 |
| Mean invalidated nodes | 140.22 | 95.93 | 64.61 | 13.18 |
| Mean WAM calls | 22.69 | 22.72 | 14.33 | 16.07 |

For method D:

- clean task success: 15/15;
- visual-occlusion task success: 14/14;
- action-noise task success: 14/14;
- unknown safe stop: 15/15;
- object-shift `local_state_update` selection: 13/14;
- object-shift final task success: 0/14.

## Decision and limitation

The comparison passes its engineering, fairness, safety, and local-invalidation checks,
but carries a material limitation. Correctly routing an object shift to a
local state update did not restore task success. A, B, C, and D all achieved
0% success for this condition. The held-out data must not be used to tune
the frozen thresholds or recovery rules.

The next recovery-cue study preserves the frozen rule and preregisters the complete condition matrix,
with object-shift trajectories treated as a named failure-analysis stratum.
Any method change motivated by that analysis requires a new validation-only
development version and a fresh untouched evaluation set.

Raw per-episode results, videos, checkpoints, and local experiment archives
remain outside Git.
