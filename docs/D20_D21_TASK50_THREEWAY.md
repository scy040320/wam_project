# D20–D21 Frozen Task50 Three-Way Comparison

## Purpose

This mechanism pilot asks two separate questions:

1. Does the frozen `K=8` candidate pool contain a better action than official `argmax(value)`?
2. Can the frozen learned attribution and belief gate select that action without outcome labels?

The six scenarios are task50 object-shift states `32, 34, 35, 37, 38, 39`.
They were completed before selector analysis. No candidate was regenerated or
discarded after its outcome was viewed.

## Track definitions

- `value_only`: official Cosmos maximum predicted value.
- `outcome_oracle`: post-outcome ceiling, ordered by success, fewer executed
  steps, fewer continuation WAM calls and candidate id.
- `learned`: frozen D18-v18i hierarchical attribution, V6 candidate-effect
  parser, belief update and hard gate; official value breaks ties among
  feasible candidates.

The outcome oracle is not the attribution oracle used in earlier diagnostics.
It is a theoretical candidate-selection upper bound.

## Frozen results

| Measure | Value-only | Outcome oracle | Learned |
|---|---:|---:|---:|
| Selected candidates | 6 | 6 | 2 |
| Explicit fallback | 0 | 0 | 4 |
| Observed successes | 2 | 4 | 1 |
| Observed failures | 4 | 2 | 1 |
| Overall success lower bound | 2/6 | 4/6 | 1/6 |
| Successes when pool contains success | 2/4 | 4/4 | 1/4 lower bound |

Candidate-pool coverage is `4/6`. Outcome oracle has four strict improvements
and no harms relative to value-only. In the two scenarios where both tracks
succeed, oracle uses `51.5` fewer executed steps and `3` fewer continuation WAM
calls on average.

The learned track's four fallback outcomes were not executed. They are
unobserved, not four observed failures. One fallback replaced a value-only
candidate that was known to succeed, so the abstention still exposes a real
conservatism risk.

## Diagnosis and decision

D20 passes only as an oracle-feasibility result: candidate choice can matter.
D21 does not pass. Object-shift attribution invalidates the old target pose,
then the dependency graph propagates uncertainty into `grasped`, `lifted` and
`place_ready`. The hard gate consequently rejects every candidate in several
scenarios. In state39 it accepts candidates but still follows the highest-value
failure.

The six scenarios are now frozen. They must not be used to tune thresholds or
rules. The next implementation change belongs to a development pool and must
specify how current observation or candidate-specific effects can re-establish
previously invalidated prerequisites. Any revised learned selector requires a
new-task validation.
