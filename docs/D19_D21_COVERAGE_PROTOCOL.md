# D19–D21 Preregistered Candidate-Coverage Protocol

This qualification study separates generator coverage from selector quality. The task and state pool was frozen before any outcome was observed.

## Fixed pool

- LIBERO-90 tasks: `11, 18, 27, 35, 46, 57, 68, 86`
- States: `37, 38, 39, 40, 41, 42, 43, 44`, paired with the tasks in order
- Conditions: `clean`, `object_shift`, `execution_contact_deviation`
- Candidate count: `K=4`
- Fixed scenarios: `8 × 1 × 3 = 24`
- Every direct candidate is executed from the same paired start state.
- A failed qualification scenario remains in the denominator. Tasks are not filtered after seeing results.

The pool covers articulated opening, rigid placement, containment and support relations. The frozen V5 source-aware predicate gates, relation parser, attribution model and fallback rules are not recalibrated on these scenarios.

## Required reports

### 1. Candidate-pool coverage

For all 24 fixed scenarios:

```text
coverage = scenarios with at least one successful direct candidate / 24
```

This is the empirical oracle ceiling of the generated pool. A scenario without a successful candidate cannot be used as evidence that one selector is better than another.

### 2. Conditional selector success

On the subset of scenarios with at least one successful candidate, report the numerator, denominator and rate for:

- official `value-only` selection;
- oracle-attribution hard gating;
- frozen learned-attribution hard gating.

The denominator must be shown even when it is zero.

### 3. Overall closed-loop outcomes

Across all 24 fixed scenarios, report for each selection track:

- task successes and overall task-success rate;
- strict harms relative to value-only;
- clean-scene harms;
- fallback counts and reasons.

Coverage and conditional selector accuracy must not be merged into one headline number. A selector cannot compensate for a candidate generator whose pool contains no successful action.

## Freeze and failure handling

- The six V5 task78/task81 scenarios remain a completed negative result and are not used to tune this protocol.
- An engineering failure before candidate execution creates a new implementation version; it does not change the scientific pool.
- Model weights, thresholds, task bindings, conditions, `K`, states and reporting formulas remain frozen throughout the qualification run.
- No D22–D28 benefit claim is opened unless the fixed pool establishes non-zero coverage and the selector comparison has a valid conditional denominator.
