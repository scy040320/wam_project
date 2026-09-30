# Evaluation protocol

## Fair comparison

All methods use the same Cosmos checkpoint, initial state, intervention,
candidate pool, action horizon, query budget, and episode execution budget.
The required baselines are:

- Cosmos direct execution with `K=1`;
- official value-only Best-of-N;
- binary mismatch rejection or global replanning;
- flat attribution without dependency propagation;
- hierarchical attribution without dependency propagation;
- the full belief-constrained selector;
- an outcome oracle reported only as a ceiling.

## Data separation

Split by `(task, initial state)`. Both intervention moments and all conditions
from a group remain in the same split. `clean_b` is paired-restoration QC and
never supervised training data. Confirmation data cannot be used for fitting,
threshold selection, feature design, task eligibility, or early stopping.

Simulator state and intervention metadata may schedule and audit experiments,
but they are excluded from deployment features. Candidate success and cost are
training targets or evaluation outcomes, never selector inputs.

## Candidate-pool reporting

Report three layers separately:

1. coverage: pools with at least one successful candidate;
2. conditional selection: success among covered pools;
3. overall performance: task success, harm relative to value-only, fallback,
   and zero-coverage rates over every frozen pool.

Zero-coverage and all-success pools remain in the denominator. Tasks or states
cannot be replaced after candidate outcomes are observed.

## Metrics

Primary metrics are task success, clean-scene degradation, value-success harm,
false rejection, and fallback frequency. Secondary efficiency metrics are
successful-run steps, continuation WAM calls, latency, and risk proxies.
Risk proxies must not be described as physical safety guarantees.

All weights and thresholds are frozen on development data before independent
evaluation. Oracle attribution and post-outcome best-candidate results are
upper bounds, not deployable methods.
