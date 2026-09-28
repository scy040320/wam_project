# D19–D21 V5 independent gate

## Frozen protocol

- Tasks and states: LIBERO-90 task78/state35 and task81/state36.
- Conditions: clean, object shift, and execution/contact deviation.
- Candidate budget: `K=4` per scenario, with one explicit requery fallback when required.
- Selection tracks: official value-only, oracle-attribution hard gate, and frozen D18-v18i learned-attribution hard gate.
- Selector rule: source-aware predicate gate, then official Cosmos value as the tie-breaker among feasible candidates.
- No uncalibrated soft-score weights are used.

## Result

All six scenarios completed with valid evidence and candidate logs. Value-only, oracle-attribution and learned-attribution each obtained `0/6` successful outcomes. None of the 24 direct candidates or six requery fallbacks succeeded, so the oracle ceiling was also `0/6`.

The learned attribution track classified all three task78 scenarios correctly. For task81, object shift was predicted as normal while clean and execution/contact deviation were correct. This error is retained, but it is not the primary reason the gate failed because the oracle track also had no successful candidate to select.

## Decision

V5 is frozen and is not tuned against these six scenarios. The result shows no observed clean harm but does not show a benefit. The bottleneck in this gate is candidate-set coverage.

The next preregistered evaluation must report two separate quantities:

1. candidate-pool coverage / oracle ceiling over every fixed task and state, including qualification failures;
2. selector performance conditional on at least one successful candidate being present.

Only the second quantity can test whether belief constraints outperform value-only selection. Reporting both prevents post-hoc task filtering from being mistaken for selector improvement.
