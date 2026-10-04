# Method contract

## Scope

CF-WAM operates at a complete native action-block boundary. It does not claim
within-block future prediction, exact 6D object-state recovery, or
simulator-ground-truth access at deployment.

## Hierarchical attribution

The attributor consumes the previous block's predicted and real terminal
observations, planned and applied actions, proprioception, and task language.
It emits a coarse cause distribution plus four deployable evidence factors:

- visual evidence corrupted;
- object or environment state changed;
- execution or contact deviated;
- cross-view conflict.

Action-record availability and explicit requested-versus-applied command
deviation are acquisition-contract hard routes, not learned image features.
Changes whose physical cause is not observable are assigned
`cause_unresolved / evidence_insufficient`; the corresponding physical-factor
loss is masked.

## Belief update

Each task fact is `true`, `false`, or `unknown`, with confidence, provenance,
evidence IDs, and update time. Attribution invalidates the minimum supported
fact and propagates only through the dependency graph. Observation
unreliability does not negate an already established physical state.

New observation evidence or a candidate's predicted effects may re-establish
a fact only through its predicate-specific rule. A generic confidence increase
cannot override a high-confidence contradiction.

## Candidate parsing and selection

For each native `16 × 7` action block, the parser exposes:

- required and proposed facts;
- target displacement;
- target–anchor relation change;
- grasp and release support;
- trajectory smoothness and risk proxies.

Selection proceeds in two stages. The hard gate rejects candidates that
violate a reliable prerequisite. The utility model then ranks the feasible
set using official value, attribution compatibility, expected progress,
dependency risk, uncertainty, and execution cost. Near-equal learned utility
returns to official value, and a Pareto guard blocks a choice that is worse
than the value anchor on both dependency risk and uncertainty.

If no candidate is feasible, the controller emits an explicit `reobserve`,
`requery`, `safe_reject`, or `safe_stop` command under a bounded recovery
budget. An unexecuted fallback has an unobserved outcome.

The adopted safe-residual policy preserves the official-value anchor when
its only rejection reasons are unknown prerequisites. This is a baseline
preservation rule, not a safety certificate: explicit false predicates and
candidate-specific hard violations still forbid that candidate. A train-only
calibrated utility margin controls deviations from the anchor. The attributor
remains frozen while fitting this downstream utility model.
