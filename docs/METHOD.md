# Method contract

The public mechanism operates at a complete native action-block boundary. It does not claim within-block future prediction, exact 6D object state, or simulator-ground-truth access at deployment.

## Hierarchical attribution and evidence routing

D18-v5 learns the five-way coarse cause plus four factors that have deployment evidence: visual corruption, object/environment-state change, execution/contact deviation, and cross-view conflict. Action-record reliability is not a learned head. The acquisition contract supplies `evidence.action_record_available`; an unavailable record deterministically projects to `unknown`, and that bit is masked from the neural input.

The existing visual features did not meet the preregistered feasibility gate for identifying a 5 mm weak displacement. Such records therefore use `cause_unresolved / evidence_insufficient`. This is an abstention state, not a claim that the model identified a hidden physical intervention. Original intervention metadata is retained only for offline audit.

## Belief update

Each task fact is `true`, `false`, or `unknown`, with confidence, evidence IDs and update time. A confident `false` world-state factor invalidates the target pose; an `unknown` factor propagates uncertainty without asserting a false physical fact. Execution/contact follows the same distinction. Observation unreliability must not by itself negate an already established physical state.

## Candidate decision

For each native `16 × 7` action block, the rule parser proposes a stage, prerequisites, effects and confidence. A candidate with a high-confidence prerequisite contradiction is rejected before scoring. Accepted candidates combine official value, attribution compatibility, expected progress, dependency risk and uncertainty. If none is feasible, the system emits `reobserve`, `requery`, or `safe_reject`.

## Planned action refinement

After reranking is validated, matched records will link an original candidate, its execution result, and a better candidate or corrected action. The first trainable action-generation extension is an external belief-conditioned model producing a bounded `16 × 7` residual. An internal Cosmos adapter is considered only if experiments show that candidate-set coverage, rather than selection, is the limiting factor.

## Gate boundary

The frozen D18-v5 development contract passes its validation thresholds. Engineering status remains blocked until the untouched one-shot confirmation protocol in `CONFIRMATION_PROTOCOL.md` runs without feeding any result back into training, thresholds, rules or the silver-label schema.
