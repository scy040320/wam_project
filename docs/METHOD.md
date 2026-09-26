# Method contract

The public mechanism operates at a complete native action-block boundary. It does not claim within-block future prediction, exact 6D object state, or simulator-ground-truth access at deployment.

## Hierarchical attribution

The interface predicts five consistency factors that may coexist: observation reliability, world-state consistency, execution/contact consistency, task-stage consistency, and whether the cause is resolved. It also exports a five-way coarse projection for evaluation compatibility. An unresolved cause or reliable cross-view conflict must project to `unknown`.

## Belief update

Each task fact is `true`, `false`, or `unknown`, with confidence, evidence IDs and update time. World-state inconsistency invalidates the target pose and propagates uncertainty to pose-dependent descendants. Execution/contact inconsistency invalidates execution consistency and its descendants. Observation unreliability must not by itself negate an already established physical state.

## Candidate decision

For each native `16 × 7` action block, the rule parser proposes a stage, prerequisites, effects and confidence. A candidate with a high-confidence prerequisite contradiction is rejected before scoring. Accepted candidates combine official value, attribution compatibility, expected progress, dependency risk and uncertainty. If none is feasible, the system emits `reobserve`, `requery`, or `safe_reject`.

## Planned action refinement

After reranking is validated, matched records will link an original candidate, its execution result, and a better candidate or corrected action. The first trainable action-generation extension is an external belief-conditioned model producing a bounded `16 × 7` residual. An internal Cosmos adapter is considered only if experiments show that candidate-set coverage, rather than selection, is the limiting factor.
