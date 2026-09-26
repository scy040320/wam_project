# Evaluation protocol

All comparisons must use the same Cosmos checkpoint, candidate count `K`, action horizon, query budget and environment budget.

Required baselines are Cosmos `K=1`, Cosmos Best-of-N by official value, binary mismatch reject/replan, coarse-label attribution without a dependency graph, hierarchical attribution without propagation, and full hierarchical belief-constrained reranking.

Report task success, clean-scene degradation, unsafe-action proxies, false rejection, successful-run steps, query count and latency. A reranker is useful only if it changes selected actions for an auditable reason and improves a preregistered task/safety/step metric without unacceptable clean degradation.

Split by episode or initial-state group. Blocks from the same trajectory must never cross training, validation and test splits. Simulator state and intervention labels may schedule or evaluate an experiment but must not enter deployment features. Weights and thresholds are frozen on the development split; oracle best-candidate results are an upper bound, not a deployable baseline.

## Paired counterfactual collection

Every condition in a group starts from one complete runtime snapshot and uses the same query observation, predicted future and planned action block. Each arm performs the same reset-and-restore procedure before execution. `clean_a` is the normal model sample; `clean_b` is QC-only.

The hard physical-pairing gate compares `clean_a` with `clean_b` after identical restoration. It retains the frozen simulation-state, dual-camera and proprioception thresholds. A previous-step cached observation compared with a forced observation after restore is diagnostic only: rebuilding the simulator observation cache is not itself a counterfactual branch mismatch.

Pilot shards must all pass strict audit before full collection begins. Dataset releases, audit revisions and failed diagnostic versions remain separately named; an audit-only repair must not silently mix raw samples, label schemas, splits or thresholds.
