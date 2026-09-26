# Evaluation protocol

All comparisons must use the same Cosmos checkpoint, candidate count `K`, action horizon, query budget and environment budget.

Required baselines are Cosmos `K=1`, Cosmos Best-of-N by official value, binary mismatch reject/replan, coarse-label attribution without a dependency graph, hierarchical attribution without propagation, and full hierarchical belief-constrained reranking.

Report task success, clean-scene degradation, unsafe-action proxies, false rejection, successful-run steps, query count and latency. A reranker is useful only if it changes selected actions for an auditable reason and improves a preregistered task/safety/step metric without unacceptable clean degradation.

Split by episode or initial-state group. Blocks from the same trajectory must never cross training, validation and test splits. Simulator state and intervention labels may schedule or evaluate an experiment but must not enter deployment features. Weights and thresholds are frozen on the development split; oracle best-candidate results are an upper bound, not a deployable baseline.
