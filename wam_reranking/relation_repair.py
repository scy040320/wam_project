"""Conservative object-state repair over a frozen attribution model.

Only deployable evidence is accepted. Training labels, task identifiers and
paired clean control features are never inference inputs. A new relation head
cannot replace an existing known cause or bypass hard evidence routes.
"""
from __future__ import annotations

import numpy as np


def relation_repair_features(normalized: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2298 or not np.isfinite(x).all():
        raise ValueError("expected finite normalized N x 2298 deployment features")
    # Local pixel/edge residuals, semantics, relation deltas and their magnitude.
    # No global scene embedding, task/state identity, action command shortcut,
    # or paired clean endpoint is supplied to the trainable repair head.
    local = x[:, 2138:2225]
    relation = x[:, 2225:2297]
    residual_indices = np.asarray(
        list(range(0, 6)) + list(range(16, 22))
        + list(range(32, 36)) + list(range(40, 44)) + [54, 66]
    )
    return np.concatenate([local, relation, np.abs(relation[:, residual_indices])], axis=1)


def apply_relation_repair(
    baseline_pred: np.ndarray,
    baseline_factors: np.ndarray,
    repair_probability: np.ndarray,
    *,
    target_available: np.ndarray,
    observation_reliable: np.ndarray,
    action_unreliable: np.ndarray,
    command_deviation: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(baseline_pred).copy()
    factors = np.asarray(baseline_factors, dtype=np.float32).copy()
    probability = np.asarray(repair_probability, dtype=np.float32)
    n = len(pred)
    if factors.shape != (n, 4) or probability.shape != (n,):
        raise ValueError("prediction/factor/repair dimensions differ")
    if not np.isfinite(factors).all() or not np.isfinite(probability).all():
        raise ValueError("nonfinite probabilities")
    if ((factors < 0) | (factors > 1)).any() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities outside [0,1]")
    inputs = [target_available, observation_reliable, action_unreliable, command_deviation]
    if any(np.asarray(a).shape != (n,) for a in inputs):
        raise ValueError("evidence dimensions differ")
    eligible = (
        np.isin(pred, [0, 4]) & (factors < threshold).all(axis=1)
        & np.asarray(target_available, dtype=bool)
        & np.asarray(observation_reliable, dtype=bool)
        & ~np.asarray(action_unreliable, dtype=bool)
        & ~np.asarray(command_deviation, dtype=bool)
    )
    activated = eligible & (probability >= threshold)
    factors[activated, 1] = np.maximum(factors[activated, 1], probability[activated])
    pred[activated] = 2
    return pred, factors, activated
