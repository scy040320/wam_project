"""Attribution-conditioned candidate utility ranking after safety gating.

The ranker is not allowed to treat attribution and selection as independent
problems.  Its features explicitly cross the previous-block cause evidence
with each candidate's predicted recovery effect.  It remains a small,
auditable pairwise model and can never bypass the prerequisite hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    AttributionOutput, CandidateDecision, CandidateEffect, CoarseCause,
    ConsistencyFactor, Stage,
)


FEATURE_NAMES = (
    "official_value",
    "effect_confidence",
    "world_routed_relation_score_after",
    "world_routed_relation_score_delta",
    "world_routed_relation_confidence",
    "cross_view_agreement",
    "execution_routed_grasp_support_after",
    "execution_routed_grasp_support_delta",
    "execution_routed_contact_confidence",
    "cause_routed_target_gripper_distance_after",
    "cause_routed_target_gripper_distance_delta",
    "world_routed_target_displacement",
    "execution_routed_predicted_grasp_support",
    "execution_routed_predicted_release_support",
    "trajectory_path_efficiency",
    "trajectory_risk",
    "stage_approach",
    "stage_grasp",
    "stage_lift",
    "stage_transport",
    "stage_place",
    "shift_relation_recovery",
    "execution_grasp_recovery",
    "execution_release_recovery",
    "occlusion_observe_compatibility",
    "stage_reset_compatibility",
    "unknown_candidate_uncertainty",
)

# Pairwise residual utilities are calibrated on a small development pool.
# Differences below one percentage point are treated as numerically
# equivalent and resolved by the frozen Cosmos value rather than by noise in
# the residual head.
UTILITY_EQUIVALENCE_TOLERANCE = 0.01


def candidate_utility_features(
    effect: CandidateEffect,
    official_value: float,
    attribution: AttributionOutput,
) -> np.ndarray:
    """Return candidate features conditioned on the diagnosed mismatch.

    Raw attribution probabilities alone are constant within a candidate set
    and therefore cannot teach a ranker which candidate to choose.  The final
    eight terms are interactions: they ask whether *this candidate* repairs
    the particular factor that attribution says is unreliable.
    """
    evidence = effect.evidence
    stage_names = (Stage.APPROACH, Stage.GRASP, Stage.LIFT, Stage.TRANSPORT, Stage.PLACE)
    relation_recovery = (
        float(evidence.get("relation_confidence", 0.0))
        * float(np.clip(evidence.get("relation_score_delta", 0.0), 0.0, 1.0))
    )
    contact = float(evidence.get("contact_confidence", 0.0))
    grasp_recovery = contact * max(
        float(evidence.get("current_grasped_support", 0.0)),
        float(evidence.get("predicted_grasp_support", 0.0)),
    )
    release_recovery = (
        contact
        * float(evidence.get("predicted_release_support", 0.0))
        * float(evidence.get("place_ready_support", 0.0))
    )
    observation_anomaly = 1.0 - attribution.factor(ConsistencyFactor.OBSERVATION_RELIABLE)
    world_anomaly = 1.0 - attribution.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT)
    execution_anomaly = 1.0 - attribution.factor(
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT
    )
    stage_anomaly = 1.0 - attribution.factor(ConsistencyFactor.TASK_STAGE_CONSISTENT)
    class_probability = attribution.class_probs
    # Source-aware two-key routing: positive recovery credit follows the
    # *resolved hierarchical projection*, not an unselected direct-head logit.
    # Secondary factor evidence is still available to the prerequisite gate
    # and uncertainty/risk terms, but it cannot reward an unrelated recovery
    # action.  Otherwise an OBJECT_SHIFT projection with a noisy high
    # execution logit can incorrectly rank grasp-like candidates above the
    # candidate that restores the target--anchor relation.
    projected_world = float(attribution.projected_cause is CoarseCause.OBJECT_SHIFT)
    projected_execution = float(
        attribution.projected_cause is CoarseCause.EXECUTION_CONTACT_DEVIATION
    )
    world_relevance = float(np.clip(
        world_anomaly * projected_world * float(attribution.confidence),
        0.0, 1.0,
    ))
    execution_relevance = float(np.clip(
        execution_anomaly * projected_execution * float(attribution.confidence),
        0.0, 1.0,
    ))
    manipulation_relevance = max(world_relevance, execution_relevance)
    candidate_uncertainty = (
        1.0 - float(effect.confidence)
        + float(evidence.get("trajectory_risk", 0.0))
        + 0.5 * (1.0 - float(evidence.get("cross_view_agreement", 0.0)))
    )
    values = [
        float(official_value),
        float(effect.confidence),
        world_relevance * float(evidence.get("relation_score_after", 0.0)),
        world_relevance * float(evidence.get("relation_score_delta", 0.0)),
        world_relevance * float(evidence.get("relation_confidence", 0.0)),
        float(evidence.get("cross_view_agreement", 0.0)),
        execution_relevance * float(evidence.get("grasp_support_after", 0.0)),
        execution_relevance * float(evidence.get("grasp_support_delta", 0.0)),
        execution_relevance * float(evidence.get("contact_confidence", 0.0)),
        manipulation_relevance
        * float(evidence.get("target_gripper_distance_after", 1.0)),
        manipulation_relevance
        * (
            float(evidence.get("target_gripper_distance_after", 1.0))
            - float(evidence.get("target_gripper_distance_before", 1.0))
        ),
        world_relevance * float(evidence.get("target_displacement", 0.0)),
        execution_relevance * float(evidence.get("predicted_grasp_support", 0.0)),
        execution_relevance * float(evidence.get("predicted_release_support", 0.0)),
        float(evidence.get("trajectory_path_efficiency", 0.0)),
        float(evidence.get("trajectory_risk", 0.0)),
        *(1.0 if effect.stage is stage else 0.0 for stage in stage_names),
        world_relevance * relation_recovery,
        execution_relevance * grasp_recovery,
        execution_relevance * release_recovery,
        float(class_probability[CoarseCause.VISUAL_OCCLUSION.value])
        * observation_anomaly
        * (1.0 if effect.stage is Stage.OBSERVE else 0.0),
        stage_anomaly
        * (1.0 if effect.stage in {Stage.OBSERVE, Stage.APPROACH, Stage.GRASP} else 0.0),
        float(class_probability[CoarseCause.UNKNOWN.value]) * candidate_uncertainty,
    ]
    output = np.asarray(values, dtype=np.float64)
    if output.shape != (len(FEATURE_NAMES),) or not np.isfinite(output).all():
        raise ValueError("candidate utility features must be finite and match the frozen schema")
    return output


@dataclass(frozen=True)
class CandidateUtilityModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    feature_names: tuple[str, ...] = FEATURE_NAMES
    schema: str = "d21_value_anchored_projected_cause_routed_residual_v5"
    # A residual candidate may replace the official-value anchor only when
    # its learned advantage exceeds this train-only calibrated margin.
    # This keeps weak/noisy residuals from damaging a successful baseline.
    switch_margin: float = UTILITY_EQUIVALENCE_TOLERANCE

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        arrays = tuple(np.asarray(item, dtype=np.float64) for item in (self.mean, self.scale, self.weights))
        if any(item.shape != (size,) for item in arrays):
            raise ValueError("candidate utility model vectors must match feature_names")
        if not all(np.isfinite(item).all() for item in arrays) or not np.isfinite(self.intercept):
            raise ValueError("candidate utility model must be finite")
        if np.any(arrays[1] <= 0.0):
            raise ValueError("candidate utility scale must be positive")
        if not np.isfinite(self.switch_margin) or self.switch_margin < 0.0:
            raise ValueError("candidate utility switch_margin must be finite and non-negative")

    def score(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != self.weights.shape or not np.isfinite(features).all():
            raise ValueError("invalid candidate utility feature vector")
        normalized = (features - self.mean) / self.scale
        # The frozen D16 scoring contract is value + learned compatibility,
        # not an unconstrained replacement for the official value baseline.
        # Keeping the value coefficient fixed prevents a small development
        # pool from learning to discard Cosmos' strongest native signal.
        residual = normalized @ self.weights + self.intercept
        return float(features[0] + residual)

    def to_record(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "intercept": float(self.intercept),
            "switch_margin": float(self.switch_margin),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "CandidateUtilityModel":
        names = tuple(str(item) for item in record["feature_names"])
        if names != FEATURE_NAMES:
            raise ValueError("candidate utility feature schema mismatch")
        return cls(
            np.asarray(record["mean"], dtype=np.float64),
            np.asarray(record["scale"], dtype=np.float64),
            np.asarray(record["weights"], dtype=np.float64),
            float(record["intercept"]),
            names,
            str(record["schema"]),
            float(record.get("switch_margin", UTILITY_EQUIVALENCE_TOLERANCE)),
        )


def fit_pairwise_utility(
    feature_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    l2: float = 1.0,
    learning_rate: float = 0.05,
    steps: int = 2000,
) -> CandidateUtilityModel:
    """Fit a deterministic pairwise logistic ranker.

    Every pair is ``(better, worse)`` according to outcomes in a development
    pool.  Test or confirmation outcomes must never be supplied here.
    """
    if not feature_pairs:
        raise ValueError("at least one development preference pair is required")
    better = np.stack([np.asarray(pair[0], dtype=np.float64) for pair in feature_pairs])
    worse = np.stack([np.asarray(pair[1], dtype=np.float64) for pair in feature_pairs])
    if better.shape != worse.shape or better.ndim != 2 or better.shape[1] != len(FEATURE_NAMES):
        raise ValueError("invalid pairwise feature matrix")
    all_features = np.concatenate([better, worse], axis=0)
    mean = all_features.mean(axis=0)
    scale = all_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    differences = (better - worse) / scale
    base_margin = better[:, 0] - worse[:, 0]
    # official_value is the fixed anchor and must not also be learned as a
    # free residual feature (which would double count it).
    differences[:, 0] = 0.0
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(int(steps)):
        logits = np.clip(base_margin + differences @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = -((1.0 - probability)[:, None] * differences).mean(axis=0) + l2 * weights
        weights -= learning_rate * gradient
        weights[0] = 0.0
    return CandidateUtilityModel(mean, scale, weights, 0.0)


def select_with_utility(
    decisions: Sequence[CandidateDecision],
    feature_by_candidate: Mapping[int, np.ndarray],
    model: CandidateUtilityModel,
) -> tuple[CandidateDecision | None, dict[int, float]]:
    """Rank only candidates already accepted by the safety gate."""
    accepted = [item for item in decisions if item.accepted]
    scores = {
        item.candidate_id: model.score(feature_by_candidate[item.candidate_id])
        for item in accepted
    }
    if not accepted:
        return None, scores
    best_score = max(scores.values())
    equivalence_tolerance = max(
        UTILITY_EQUIVALENCE_TOLERANCE, float(model.switch_margin)
    )
    equivalent = [
        item for item in accepted
        if best_score - scores[item.candidate_id] <= equivalence_tolerance
    ]
    chosen = max(equivalent, key=lambda item: (item.official_value, -item.candidate_id))

    # A learned residual may not replace the highest-value feasible candidate
    # with one that is strictly worse on *both* auditable safety axes.  This is
    # a parameter-free Pareto guard: it does not inspect outcomes and it still
    # permits a lower-value candidate when it improves dependency risk or
    # uncertainty.
    value_anchor = max(
        accepted, key=lambda item: (item.official_value, -item.candidate_id)
    )
    if chosen.candidate_id != value_anchor.candidate_id:
        chosen_risk = float(chosen.components.get("dependency_risk", 0.0))
        anchor_risk = float(value_anchor.components.get("dependency_risk", 0.0))
        chosen_uncertainty = float(chosen.components.get("uncertainty", 0.0))
        anchor_uncertainty = float(value_anchor.components.get("uncertainty", 0.0))
        if chosen_risk > anchor_risk and chosen_uncertainty > anchor_uncertainty:
            chosen = value_anchor
    return chosen, scores
