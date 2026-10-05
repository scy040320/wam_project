"""Shared candidate backbone with an evidence-conditioned, bounded residual.

This is a new policy contract, not a reinterpretation of frozen models.  RGB
terminal progress proxies are not physical safety certificates.  The existing
selector and its historical feature schema remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph, update_belief
from .candidate_effects import parse_candidate_effect
from .candidate_utility import CandidateUtilityModel, FEATURE_NAMES
from .contracts import (
    AttributionOutput, BeliefState, CandidateDecision, CandidateEffect,
    CoarseCause, ConsistencyFactor, ScoreWeights, TriValue,
)
from .policy import refresh_from_current_observation, _epistemic_only_rejection
from .reranker import evaluate_candidate


CAUSE_FEATURE_NAMES = (
    "world_relation_recovery", "world_relation_regression",
    "world_target_motion", "execution_contact_recovery",
    "execution_contact_regression", "execution_release_support",
    "occlusion_visual_uncertainty", "unresolved_visual_uncertainty",
    "dependency_violation", "dependency_uncertainty",
)
TERMINAL_PROGRESS_PROXIES = frozenset({"predicted_target_anchor_relation_regresses"})


def candidate_backbone_features(effect: CandidateEffect, value: float) -> np.ndarray:
    """Exactly reproduce the frozen candidate-only control's deployable inputs."""
    e = effect.evidence
    values = [
        float(value), float(effect.confidence),
        float(e.get("relation_score_after", 0)), float(e.get("relation_score_delta", 0)),
        float(e.get("relation_confidence", 0)), float(e.get("cross_view_agreement", 0)),
        float(e.get("grasp_support_after", 0)), float(e.get("grasp_support_delta", 0)),
        float(e.get("contact_confidence", 0)), float(e.get("target_gripper_distance_after", 1)),
        float(e.get("target_gripper_distance_after", 1)) - float(e.get("target_gripper_distance_before", 1)),
        float(e.get("target_displacement", 0)), float(e.get("predicted_grasp_support", 0)),
        float(e.get("predicted_release_support", 0)), float(e.get("trajectory_path_efficiency", 0)),
        float(e.get("trajectory_risk", 0)),
        *(float(effect.stage.value == s) for s in ("approach", "grasp", "lift", "transport", "place")),
        0., 0., 0., 0., 0., 0.,
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("invalid candidate backbone feature contract")
    return result


def typed_gate_effect(effect: CandidateEffect, relation: str) -> CandidateEffect:
    """Separate terminal progress proxies from witnessed prerequisite violations.

    No hard-confidence threshold is lowered.  An RGB endpoint that regresses
    can describe a useful intermediate detour and is retained as a soft risk.
    Articulated manipulation never requires carrying/lifting the whole drawer.
    """
    requirements = dict(effect.required_facts)
    proposals = dict(effect.proposed_effects)
    evidence = dict(effect.evidence)
    violations = dict(effect.hard_violations)
    for reason in TERMINAL_PROGRESS_PROXIES:
        if reason in violations:
            evidence["terminal_progress_proxy_risk"] = float(violations.pop(reason))
    if str(relation).lower() in {"open", "closed", "articulated"}:
        requirements = {"target_visible": 0.5, "target_pose_current": 0.6}
        proposals = {}
        evidence["articulated_operation"] = 1.0
    return replace(effect, required_facts=requirements, proposed_effects=proposals,
                   evidence=evidence, hard_violations=violations)


def cause_residual_features(effect: CandidateEffect, attr: AttributionOutput,
                            decision: CandidateDecision) -> np.ndarray:
    """Only supported cause/effect interactions, never task IDs or outcomes."""
    e = effect.evidence
    quality = attr.evidence_quality
    observed = attr.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is TriValue.TRUE
    resolved = attr.factor_state(ConsistencyFactor.CAUSE_RESOLVED) is TriValue.TRUE
    usable = observed and resolved and not quality.cross_view_conflict
    world = (1 - attr.factor(ConsistencyFactor.WORLD_STATE_CONSISTENT)) if usable and attr.projected_cause is CoarseCause.OBJECT_SHIFT else 0.
    execution = (1 - attr.factor(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT)) if usable and quality.execution_reliable and attr.projected_cause is CoarseCause.EXECUTION_CONTACT_DEVIATION else 0.
    # A localizer's contrast is not calibrated physical certainty. Requiring
    # view agreement prevents one sharp but incompatible map from dominating.
    relation_quality = min(float(e.get("relation_confidence", 0)), float(e.get("cross_view_agreement", 0)))
    contact_quality = min(float(e.get("contact_confidence", 0)), float(e.get("cross_view_agreement", 0)))
    delta = float(e.get("relation_score_delta", 0))
    grasp_delta = float(e.get("grasp_support_delta", 0))
    uncertainty = 1 - effect.confidence + float(e.get("trajectory_risk", 0)) + 0.5 * (1 - float(e.get("cross_view_agreement", 0)))
    occlusion = float(attr.projected_cause is CoarseCause.VISUAL_OCCLUSION)
    unresolved = float(attr.projected_cause is CoarseCause.UNKNOWN)
    dependency_route = max(world, execution)
    result = np.asarray([
        world * relation_quality * max(delta, 0),
        world * relation_quality * max(-delta, 0),
        world * relation_quality * float(e.get("target_displacement", 0)),
        execution * contact_quality * max(grasp_delta, 0),
        execution * contact_quality * max(-grasp_delta, 0),
        execution * contact_quality * float(e.get("predicted_release_support", 0)),
        occlusion * uncertainty, unresolved * uncertainty,
        dependency_route * float(decision.components.get("dependency_risk", 0)),
        dependency_route * float(decision.components.get("uncertainty", 0)),
    ], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite cause residual evidence")
    return result


@dataclass(frozen=True)
class EvidenceResidualModel:
    scale: np.ndarray
    weights: np.ndarray
    train_abs_bound: np.ndarray
    gain: float = 0.0
    score_bound: float = 0.0
    schema: str = "shared_backbone_evidence_residual_v1"

    def __post_init__(self):
        for vector in (self.scale, self.weights, self.train_abs_bound):
            if np.asarray(vector).shape != (len(CAUSE_FEATURE_NAMES),) or not np.isfinite(vector).all():
                raise ValueError("invalid evidence residual vector")
        if np.any(self.scale <= 0) or np.any(self.train_abs_bound < 0):
            raise ValueError("invalid evidence residual support")
        if not np.isfinite([self.gain, self.score_bound]).all() or min(self.gain, self.score_bound) < 0:
            raise ValueError("invalid evidence residual bounds")

    def score(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != self.weights.shape or not np.isfinite(features).all():
            raise ValueError("invalid cause residual features")
        # Do not extrapolate to an evidence combination outside train support.
        if np.any(np.abs(features) > self.train_abs_bound + 1e-8):
            return 0.0
        raw = self.gain * float((features / self.scale) @ self.weights)
        return float(np.clip(raw, -self.score_bound, self.score_bound))

    def to_record(self):
        return {"schema": self.schema, "feature_names": list(CAUSE_FEATURE_NAMES),
                "scale": self.scale.tolist(), "weights": self.weights.tolist(),
                "train_abs_bound": self.train_abs_bound.tolist(), "gain": self.gain,
                "score_bound": self.score_bound}

    @classmethod
    def from_record(cls, record):
        if tuple(record["feature_names"]) != CAUSE_FEATURE_NAMES or record["schema"] != "shared_backbone_evidence_residual_v1":
            raise ValueError("evidence residual schema mismatch")
        return cls(np.asarray(record["scale"]), np.asarray(record["weights"]),
                   np.asarray(record["train_abs_bound"]), float(record["gain"]),
                   float(record["score_bound"]))


def fit_evidence_residual(pairs: Sequence[tuple[np.ndarray, np.ndarray, float]],
                          support: Sequence[np.ndarray]) -> EvidenceResidualModel:
    """Train on development preferences; the candidate backbone stays frozen."""
    support = np.asarray(support, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != len(CAUSE_FEATURE_NAMES) or not np.isfinite(support).all():
        raise ValueError("invalid training evidence support")
    scale = support.std(axis=0)
    scale[scale < 1e-8] = 1.
    bound = np.max(np.abs(support), axis=0)
    weights = np.zeros(len(CAUSE_FEATURE_NAMES))
    if pairs:
        delta = np.stack([(a - b) / scale for a, b, _ in pairs])
        base = np.asarray([margin for _, _, margin in pairs])
        for _ in range(2000):
            probability = 1 / (1 + np.exp(-np.clip(base + delta @ weights, -30, 30)))
            gradient = -((1 - probability)[:, None] * delta).mean(axis=0) + weights
            weights -= .05 * gradient
    raw_scores = (support / scale) @ weights
    return EvidenceResidualModel(scale, weights, bound, 1., float(np.max(np.abs(raw_scores))))


def prepare_evidence_decisions(*, belief: BeliefState, attribution: AttributionOutput,
                               block_index: int, effects: Sequence[CandidateEffect],
                               values: Sequence[float], relation: str,
                               graph: DependencyGraph = DEFAULT_GRAPH,
                               backbone: CandidateUtilityModel | None = None) -> tuple[CandidateDecision, ...]:
    update_belief(belief, attribution, block_index, graph=graph)
    refresh_from_current_observation(belief, attribution, block_index)
    weights = ScoreWeights(0., 0., 0., 0., calibrated=True)
    decisions = [evaluate_candidate(belief, attribution, typed_gate_effect(e, relation),
                                   float(v), weights) for e, v in zip(effects, values, strict=True)]
    anchors = {max(decisions, key=lambda x: (x.official_value, -x.candidate_id)).candidate_id}
    if backbone is not None:
        unfiltered = [replace(d, accepted=True, rejection_reasons=(),
                              components={"dependency_risk": 0., "uncertainty": 0.}) for d in decisions]
        from .candidate_utility import select_with_utility
        base, _ = select_with_utility(unfiltered,
            {e.candidate_id: candidate_backbone_features(e, float(v)) for e,v in zip(effects,values,strict=True)}, backbone)
        anchors.add(base.candidate_id)
    # This reproduces the existing epistemic-only fallback. Direct FALSE
    # facts and predicted missing contact remain non-overridable.
    decisions = [replace(d, accepted=True, total_score=d.official_value,
                         rejection_reasons=(), components={**d.components,
                         "baseline_preserving_epistemic_override": 1.})
                 if d.candidate_id in anchors and not d.accepted and _epistemic_only_rejection(d) else d
                 for d in decisions]
    insufficient = (
        attribution.projected_cause is CoarseCause.UNKNOWN
        or attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE) is not TriValue.TRUE
        or attribution.factor_state(ConsistencyFactor.CAUSE_RESOLVED) is not TriValue.TRUE
        or attribution.evidence_quality.cross_view_conflict
    )
    if insufficient:
        decisions = [replace(d, components={**d.components, "preserve_official_value": 1.}) for d in decisions]
    return tuple(decisions)


def select_evidence_residual(*, decisions: Sequence[CandidateDecision],
                             backbone_features: Mapping[int, np.ndarray],
                             cause_features: Mapping[int, np.ndarray],
                             backbone: CandidateUtilityModel,
                             residual: EvidenceResidualModel) -> tuple[CandidateDecision | None, dict[int, float]]:
    """Use the same trained candidate backbone in both full and control arms."""
    accepted = [d for d in decisions if d.accepted]
    if not accepted:
        return None, {}
    if any(d.components.get("preserve_official_value", 0.) for d in accepted):
        scores = {d.candidate_id: d.official_value for d in accepted}
        return max(accepted, key=lambda d: (d.official_value, -d.candidate_id)), scores
    base_scores = {d.candidate_id: backbone.score(backbone_features[d.candidate_id]) for d in accepted}
    # Each score correction is bounded; the pairwise change is at most
    # twice the bound. The bound is stored explicitly in the model record.
    corrections = {d.candidate_id: residual.score(cause_features[d.candidate_id]) for d in accepted}
    scores = {k: base_scores[k] + corrections[k] for k in base_scores}
    margin = max(.01, backbone.switch_margin)
    top = max(scores.values())
    equivalent = [d for d in accepted if top - scores[d.candidate_id] <= margin]
    chosen = max(equivalent, key=lambda d: (d.official_value, -d.candidate_id))
    # Witnessed prerequisite violations have already been hard gated. Soft
    # image-space risk enters the learned residual; it must not introduce a
    # second, uncalibrated veto that the shared candidate control lacks.
    return chosen, scores
