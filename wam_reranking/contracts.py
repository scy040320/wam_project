"""Typed, simulator-independent contracts used by the reranking layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Mapping

import numpy as np


class CoarseCause(str, Enum):
    NORMAL = "normal"
    VISUAL_OCCLUSION = "visual_occlusion"
    OBJECT_SHIFT = "object_shift"
    EXECUTION_CONTACT_DEVIATION = "execution_contact_deviation"
    UNKNOWN = "unknown"


class ConsistencyFactor(str, Enum):
    OBSERVATION_RELIABLE = "observation_reliable"
    WORLD_STATE_CONSISTENT = "world_state_consistent"
    EXECUTION_CONTACT_CONSISTENT = "execution_contact_consistent"
    TASK_STAGE_CONSISTENT = "task_stage_consistent"
    CAUSE_RESOLVED = "cause_resolved"


class TriValue(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class Stage(str, Enum):
    OBSERVE = "observe"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    UNCERTAIN = "uncertain"


PREDICATES = (
    "target_visible", "target_pose_current", "target_reachable", "grasped", "lifted",
    "receptacle_visible", "place_ready", "placed", "execution_consistent",
)


@dataclass(frozen=True)
class EvidenceQuality:
    primary_reliable: bool
    wrist_reliable: bool
    execution_reliable: bool
    cross_view_conflict: bool = False


@dataclass(frozen=True)
class AttributionOutput:
    """Hierarchical attribution plus a backwards-compatible coarse projection."""

    factor_probs: Mapping[str, float]
    class_probs: Mapping[str, float]
    projected_cause: CoarseCause
    confidence: float
    entropy: float
    evidence_quality: EvidenceQuality
    source_block_id: str
    factor_states: Mapping[str, TriValue] = field(default_factory=dict)
    factor_confidences: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_factors = {item.value for item in ConsistencyFactor}
        expected_classes = {item.value for item in CoarseCause}
        if set(self.factor_probs) != expected_factors:
            raise ValueError(f"factor_probs must contain {sorted(expected_factors)}")
        if set(self.class_probs) != expected_classes:
            raise ValueError(f"class_probs must contain {sorted(expected_classes)}")
        factors = np.asarray(list(self.factor_probs.values()), dtype=np.float64)
        classes = np.asarray(list(self.class_probs.values()), dtype=np.float64)
        if not np.isfinite(factors).all() or ((factors < 0) | (factors > 1)).any():
            raise ValueError("factor probabilities must be finite values in [0,1]")
        if not np.isfinite(classes).all() or (classes < 0).any():
            raise ValueError("class probabilities must be finite and non-negative")
        if not math.isclose(float(classes.sum()), 1.0, abs_tol=1e-5):
            raise ValueError("class probabilities must sum to one")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.evidence_quality.cross_view_conflict and self.projected_cause is not CoarseCause.UNKNOWN:
            raise ValueError("reliable-view conflict must project to unknown")
        if self.factor_probs[ConsistencyFactor.CAUSE_RESOLVED.value] < 0.5 and self.projected_cause is not CoarseCause.UNKNOWN:
            raise ValueError("unresolved attribution must project to unknown")
        if self.factor_states and set(self.factor_states) != expected_factors:
            raise ValueError(f"factor_states must contain {sorted(expected_factors)}")
        if self.factor_confidences and set(self.factor_confidences) != expected_factors:
            raise ValueError(f"factor_confidences must contain {sorted(expected_factors)}")
        if self.factor_confidences:
            confidences = np.asarray(list(self.factor_confidences.values()), dtype=np.float64)
            if not np.isfinite(confidences).all() or ((confidences < 0) | (confidences > 1)).any():
                raise ValueError("factor confidences must be finite values in [0,1]")

    def factor(self, factor: ConsistencyFactor) -> float:
        return float(self.factor_probs[factor.value])

    def factor_state(self, factor: ConsistencyFactor) -> TriValue:
        if self.factor_states:
            return self.factor_states[factor.value]
        return TriValue.TRUE if self.factor(factor) >= 0.5 else TriValue.FALSE

    def factor_confidence(self, factor: ConsistencyFactor) -> float:
        if self.factor_confidences:
            return float(self.factor_confidences[factor.value])
        return self.confidence


@dataclass
class BeliefFact:
    value: TriValue = TriValue.UNKNOWN
    confidence: float = 0.0
    source: str = "observation"
    updated_at_block: int = -1
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("belief confidence must be in [0,1]")
        if self.source not in {"observation", "attribution", "candidate_prediction"}:
            raise ValueError(f"unsupported belief source: {self.source}")


@dataclass(frozen=True)
class BeliefChange:
    predicate: str
    old_value: TriValue
    old_confidence: float
    new_value: TriValue
    new_confidence: float
    reason: str
    direct: bool
    propagation_path: tuple[str, ...]
    evidence_id: str


@dataclass
class BeliefState:
    task_id: int
    target_object: str
    receptacle: str
    task_stage: Stage = Stage.OBSERVE
    facts: dict[str, BeliefFact] = field(default_factory=dict)
    history: list[BeliefChange] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in PREDICATES:
            self.facts.setdefault(name, BeliefFact())

    def snapshot(self) -> dict:
        return {
            "task_id": self.task_id,
            "target_object": self.target_object,
            "receptacle": self.receptacle,
            "task_stage": self.task_stage.value,
            "facts": {name: asdict(fact) | {"value": fact.value.value} for name, fact in self.facts.items()},
            "history": [asdict(change) | {"old_value": change.old_value.value, "new_value": change.new_value.value}
                        for change in self.history],
        }


@dataclass(frozen=True)
class TaskBinding:
    task_id: int
    language: str
    target_object: str
    receptacle: str


@dataclass(frozen=True)
class CandidateEffect:
    candidate_id: int
    stage: Stage
    required_facts: Mapping[str, float]
    proposed_effects: Mapping[str, tuple[TriValue, float]]
    confidence: float
    evidence: Mapping[str, float]


@dataclass(frozen=True)
class ScoreWeights:
    attribution_compatibility: float
    progress: float
    dependency_risk: float
    uncertainty: float
    calibrated: bool = False
    requery_cost: float = 0.0


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: int
    accepted: bool
    official_value: float
    total_score: float | None
    rejection_reasons: tuple[str, ...]
    components: Mapping[str, float]


@dataclass(frozen=True)
class ActionRefinementRequest:
    """Future external-refiner input; no refiner model is claimed yet."""
    candidate_actions: np.ndarray
    belief_snapshot: Mapping[str, object]
    attribution: AttributionOutput

    def __post_init__(self) -> None:
        actions = np.asarray(self.candidate_actions)
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError("candidate_actions must be finite with shape (16,7)")


@dataclass(frozen=True)
class ActionRefinementResult:
    residual: np.ndarray
    confidence: float
    max_abs_residual: float

    def __post_init__(self) -> None:
        residual = np.asarray(self.residual)
        if residual.shape != (16, 7) or not np.isfinite(residual).all():
            raise ValueError("residual must be finite with shape (16,7)")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if float(np.max(np.abs(residual))) > self.max_abs_residual + 1e-8:
            raise ValueError("action residual exceeds declared safety bound")
