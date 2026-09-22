"""Stable, platform-neutral types for the v1 attribution interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Cause(str, Enum):
    NORMAL = "normal"
    VISUAL_OCCLUSION = "visual_occlusion"
    OBJECT_SHIFT = "object_shift"
    ACTION_NOISE = "action_noise"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    GUARDED_REOBSERVE = "guarded_reobserve"
    REOBSERVE = "reobserve"
    LOCAL_STATE_UPDATE = "local_state_update"
    LOCAL_ACTION_CORRECTION = "local_action_correction"
    SAFE_STOP_GLOBAL_REFRESH = "safe_stop_global_refresh"


class NodeKind(str, Enum):
    TASK_PHASE = "task_phase"
    OBSERVATION = "observation"
    EFFECTOR = "effector_gripper"
    OBJECT = "object"
    TARGET = "target_region"
    ACTION_SEGMENT = "action_segment"


class EdgeKind(str, Enum):
    OBSERVATION_SUPPORTS_STATE = "observation_supports_state"
    ACTION_ACTS_ON_EFFECTOR = "action_acts_on_effector"
    EFFECTOR_INTERACTS_OBJECT = "effector_interacts_object"
    OBJECT_SERVES_TARGET = "object_serves_target"
    PHASE_DEPENDS_ON_PREDECESSOR = "phase_depends_on_predecessor"


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    kind: NodeKind
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    kind: EdgeKind


@dataclass
class NodeBelief:
    node_id: str
    valid: bool = True
    observed_at: int | None = None
    provenance: str = "unobserved"
    feature: list[float] | None = None


@dataclass(frozen=True)
class AttributionResult:
    cause: Cause
    cause_probabilities: dict[Cause, float]
    affected_nodes: frozenset[str]
    residual_energy: float
    entropy: float
    abstained: bool
