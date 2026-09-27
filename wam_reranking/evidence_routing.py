"""D18-v5 auditable evidence routing.

This module deliberately separates learned mismatch factors from evidence
availability checks.  In particular, action-record reliability is read from
the acquisition contract and is never inferred by a learned head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .contracts import CoarseCause


LEARNED_FACTOR_NAMES = (
    "visual_evidence_corrupted",
    "object_or_environment_state_changed",
    "execution_or_contact_deviated",
    "cross_view_conflict",
)


class EvidenceRoute(str, Enum):
    RESOLVED = "resolved"
    ACTION_RECORD_UNRELIABLE = "action_record_unreliable"
    CAUSE_UNRESOLVED = "cause_unresolved"


@dataclass(frozen=True)
class EvidenceRoutingDecision:
    route: EvidenceRoute
    projected_cause: CoarseCause
    hard_rule_applied: bool
    reasons: tuple[str, ...]


def route_evidence(
    *,
    action_record_available: bool,
    projected_cause: CoarseCause,
    learned_factor_probs: Mapping[str, float],
    factor_threshold: float = 0.5,
) -> EvidenceRoutingDecision:
    """Apply the frozen D18-v5 routing contract.

    ``cause_unresolved`` is an abstention state.  It must not be interpreted as
    identifying a hidden low-magnitude physical intervention.
    """
    if set(learned_factor_probs) != set(LEARNED_FACTOR_NAMES):
        raise ValueError(f"learned factors must be {sorted(LEARNED_FACTOR_NAMES)}")
    if not 0.0 <= factor_threshold <= 1.0:
        raise ValueError("factor_threshold must be in [0,1]")
    if any(not 0.0 <= float(value) <= 1.0 for value in learned_factor_probs.values()):
        raise ValueError("learned factor probabilities must be in [0,1]")

    if not action_record_available:
        return EvidenceRoutingDecision(
            EvidenceRoute.ACTION_RECORD_UNRELIABLE,
            CoarseCause.UNKNOWN,
            True,
            ("action record unavailable by acquisition contract",),
        )

    active = tuple(
        name for name in LEARNED_FACTOR_NAMES
        if float(learned_factor_probs[name]) >= factor_threshold
    )
    if projected_cause is CoarseCause.UNKNOWN and not active:
        return EvidenceRoutingDecision(
            EvidenceRoute.CAUSE_UNRESOLVED,
            CoarseCause.UNKNOWN,
            False,
            ("unknown coarse cause with no supported learned factor",),
        )

    return EvidenceRoutingDecision(
        EvidenceRoute.RESOLVED,
        projected_cause,
        False,
        tuple(f"supported factor: {name}" for name in active),
    )
