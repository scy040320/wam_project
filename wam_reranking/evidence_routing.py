"""D18-v5 auditable evidence routing.

This module deliberately separates learned mismatch factors from evidence
availability checks.  In particular, action-record reliability is read from
the acquisition contract and is never inferred by a learned head.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

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
    COMMAND_EXECUTION_DEVIATION = "command_execution_deviation"
    CAUSE_UNRESOLVED = "cause_unresolved"


@dataclass(frozen=True)
class EvidenceRoutingDecision:
    route: EvidenceRoute
    projected_cause: CoarseCause
    hard_rule_applied: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CommandExecutionEvidence:
    """Auditable requested-versus-applied action evidence.

    This identifies command execution deviation only. It must not be used to
    claim abnormal contact dynamics when both commands agree.
    """

    deviated: bool
    max_abs: float
    l2: float
    threshold: float


def evaluate_command_execution(
    requested_actions: np.ndarray,
    applied_actions: np.ndarray,
    *,
    threshold: float = 1e-6,
) -> CommandExecutionEvidence:
    requested = np.asarray(requested_actions, dtype=np.float64)
    applied = np.asarray(applied_actions, dtype=np.float64)
    if requested.shape != (16, 7) or applied.shape != (16, 7):
        raise ValueError("requested_actions and applied_actions must have shape (16,7)")
    if not np.isfinite(requested).all() or not np.isfinite(applied).all():
        raise ValueError("requested_actions and applied_actions must be finite")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    residual = applied - requested
    max_abs = float(np.max(np.abs(residual)))
    return CommandExecutionEvidence(
        deviated=max_abs > threshold,
        max_abs=max_abs,
        l2=float(np.linalg.norm(residual)),
        threshold=float(threshold),
    )


def route_evidence(
    *,
    action_record_available: bool,
    projected_cause: CoarseCause,
    learned_factor_probs: Mapping[str, float],
    factor_threshold: float = 0.5,
    command_execution: CommandExecutionEvidence | None = None,
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

    if command_execution is not None and command_execution.deviated:
        return EvidenceRoutingDecision(
            EvidenceRoute.COMMAND_EXECUTION_DEVIATION,
            CoarseCause.EXECUTION_CONTACT_DEVIATION,
            True,
            (
                "applied action differs from requested action",
                f"max_abs={command_execution.max_abs:.8g} > threshold={command_execution.threshold:.8g}",
                "this hard rule does not assert a contact-dynamics cause",
            ),
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
