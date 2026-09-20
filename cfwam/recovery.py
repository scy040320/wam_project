"""Transparent recovery routing for attribution outputs."""

from __future__ import annotations

from dataclasses import dataclass

from .graph import BeliefGraph
from .types import AttributionResult, Cause, NodeKind, RecoveryAction


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    invalidated_nodes: frozenset[str]
    refresh_wam: bool
    preserve_next_segment: bool
    note: str


class RecoveryRouter:
    """Maps causes to auditable recovery decisions, never physical rollback."""

    def __init__(self, graph: BeliefGraph):
        self.graph = graph

    def route(self, attribution: AttributionResult, timestamp: int) -> RecoveryDecision:
        if attribution.abstained or attribution.cause is Cause.UNKNOWN:
            invalidated = frozenset(self.graph.invalidate(self.graph.beliefs, "unknown", timestamp))
            return RecoveryDecision(
                RecoveryAction.SAFE_STOP_GLOBAL_REFRESH, invalidated, True, False,
                "unknown/abstain: pause control and refresh globally",
            )
        if attribution.cause is Cause.NORMAL:
            return RecoveryDecision(RecoveryAction.CONTINUE, frozenset(), False, True, "no attributable mismatch")
        if attribution.cause is Cause.VISUAL_OCCLUSION:
            cameras = [node.node_id for node in self.graph.spec.nodes if node.kind is NodeKind.OBSERVATION]
            self.graph.refresh(cameras, timestamp, "reobserve")
            return RecoveryDecision(
                RecoveryAction.REOBSERVE, frozenset(cameras), False, True,
                "refresh visual evidence before continuing",
            )
        if attribution.cause is Cause.OBJECT_SHIFT:
            invalidated = frozenset(self.graph.invalidate(attribution.affected_nodes, "object_shift", timestamp))
            return RecoveryDecision(
                RecoveryAction.LOCAL_STATE_UPDATE, invalidated, True, False,
                "invalidate affected belief descendants and refresh next action segment",
            )
        if attribution.cause is Cause.ACTION_NOISE:
            action_nodes = [node.node_id for node in self.graph.spec.nodes if node.kind is NodeKind.ACTION_SEGMENT]
            self.graph.invalidate(action_nodes, "action_noise", timestamp)
            return RecoveryDecision(
                RecoveryAction.LOCAL_ACTION_CORRECTION, frozenset(action_nodes), False, False,
                "apply execution-feedback correction to the next action segment",
            )
        raise AssertionError(f"Unhandled cause: {attribution.cause}")
