"""Task graph specification and online belief/provenance graph.

No simulator state belongs in this module.  A graph node stores only online
observations, validity and provenance.  Simulator truth can supervise masks
offline in the data-generation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .types import EdgeKind, EdgeSpec, NodeBelief, NodeKind, NodeSpec


@dataclass(frozen=True)
class TaskGraphSpec:
    task_id: str
    language: str
    nodes: tuple[NodeSpec, ...]
    edges: tuple[EdgeSpec, ...]
    phase_steps: dict[str, int]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TaskGraphSpec":
        nodes = tuple(
            NodeSpec(item["id"], NodeKind(item["kind"]), dict(item.get("metadata", {})))
            for item in payload["nodes"]
        )
        edges = tuple(
            EdgeSpec(item["source"], item["target"], EdgeKind(item["kind"]))
            for item in payload["edges"]
        )
        spec = cls(
            task_id=str(payload["task_id"]),
            language=str(payload["language"]),
            nodes=nodes,
            edges=edges,
            phase_steps={str(key): int(value) for key, value in payload["phase_steps"].items()},
        )
        spec.validate()
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskGraphSpec":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - actionable runtime error
            raise RuntimeError("PyYAML is required to load TaskGraphSpec YAML files.") from exc
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))

    def validate(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Task graph node ids must be unique.")
        required = {
            NodeKind.TASK_PHASE, NodeKind.OBSERVATION, NodeKind.EFFECTOR,
            NodeKind.OBJECT, NodeKind.TARGET, NodeKind.ACTION_SEGMENT,
        }
        actual = {node.kind for node in self.nodes}
        missing = required - actual
        if missing:
            raise ValueError(f"Task graph misses node kinds: {sorted(item.value for item in missing)}")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"Unknown graph endpoint: {edge}")
        present_edges = {edge.kind for edge in self.edges}
        required_edges = set(EdgeKind)
        if required_edges - present_edges:
            raise ValueError(f"Task graph misses edge kinds: {sorted(item.value for item in required_edges - present_edges)}")
        phases = {node.node_id for node in self.nodes if node.kind is NodeKind.TASK_PHASE}
        if set(self.phase_steps) != phases:
            raise ValueError("phase_steps must name exactly the task_phase nodes.")


class BeliefGraph:
    """Mutable online graph with dependency-aware invalidation.

    Edges point from a prerequisite to a dependent state.  Invalidating a node
    invalidates its reachable descendants, but never changes simulator physics.
    """

    def __init__(self, spec: TaskGraphSpec):
        self.spec = spec
        self.beliefs = {node.node_id: NodeBelief(node.node_id) for node in spec.nodes}
        self._children: dict[str, set[str]] = {node.node_id: set() for node in spec.nodes}
        for edge in spec.edges:
            self._children[edge.source].add(edge.target)

    def descendants(self, node_ids: Iterable[str]) -> set[str]:
        pending = list(node_ids)
        closure = set(pending)
        while pending:
            source = pending.pop()
            for target in self._children[source]:
                if target not in closure:
                    closure.add(target)
                    pending.append(target)
        return closure

    def invalidate(self, node_ids: Iterable[str], reason: str, timestamp: int) -> set[str]:
        closure = self.descendants(node_ids)
        for node_id in closure:
            belief = self.beliefs[node_id]
            belief.valid = False
            belief.observed_at = timestamp
            belief.provenance = f"invalidated:{reason}"
        return closure

    def refresh(self, node_ids: Iterable[str], timestamp: int, provenance: str) -> None:
        for node_id in node_ids:
            belief = self.beliefs[node_id]
            belief.valid = True
            belief.observed_at = timestamp
            belief.provenance = provenance

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {
            node_id: {
                "valid": belief.valid,
                "observed_at": belief.observed_at,
                "provenance": belief.provenance,
            }
            for node_id, belief in self.beliefs.items()
        }
