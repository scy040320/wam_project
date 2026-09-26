"""Object-centric belief updates and dependency-aware invalidation."""

from __future__ import annotations

from typing import Iterable, Mapping

from .contracts import (
    AttributionOutput, BeliefChange, BeliefFact, BeliefState, CoarseCause,
    ConsistencyFactor, PREDICATES, Stage, TaskBinding, TriValue,
)

DEFAULT_TASK_BINDINGS: dict[int, TaskBinding] = {
    0: TaskBinding(0, "put alphabet soup and tomato sauce in basket", "alphabet_soup_1", "basket"),
    1: TaskBinding(1, "put cream cheese box and butter in basket", "butter_1", "basket"),
    2: TaskBinding(2, "turn on stove and put moka pot on it", "moka_pot_1", "stove"),
    3: TaskBinding(3, "put black bowl in bottom drawer and close it", "akita_black_bowl_1", "bottom_drawer"),
}


class DependencyGraph:
    def __init__(self, edges: Iterable[tuple[str, str]]) -> None:
        self.edges = tuple(edges)
        self.nodes = set(PREDICATES)
        self.children = {name: set() for name in self.nodes}
        for parent, child in self.edges:
            if parent not in self.nodes or child not in self.nodes:
                raise ValueError(f"unknown graph predicate: {(parent, child)}")
            self.children[parent].add(child)
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        temporary: set[str] = set()
        permanent: set[str] = set()
        def visit(node: str) -> None:
            if node in permanent:
                return
            if node in temporary:
                raise ValueError("dependency graph must be acyclic")
            temporary.add(node)
            for child in self.children[node]:
                visit(child)
            temporary.remove(node)
            permanent.add(node)
        for node in sorted(self.nodes):
            visit(node)

    def descendants_with_paths(self, root: str) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        queue: list[tuple[str, tuple[str, ...]]] = [(root, (root,))]
        while queue:
            node, path = queue.pop(0)
            for child in sorted(self.children[node]):
                if child not in result:
                    result[child] = path + (child,)
                    queue.append((child, path + (child,)))
        return result


DEFAULT_GRAPH = DependencyGraph((
    ("target_visible", "target_pose_current"), ("target_pose_current", "target_reachable"),
    ("target_reachable", "grasped"), ("execution_consistent", "grasped"),
    ("grasped", "lifted"), ("lifted", "place_ready"),
    ("receptacle_visible", "place_ready"), ("place_ready", "placed"),
))


def initial_belief(task_id: int, *, block_index: int = 0,
                   bindings: Mapping[int, TaskBinding] = DEFAULT_TASK_BINDINGS) -> BeliefState:
    binding = bindings[task_id]
    belief = BeliefState(task_id, binding.target_object, binding.receptacle)
    for name in ("target_visible", "target_pose_current", "target_reachable", "receptacle_visible", "execution_consistent"):
        belief.facts[name] = BeliefFact(TriValue.TRUE, 0.7, "observation", block_index, ("initial_observation",))
    return belief


def _set_fact(belief: BeliefState, name: str, value: TriValue, confidence: float, reason: str,
              block_index: int, evidence_id: str, *, direct: bool, path: tuple[str, ...]) -> None:
    old = belief.facts[name]
    confidence = min(1.0, max(0.0, float(confidence)))
    belief.facts[name] = BeliefFact(value, confidence, "attribution", block_index, (evidence_id,))
    belief.history.append(BeliefChange(name, old.value, old.confidence, value, confidence, reason, direct, path, evidence_id))


def _invalidate(belief: BeliefState, graph: DependencyGraph, root: str, confidence: float,
                reason: str, block_index: int, evidence_id: str) -> None:
    _set_fact(belief, root, TriValue.FALSE, confidence, reason, block_index, evidence_id, direct=True, path=(root,))
    for child, path in graph.descendants_with_paths(root).items():
        propagated = max(0.05, confidence * (0.85 ** (len(path) - 1)))
        _set_fact(belief, child, TriValue.UNKNOWN, propagated, reason, block_index, evidence_id, direct=False, path=path)


def _mark_uncertain(belief: BeliefState, graph: DependencyGraph, root: str, confidence: float,
                    reason: str, block_index: int, evidence_id: str) -> None:
    _set_fact(belief, root, TriValue.UNKNOWN, confidence, reason, block_index, evidence_id,
              direct=True, path=(root,))
    for child, path in graph.descendants_with_paths(root).items():
        propagated = max(0.05, confidence * (0.85 ** (len(path) - 1)))
        _set_fact(belief, child, TriValue.UNKNOWN, propagated, reason, block_index,
                  evidence_id, direct=False, path=path)


def update_belief(belief: BeliefState, attribution: AttributionOutput, block_index: int,
                  graph: DependencyGraph = DEFAULT_GRAPH, inconsistency_threshold: float = 0.5) -> BeliefState:
    """Apply every supported inconsistency factor; multiple causes may coexist."""
    evidence_id = attribution.source_block_id
    confidence = attribution.confidence
    reasons: list[str] = []
    observation = attribution.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE)
    world = attribution.factor_state(ConsistencyFactor.WORLD_STATE_CONSISTENT)
    execution = attribution.factor_state(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT)
    stage = attribution.factor_state(ConsistencyFactor.TASK_STAGE_CONSISTENT)
    resolved = attribution.factor_state(ConsistencyFactor.CAUSE_RESOLVED)
    if attribution.factor_confidence(ConsistencyFactor.OBSERVATION_RELIABLE) < inconsistency_threshold:
        observation = TriValue.UNKNOWN
    if attribution.factor_confidence(ConsistencyFactor.WORLD_STATE_CONSISTENT) < inconsistency_threshold:
        world = TriValue.UNKNOWN
    if attribution.factor_confidence(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT) < inconsistency_threshold:
        execution = TriValue.UNKNOWN
    if attribution.factor_confidence(ConsistencyFactor.TASK_STAGE_CONSISTENT) < inconsistency_threshold:
        stage = TriValue.UNKNOWN
    if attribution.factor_confidence(ConsistencyFactor.CAUSE_RESOLVED) < inconsistency_threshold:
        resolved = TriValue.UNKNOWN
    if observation is not TriValue.TRUE:
        reasons.append(ConsistencyFactor.OBSERVATION_RELIABLE.value)
        _set_fact(belief, "target_visible", TriValue.UNKNOWN, confidence, reasons[-1], block_index, evidence_id,
                  direct=True, path=("target_visible",))
        _set_fact(belief, "target_pose_current", TriValue.UNKNOWN, confidence, reasons[-1], block_index, evidence_id,
                  direct=True, path=("target_pose_current",))
    if world is TriValue.FALSE:
        reasons.append(ConsistencyFactor.WORLD_STATE_CONSISTENT.value)
        _invalidate(belief, graph, "target_pose_current", confidence, reasons[-1], block_index, evidence_id)
    elif world is TriValue.UNKNOWN:
        reasons.append(ConsistencyFactor.WORLD_STATE_CONSISTENT.value)
        _mark_uncertain(belief, graph, "target_pose_current", confidence, reasons[-1], block_index, evidence_id)
    if execution is TriValue.FALSE:
        reasons.append(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value)
        _invalidate(belief, graph, "execution_consistent", confidence, reasons[-1], block_index, evidence_id)
    elif execution is TriValue.UNKNOWN:
        reasons.append(ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value)
        _mark_uncertain(belief, graph, "execution_consistent", confidence, reasons[-1], block_index, evidence_id)
    if stage is not TriValue.TRUE:
        reasons.append(ConsistencyFactor.TASK_STAGE_CONSISTENT.value)
        belief.task_stage = Stage.UNCERTAIN
    unresolved = resolved is not TriValue.TRUE
    if unresolved or attribution.projected_cause is CoarseCause.UNKNOWN:
        risky = {
            Stage.GRASP: ("target_pose_current", "execution_consistent"),
            Stage.LIFT: ("grasped", "execution_consistent"),
            Stage.TRANSPORT: ("grasped", "lifted"),
            Stage.PLACE: ("lifted", "place_ready"),
        }.get(belief.task_stage, ("target_pose_current", "execution_consistent"))
        for name in risky:
            _set_fact(belief, name, TriValue.UNKNOWN, confidence, ConsistencyFactor.CAUSE_RESOLVED.value,
                      block_index, evidence_id, direct=True, path=(name,))
    if not reasons and not unresolved and attribution.projected_cause is CoarseCause.NORMAL:
        for name, fact in list(belief.facts.items()):
            if fact.value is not TriValue.UNKNOWN:
                _set_fact(belief, name, fact.value, min(1.0, fact.confidence + 0.05 * confidence),
                          "consistent_block", block_index, evidence_id, direct=True, path=(name,))
    return belief
