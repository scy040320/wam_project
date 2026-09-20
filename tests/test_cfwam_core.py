from cfwam.graph import BeliefGraph, TaskGraphSpec
from cfwam.recovery import RecoveryRouter
from cfwam.types import AttributionResult, Cause
from cfwam.protocol import DevelopmentSplit, expected_short_trajectory_count
from cfwam.runtime import AbstainPolicy
from pathlib import Path


def spec():
    return TaskGraphSpec.from_mapping({
        "task_id": "test", "language": "test",
        "phase_steps": {"phase_approach": 0, "phase_grasp": 4, "phase_transport": 8},
        "nodes": [
            {"id": "phase_approach", "kind": "task_phase"},
            {"id": "phase_grasp", "kind": "task_phase"},
            {"id": "phase_transport", "kind": "task_phase"},
            {"id": "primary", "kind": "observation"},
            {"id": "wrist", "kind": "observation"},
            {"id": "gripper", "kind": "effector_gripper"},
            {"id": "object", "kind": "object"},
            {"id": "target", "kind": "target_region"},
            {"id": "action", "kind": "action_segment"},
        ],
        "edges": [
            {"source": "primary", "target": "object", "kind": "observation_supports_state"},
            {"source": "action", "target": "gripper", "kind": "action_acts_on_effector"},
            {"source": "gripper", "target": "object", "kind": "effector_interacts_object"},
            {"source": "object", "target": "target", "kind": "object_serves_target"},
            {"source": "phase_approach", "target": "phase_grasp", "kind": "phase_depends_on_predecessor"},
            {"source": "phase_grasp", "target": "phase_transport", "kind": "phase_depends_on_predecessor"},
        ],
    })


def result(cause, nodes=(), abstained=False):
    return AttributionResult(cause, {cause: 1.0}, frozenset(nodes), 0.0, 0.0, abstained)


def test_object_shift_invalidates_only_dependency_closure():
    graph = BeliefGraph(spec())
    invalidated = graph.invalidate(["object"], "object_shift", 4)
    assert invalidated == {"object", "target"}
    assert graph.beliefs["primary"].valid
    assert graph.beliefs["gripper"].valid


def test_visual_occlusion_reobserves_without_global_refresh():
    graph = BeliefGraph(spec())
    decision = RecoveryRouter(graph).route(result(Cause.VISUAL_OCCLUSION), 4)
    assert decision.refresh_wam is False
    assert decision.preserve_next_segment is True
    assert decision.invalidated_nodes == {"primary", "wrist"}


def test_unknown_is_global_safe_stop():
    graph = BeliefGraph(spec())
    decision = RecoveryRouter(graph).route(result(Cause.UNKNOWN, abstained=True), 4)
    assert decision.refresh_wam is True
    assert decision.preserve_next_segment is False
    assert len(decision.invalidated_nodes) == len(graph.beliefs)


def test_development_split_is_grouped_and_preregistered():
    split = DevelopmentSplit()
    assert split.subset(0) == "train"
    assert split.subset(24) == "validation"
    assert split.subset(39) == "test"
    assert expected_short_trajectory_count() == 2400


def test_low_confidence_becomes_unknown():
    policy = AbstainPolicy(min_known_probability=0.7, max_entropy=1.8, max_residual_energy=5.0)
    result = policy.apply({
        Cause.NORMAL: 0.3, Cause.VISUAL_OCCLUSION: 0.3, Cause.OBJECT_SHIFT: 0.2,
        Cause.ACTION_NOISE: 0.1, Cause.UNKNOWN: 0.1,
    }, Cause.NORMAL, 1.0)
    assert result.abstained
    assert result.cause is Cause.UNKNOWN


def test_all_reviewed_development_graphs_are_valid():
    root = Path(__file__).parents[1] / "configs"
    for task_id in range(4):
        TaskGraphSpec.from_yaml(root / f"libero_task_{task_id}.yaml")
