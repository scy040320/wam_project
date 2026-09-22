from cfwam.graph import BeliefGraph, TaskGraphSpec
from cfwam.recovery import RecoveryRouter
from cfwam.types import AttributionResult, Cause, RecoveryAction
from cfwam.protocol import DevelopmentSplit, expected_short_trajectory_count
from cfwam.runtime import (
    AbstainPolicy,
    CausalConsistencyGuard,
    GuardLevel,
    TemporalAbstainGate,
)
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


def guard_result(unknown_probability: float, abstained: bool) -> AttributionResult:
    probabilities = {
        Cause.NORMAL: 1.0 - unknown_probability,
        Cause.VISUAL_OCCLUSION: 0.0,
        Cause.OBJECT_SHIFT: 0.0,
        Cause.ACTION_NOISE: 0.0,
        Cause.UNKNOWN: unknown_probability,
    }
    return AttributionResult(
        Cause.UNKNOWN if abstained else Cause.NORMAL,
        probabilities,
        frozenset(),
        0.0,
        0.0,
        abstained,
    )


def test_temporal_gate_strong_unknown_requires_confirmation():
    gate = TemporalAbstainGate(strong_unknown_probability=0.995)
    first = gate.update(guard_result(0.999, True))
    second = gate.update(guard_result(0.999, True))
    assert first.level is GuardLevel.STRONG
    assert first.guarded_reobserve
    assert not first.force_safe_stop
    assert first.weak_streak == 1
    assert second.level is GuardLevel.STRONG
    assert second.force_safe_stop
    assert not second.guarded_reobserve
    assert second.weak_streak == 2


def test_temporal_gate_causal_unknown_stops_immediately():
    gate = TemporalAbstainGate(strong_unknown_probability=0.995)
    outcome = gate.update(guard_result(0.8, True), immediate_safe_stop=True)
    assert outcome.level is GuardLevel.STRONG
    assert outcome.force_safe_stop
    assert not outcome.guarded_reobserve


def test_temporal_gate_first_weak_reobserves_then_confirms():
    gate = TemporalAbstainGate(strong_unknown_probability=0.995)
    first = gate.update(guard_result(0.8, True))
    second = gate.update(guard_result(0.7, True))
    assert first.level is GuardLevel.WEAK
    assert first.guarded_reobserve
    assert first.weak_streak == 1
    assert second.force_safe_stop
    assert second.weak_streak == 2


def test_temporal_gate_clear_evidence_resets_streak_and_episode():
    gate = TemporalAbstainGate(strong_unknown_probability=0.995)
    gate.update(guard_result(0.8, True))
    clear = gate.update(guard_result(0.1, False))
    assert clear.level is GuardLevel.CLEAR
    assert clear.weak_streak == 0
    gate.update(guard_result(0.8, True))
    gate.reset_episode()
    assert gate.weak_streak == 0


def test_guarded_reobserve_refreshes_only_camera_evidence():
    graph = BeliefGraph(spec())
    decision = RecoveryRouter(graph).guarded_reobserve(8)
    assert decision.action is RecoveryAction.GUARDED_REOBSERVE
    assert decision.refresh_wam
    assert not decision.preserve_next_segment
    assert decision.invalidated_nodes == {"primary", "wrist"}


def test_causal_guard_uses_observed_action_delta_for_action_noise():
    guard = CausalConsistencyGuard(0.3, 0.2)
    outcome = guard.apply(result(Cause.NORMAL), 0.7, 0.0, {"object", "target"})
    assert outcome.attribution.cause is Cause.ACTION_NOISE
    assert outcome.reason == "observed_action_execution_delta"
    assert "action_segment_4" in outcome.attribution.affected_nodes


def test_causal_guard_treats_simultaneous_visual_and_action_evidence_as_unknown():
    guard = CausalConsistencyGuard(0.3, 0.2)
    outcome = guard.apply(result(Cause.VISUAL_OCCLUSION, {"camera_primary"}), 0.7, 0.0, {"object"})
    assert outcome.attribution.cause is Cause.UNKNOWN
    assert outcome.attribution.abstained


def test_causal_guard_rescues_temporal_object_shift_without_action_delta():
    guard = CausalConsistencyGuard(0.3, 0.2)
    outcome = guard.apply(result(Cause.NORMAL), 0.0, 0.25, {"object", "target"})
    assert outcome.attribution.cause is Cause.OBJECT_SHIFT
    assert outcome.attribution.affected_nodes == {"object", "target"}


def test_causal_guard_model_unknown_with_action_evidence_is_explicit():
    guard = CausalConsistencyGuard(0.3, 0.2)
    original = result(Cause.UNKNOWN, abstained=True)
    outcome = guard.apply(original, 0.7, 0.3, {"object"})
    assert outcome.attribution.cause is Cause.UNKNOWN
    assert outcome.reason == "model_unknown_with_execution_evidence"
    assert "action_segment_4" in outcome.attribution.affected_nodes


def test_causal_guard_preserves_unexplained_model_unknown():
    guard = CausalConsistencyGuard(0.3, 0.2)
    original = result(Cause.UNKNOWN, abstained=True)
    outcome = guard.apply(original, 0.0, 0.3, {"object"})
    assert outcome.attribution is original
    assert outcome.reason == "model_unknown"
