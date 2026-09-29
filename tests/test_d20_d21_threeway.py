from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_d20_d21_threeway.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_d20_d21_threeway.py"
SPEC = spec_from_file_location("d20_threeway", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def record(predicted="object_shift", unknown=0.2, shift=0.9, conflict=0.1):
    return {
        "predicted": predicted,
        "confidence": 0.8,
        "class_probs": {
            "normal": 0.05,
            "visual_occlusion": 0.05,
            "object_shift": 0.8,
            "execution_contact_deviation": 0.05,
            "unknown": 0.05,
        },
        "factor_prob": {
            "visual_evidence_corrupted": 0.1,
            "object_or_environment_state_changed": shift,
            "execution_or_contact_deviated": 0.1,
            "cross_view_conflict": conflict,
        },
        "unknown_prob": unknown,
        "unknown_threshold": 0.64,
        "primary_available": True,
        "wrist_available": True,
        "action_record_available": True,
    }


def test_object_shift_head_invalidates_world_consistency():
    output = MODULE.learned_attribution(record(), "x")
    assert output.projected_cause.value == "object_shift"
    assert output.factor_states["world_state_consistent"].value == "false"
    assert output.factor_states["observation_reliable"].value == "true"


def test_unknown_threshold_is_preserved():
    output = MODULE.learned_attribution(record(predicted="normal", unknown=0.7), "x")
    assert output.projected_cause.value == "unknown"
    assert output.factor_states["cause_resolved"].value == "false"


def test_cross_view_conflict_forces_unknown():
    output = MODULE.learned_attribution(record(conflict=0.9), "x")
    assert output.projected_cause.value == "unknown"
    assert output.evidence_quality.cross_view_conflict


def test_outcome_oracle_prioritizes_success_then_cost():
    candidates = [
        {"candidate_id": 0, "outcome": {"success": False, "executed_steps": 10, "continuation_wam_calls": 1}},
        {"candidate_id": 1, "outcome": {"success": True, "executed_steps": 30, "continuation_wam_calls": 3}},
        {"candidate_id": 2, "outcome": {"success": True, "executed_steps": 20, "continuation_wam_calls": 4}},
        {"candidate_id": 3, "outcome": {"success": True, "executed_steps": 20, "continuation_wam_calls": 2}},
    ]
    assert max(candidates, key=MODULE.outcome_key)["candidate_id"] == 3


def test_fallback_without_rollout_is_not_counted_as_observed_harm():
    scenarios = [{
        "value_only": {"outcome": {"success": True, "executed_steps": 20, "continuation_wam_calls": 2}},
        "learned": {"outcome": None},
    }]
    result = MODULE.paired_delta(scenarios, "learned")
    assert result["strict_harms"] == 0
    assert result["unobserved_fallback_outcomes"] == 1
    assert result["baseline_success_replaced_by_fallback"] == 1
