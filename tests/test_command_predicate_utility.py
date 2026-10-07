import importlib.util
from pathlib import Path
import sys
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

name = "wam_reranking.command_predicate_utility"
module_path = Path(__file__).with_name("command_predicate_utility.py")
if not module_path.exists():
    module_path = Path(__file__).resolve().parents[1] / "wam_reranking" / "command_predicate_utility.py"
spec = importlib.util.spec_from_file_location(name, module_path)
api = importlib.util.module_from_spec(spec); sys.modules[name] = api; spec.loader.exec_module(api)
from wam_reranking import initial_belief
from wam_reranking.contracts import (AttributionOutput, BeliefFact, CandidateDecision, CandidateEffect,
    CoarseCause, ConsistencyFactor, EvidenceQuality, Stage, TriValue)
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.evidence_residual import EvidenceResidualModel
from wam_reranking.evidence_arbitration import select_evidence_arbitration


def attr(visual=True, execution=True, conflict=False, shift=False):
    p = {f.value: 1. for f in ConsistencyFactor}
    p["observation_reliable"] = float(visual)
    p["world_state_consistent"] = float(not shift)
    cause = CoarseCause.UNKNOWN if conflict else CoarseCause.OBJECT_SHIFT if shift else CoarseCause.NORMAL
    return AttributionOutput(p, {c.value: float(c is cause) for c in CoarseCause}, cause, .9, 0.,
        EvidenceQuality(visual, visual, execution, conflict), "previous_block",
        {f.value: TriValue.TRUE if p[f.value] else TriValue.FALSE for f in ConsistencyFactor})


def command(**kw):
    a = np.zeros((16, 7)); b = a.copy(); b[4:8, 0] = .12
    return api.command_evidence(**dict(requested=a, applied=b, record_available=True,
        source_block_id="previous_block", source_block_index=2, use_block_index=3,
        policy_step_t=32, endpoint_policy_step=48, observation_policy_step=48,
        executed_steps=16, alignment_valid=True, evidence_ids=("requested_sha", "applied_sha"), **kw))


def effect(contact=.65, agreement=.45):
    return CandidateEffect(0, Stage.LIFT, {"grasped": .6, "execution_consistent": .6},
        {"lifted": (TriValue.TRUE, .9)}, .9,
        {"cross_view_agreement": agreement, "relation_confidence": .9,
         "current_contact_reliability": contact, "contact_confidence": .9,
         "grasp_support_after": .9, "visual_support": .9, "target_displacement": .3})


def feature(**kw):
    plan = np.zeros((16, 7)); plan[:, 2] = .1; plan[:, 6] = 1.
    args = dict(prior=initial_belief(0), attribution=attr(), effect=effect(), actions=plan, command=command())
    args.update(kw); return api.conditioned_features(**args)


def test_measurement_not_changed_model_factor():
    a = attr(); original = deepcopy(a)
    x, t = feature(attribution=a)
    assert a == original and a.factor_probs["execution_contact_consistent"] == 1.
    assert t["deficits"]["execution_consistent"]["state"] == "false"
    assert t["deficits"]["grasped"]["state"] == "unknown"
    assert np.any(x)


@pytest.mark.parametrize("field,value", [("record_available", False), ("executed_steps", 15),
    ("alignment_valid", False), ("observation_policy_step", 47), ("use_block_index", 4)])
def test_invalid_command_fails_closed(field, value):
    a = np.zeros((16, 7)); b = a.copy(); b[0, 0] = .12
    kw = dict(requested=a, applied=b, record_available=True, source_block_id="previous_block",
        source_block_index=2, use_block_index=3, policy_step_t=32, endpoint_policy_step=48,
        observation_policy_step=48, executed_steps=16, alignment_valid=True)
    kw[field] = value; c = api.command_evidence(**kw)
    assert not c.valid and not c.deviated
    assert not feature(command=c)[1]["deficits"]


@pytest.mark.parametrize("amount,expected", [(0., False), (1e-6, False), (1.0001e-6, True)])
def test_frozen_command_threshold(amount, expected):
    a = np.zeros((16, 7)); b = a.copy(); b[0, 0] = amount
    c = api.command_evidence(requested=a, applied=b, record_available=True, source_block_id="previous_block",
        source_block_index=2, use_block_index=3, policy_step_t=32, endpoint_policy_step=48,
        observation_policy_step=48, executed_steps=16, alignment_valid=True)
    assert c.deviated is expected


def test_nonfinite_and_short_commands_do_not_establish_consistency():
    for a in (np.zeros((15, 7)), np.full((16, 7), np.nan)):
        c = api.command_evidence(requested=a, applied=a, record_available=True, source_block_id="previous_block",
            source_block_index=2, use_block_index=3, policy_step_t=32, endpoint_policy_step=48,
            observation_policy_step=48, executed_steps=16, alignment_valid=True)
        assert not c.valid and c.max_abs is None


def test_fresh_observation_protected_future_prediction_not_certificate():
    b = initial_belief(0)
    b.facts["grasped"] = BeliefFact(TriValue.TRUE, .95, "observation", 3, ("actual_contact",))
    b.facts["lifted"] = BeliefFact(TriValue.TRUE, .99, "candidate_prediction", 3, ("forecast",))
    original = deepcopy(b.snapshot())
    fixed = api.effective_belief(prior=b, attribution=attr(), block_index=3, command=command())
    assert b.snapshot() == original
    assert fixed.facts["grasped"].value is TriValue.TRUE
    assert fixed.facts["lifted"].value is TriValue.UNKNOWN
    assert fixed.facts["execution_consistent"].value is TriValue.FALSE


def test_no_dag_has_only_measured_root():
    _, t = feature(no_dag=True)
    assert set(t["deficits"]) == {"execution_consistent"}


def test_witnessed_physical_false_not_erased_by_command_unknown():
    b = initial_belief(0)
    b.facts["grasped"] = BeliefFact(TriValue.FALSE, .95, "observation", 2, ("observed_no_grasp",))
    fixed = api.effective_belief(prior=b, attribution=attr(), block_index=3, command=command())
    assert fixed.facts["grasped"].value is TriValue.FALSE


def test_contact_not_silenced_by_relation_agreement():
    _, t = feature()
    assert t["predicate_usable"]["grasped"]
    assert not t["predicate_usable"]["place_ready"]
    assert t["predicate_source"]["grasped"] == "subject_gripper"


@pytest.mark.parametrize("visual,conflict", [(False, False), (True, True)])
def test_visual_source_failure_still_closes_contact(visual, conflict):
    _, t = feature(attribution=attr(visual=visual, conflict=conflict))
    assert not t["predicate_usable"]["grasped"]
    assert t["execution_usable"]


def test_no_command_or_unreliable_execution_does_not_fabricate_deficit():
    assert not feature(command=None)[1]["deficits"]
    assert not feature(attribution=attr(execution=False))[1]["deficits"]


def test_contact_threshold_not_lowered():
    assert not feature(effect=effect(contact=.549999))[1]["predicate_usable"]["grasped"]
    assert feature(effect=effect(contact=.55))[1]["predicate_usable"]["grasped"]


def fixture():
    backbone = CandidateUtilityModel(np.zeros(27), np.ones(27), np.zeros(27), 0., switch_margin=.01)
    residual = EvidenceResidualModel(np.ones(10), np.zeros(10), np.ones(10), 0., .1)
    ds = [CandidateDecision(i, True, .6 - .01 * i, None, (), {}) for i in range(2)]
    xs = {i: np.zeros(27) for i in range(2)}
    for i in range(2): xs[i][0] = ds[i].official_value; xs[i][16] = 1.; xs[i][5] = xs[i][8] = .9
    _, t = feature()
    return dict(decisions=ds, backbone_features=xs, cause_features={i: np.zeros(10) for i in range(2)},
        backbone=backbone, frozen_residual=residual, attribution=attr(), traces=[deepcopy(t) for _ in range(2)])


def test_zero_exact_v8_scores_choice():
    k = fixture(); old = {n: v for n, v in k.items() if n not in {"frozen_residual", "traces"}}
    original, scores = select_evidence_arbitration(**old, residual=k["frozen_residual"])
    fixed, new, _ = api.select_conditioned(**k, deltas=[0., 0.])
    assert fixed == original and new == scores


def test_hard_rejected_candidate_cannot_be_unlocked():
    k = fixture(); k["decisions"][1] = replace(k["decisions"][1], accepted=False, rejection_reasons=("physical_false",))
    chosen, _, _ = api.select_conditioned(**k, deltas=[-.04, .04])
    assert chosen.candidate_id == 0


def test_frozen_pairwise_cap_enforced():
    with pytest.raises(ValueError): api.select_conditioned(**fixture(), deltas=[-.06, .06])


def test_normal_does_not_invent_recovery_need():
    x, t = feature(command=None, effect=effect(agreement=.9))
    assert not np.any(x) and not t["deficits"]


def test_wrong_time_typed_evidence_rejected():
    with pytest.raises(ValueError):
        feature(command=command(), block_index=4)
