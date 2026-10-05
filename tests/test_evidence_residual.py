"""Source semantics and shared-backbone regression tests (no simulator GT)."""
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking import (
    AttributionOutput, BeliefFact, CandidateDecision, CoarseCause,
    ConsistencyFactor, EvidenceQuality, Stage, TriValue, initial_belief,
    parse_candidate_effect,
)
from wam_reranking.candidate_utility import CandidateUtilityModel, FEATURE_NAMES
from wam_reranking.evidence_residual import (
    CAUSE_FEATURE_NAMES, EvidenceResidualModel, candidate_backbone_features,
    cause_residual_features, fit_evidence_residual, prepare_evidence_decisions,
    select_evidence_residual, typed_gate_effect,
)


def attr(cause=CoarseCause.NORMAL, conflict=False):
    probs = {f.value: .95 for f in ConsistencyFactor}
    if cause is CoarseCause.OBJECT_SHIFT:
        probs[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = .05
    if cause is CoarseCause.UNKNOWN:
        probs[ConsistencyFactor.CAUSE_RESOLVED.value] = .05
    classes = {c.value: .025 for c in CoarseCause}
    classes[cause.value] = .9
    return AttributionOutput(probs, classes, cause, .9, .3,
                             EvidenceQuality(True, True, True, conflict), "block")


def effect(stage=Stage.PLACE):
    a = np.zeros((16, 7))
    if stage is Stage.PLACE:
        a[:8, 6] = -1; a[8:, 6] = 1
    else:
        a[:, 0] = .02
    return parse_candidate_effect(0, a)


def model():
    n = len(FEATURE_NAMES)
    return CandidateUtilityModel(np.zeros(n), np.ones(n), np.zeros(n), 0.)


def residual(gain=0., bound=0., weights=None):
    n = len(CAUSE_FEATURE_NAMES)
    return EvidenceResidualModel(np.ones(n), np.ones(n) if weights is None else weights,
                                 np.ones(n) * 10, gain, bound)


class EvidenceResidualTests(unittest.TestCase):
    def test_normal_preserves_raw_relation_and_contact_information(self):
        e = replace(effect(), evidence={"relation_score_after": .8,
                    "relation_score_delta": .2, "grasp_support_after": .7})
        x = candidate_backbone_features(e, .6)
        self.assertEqual(x[2], .8); self.assertEqual(x[3], .2); self.assertEqual(x[6], .7)

    def test_normal_has_no_cause_residual(self):
        d = CandidateDecision(0, True, .5, .5, (), {"dependency_risk": 1.})
        np.testing.assert_array_equal(cause_residual_features(effect(), attr(), d), np.zeros(10))

    def test_two_view_conflict_cannot_reward_world_recovery(self):
        e = replace(effect(), evidence={"relation_score_delta": .2,
            "relation_confidence": .9, "cross_view_agreement": .9})
        d = CandidateDecision(0, True, .5, .5, (), {})
        a = attr(CoarseCause.UNKNOWN, True)
        self.assertEqual(cause_residual_features(e, a, d)[0], 0.)

    def test_weak_view_agreement_reduces_recovery_credit(self):
        d = CandidateDecision(0, True, .5, .5, (), {})
        e = replace(effect(), evidence={"relation_score_delta": .2,
            "relation_confidence": .9, "cross_view_agreement": .1})
        self.assertLess(cause_residual_features(e, attr(CoarseCause.OBJECT_SHIFT), d)[0], .02)

    def test_terminal_relation_regression_is_soft_not_physical_violation(self):
        e = replace(effect(), hard_violations={"predicted_target_anchor_relation_regresses": .9})
        fixed = typed_gate_effect(e, "inside")
        self.assertFalse(fixed.hard_violations)
        self.assertEqual(fixed.evidence["terminal_progress_proxy_risk"], .9)

    def test_contact_loss_violation_is_not_overridden(self):
        e = replace(effect(), hard_violations={"predicted_target_gripper_contact_missing": .8})
        self.assertEqual(typed_gate_effect(e, "inside").hard_violations, e.hard_violations)

    def test_articulated_operation_does_not_lift_whole_drawer(self):
        for relation in ("open", "closed", "articulated"):
            fixed = typed_gate_effect(effect(), relation)
            self.assertFalse({"lifted", "grasped", "place_ready"} & fixed.required_facts.keys())
            self.assertFalse(fixed.proposed_effects)

    def test_rigid_place_still_requires_lifted(self):
        self.assertIn("lifted", typed_gate_effect(effect(), "inside").required_facts)

    def test_witnessed_false_prerequisite_is_not_overridden(self):
        b = initial_belief(0)
        b.facts["lifted"] = BeliefFact(TriValue.FALSE, .99, "observation")
        d = prepare_evidence_decisions(belief=b, attribution=attr(), block_index=3,
            effects=[effect()], values=[.9], relation="inside")[0]
        self.assertFalse(d.accepted)
        self.assertTrue(any("lifted=false" in x for x in d.rejection_reasons))

    def test_zero_residual_is_same_shared_backbone(self):
        ds = [CandidateDecision(i, True, v, v, (), {}) for i,v in enumerate((.5,.7))]
        xs = {i:candidate_backbone_features(effect(), d.official_value) for i,d in enumerate(ds)}
        chosen,_ = select_evidence_residual(decisions=ds, backbone_features=xs,
            cause_features={i:np.zeros(10) for i in xs}, backbone=model(), residual=residual())
        self.assertEqual(chosen.candidate_id, 1)

    def test_unresolved_routing_preserves_value_not_noisy_residual(self):
        b=initial_belief(0)
        es=[replace(effect(Stage.APPROACH),candidate_id=i) for i in range(2)]
        ds=prepare_evidence_decisions(belief=b,attribution=attr(CoarseCause.UNKNOWN),
            block_index=3,effects=es,values=[.5,.6],relation="inside",backbone=model())
        xs={i:candidate_backbone_features(e,v) for i,(e,v) in enumerate(zip(es,[.5,.6]))}
        chosen,_=select_evidence_residual(decisions=ds,backbone_features=xs,
            cause_features={0:np.ones(10),1:np.zeros(10)},backbone=model(),residual=residual(1.,1.))
        self.assertEqual(chosen.candidate_id,1)

    def test_epistemic_override_does_not_override_direct_contact_danger(self):
        b=initial_belief(0)
        e=replace(effect(),hard_violations={"predicted_target_gripper_contact_missing":.9})
        ds=prepare_evidence_decisions(belief=b,attribution=attr(CoarseCause.UNKNOWN),
            block_index=3,effects=[e],values=[.9],relation="inside",backbone=model())
        self.assertFalse(ds[0].accepted)

    def test_supported_cause_can_change_choice(self):
        ds = [CandidateDecision(i, True, v, v, (), {}) for i,v in enumerate((.5,.51))]
        xs = {i:candidate_backbone_features(effect(), d.official_value) for i,d in enumerate(ds)}
        cs = {0:np.eye(10)[0]*.1, 1:np.zeros(10)}
        chosen,_ = select_evidence_residual(decisions=ds, backbone_features=xs,
            cause_features=cs, backbone=model(), residual=residual(1.,.1))
        self.assertEqual(chosen.candidate_id, 0)

    def test_residual_cannot_select_rejected_candidate(self):
        ds = [CandidateDecision(0, False, .9, None, ("lifted=false",), {}),
              CandidateDecision(1, True, .5, .5, (), {})]
        xs = {i:candidate_backbone_features(effect(),d.official_value) for i,d in enumerate(ds)}
        chosen,_=select_evidence_residual(decisions=ds,backbone_features=xs,
            cause_features={0:np.ones(10),1:np.zeros(10)},backbone=model(),residual=residual(1.,1.))
        self.assertEqual(chosen.candidate_id,1)

    def test_extrapolation_falls_back_to_zero_correction(self):
        self.assertEqual(residual(1.,1.).score(np.ones(10)*11),0.)

    def test_normal_zero_is_not_centered_into_nonzero_correction(self):
        self.assertEqual(residual(1.,1.).score(np.zeros(10)),0.)

    def test_per_score_correction_bound(self):
        self.assertEqual(residual(1.,.05).score(np.ones(10)),.05)

    def test_schema_roundtrip(self):
        r=residual(1.,.1)
        self.assertEqual(EvidenceResidualModel.from_record(r.to_record()).to_record(),r.to_record())

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError): residual().score(np.ones(9))
        with self.assertRaises(ValueError): residual().score(np.ones(10)*np.nan)

    def test_fit_is_deterministic_and_normal_zero_stays_zero(self):
        support=[np.zeros(10),np.eye(10)[0]*.1]
        pairs=[(support[1],support[0],-.01)]
        a=fit_evidence_residual(pairs,support);b=fit_evidence_residual(pairs,support)
        np.testing.assert_array_equal(a.weights,b.weights)
        self.assertEqual(a.score(np.zeros(10)),0.)


if __name__ == "__main__": unittest.main()
