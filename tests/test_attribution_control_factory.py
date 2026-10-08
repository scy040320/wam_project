"""Synthetic before-only controls; not an empirical ablation or training gate."""
from copy import deepcopy
from dataclasses import replace
import inspect
import json
import unittest
import numpy as np

from wam_reranking.attribution_control_factory import (
    ControlPool, build_complete_attribution_controls, neutral_attribution, prepare_masked_context,control_factory_record)
from wam_reranking.belief import DEFAULT_GRAPH, DependencyGraph
from wam_reranking.contracts import (AttributionOutput,BeliefState,BeliefFact,
    CandidateEffect,CoarseCause,ConsistencyFactor,EvidenceQuality,Stage,TriValue)
from wam_reranking.recovery_contract import CurrentFactEvidence,prepare_recovery_context,ROUTES


def pool(task=46,state=10,split="train",cause=CoarseCause.OBJECT_SHIFT,quality=None):
    prior=BeliefState(task,"target","anchor",task_stage=Stage.LIFT)
    for name in prior.facts:
        prior.facts[name]=BeliefFact(TriValue.TRUE,.9,"observation",2,("verified:actual:before",))
    factors={f.value:.95 for f in ConsistencyFactor}
    flip={CoarseCause.OBJECT_SHIFT:ConsistencyFactor.WORLD_STATE_CONSISTENT,
          CoarseCause.EXECUTION_CONTACT_DEVIATION:ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT,
          CoarseCause.UNKNOWN:ConsistencyFactor.CAUSE_RESOLVED,
          CoarseCause.VISUAL_OCCLUSION:ConsistencyFactor.OBSERVATION_RELIABLE}.get(cause)
    if flip:factors[flip.value]=.1
    attr=AttributionOutput(factors,{c.value:float(c is cause) for c in CoarseCause},cause,.9,0.,
        quality or EvidenceQuality(True,True,True),f"predicted:{task}:{state}:3")
    plan=np.zeros((16,7));plan[:,6]=1.;plan[4:,2]=.2
    effects=(CandidateEffect(0,Stage.LIFT,{"grasped":.85},{"grasped":(TriValue.TRUE,.99)},.9,{}),
             CandidateEffect(1,Stage.LIFT,{"grasped":.85},{},.9,{}))
    goals=() if cause in (CoarseCause.UNKNOWN,CoarseCause.VISUAL_OCCLUSION) else ("grasped",)
    return ControlPool(identity=dict(dataset="native",suite="libero90",task=task,state=state,observed_block_id="block3"),
        pool_id=f"task{task}_state{state}",split=split,prior=prior,attribution=attr,observations=(),
        effects=effects,plans=(plan,plan.copy()),official_values=(.6,.5),backbone_scores=(.4,.45),
        budget={"K":2,"policy_steps":400,"best_of_n":1,"fallback_budget":400},block_index=3,
        frozen_boundary_goals=goals)


def variants(p):
    return build_complete_attribution_controls((p,))["pools"][0]["controls"]


class AttributionControlsTests(unittest.TestCase):
    def test_no_dag_reruns_belief_not_only_weights(self):
        v=variants(pool())
        self.assertIs(v["full"]["context"].belief.facts["grasped"].value,TriValue.UNKNOWN)
        self.assertIs(v["no_dag"]["context"].belief.facts["grasped"].value,TriValue.TRUE)
        self.assertIs(v["no_dag"]["context"].belief.facts["target_pose_current"].value,TriValue.FALSE)
        self.assertEqual(v["no_dag"]["context"].graph.edges,())
        self.assertEqual(v["no_dag"]["context"].graph.nodes,DEFAULT_GRAPH.nodes)
        self.assertTrue(v["no_dag"]["gate_differences_vs_full"])
        self.assertFalse(v["no_dag"]["gate_outcomes_equal_full"])

    def test_unknown_conf0_old_prepare_is_not_neutral(self):
        p=pool();n=neutral_attribution(p.attribution)
        ordinary=prepare_recovery_context(prior=p.prior,attribution=n,block_index=3)
        masked=prepare_masked_context(prior=p.prior,attribution=p.attribution,
            observations=(),block_index=3,graph=DEFAULT_GRAPH)
        self.assertIs(ordinary.belief.facts["target_pose_current"].value,TriValue.UNKNOWN)
        self.assertIs(masked.belief.facts["target_pose_current"].value,TriValue.TRUE)
        self.assertIs(masked.belief.task_stage,Stage.LIFT)
        self.assertEqual(masked.changes,())

    def test_mask_is_not_fake_normal_or_prediction(self):
        p=pool();v=variants(p)["masked"]
        self.assertIs(v["context"].attribution.projected_cause,CoarseCause.UNKNOWN)
        self.assertEqual(v["context"].attribution.confidence,0.)
        self.assertEqual(v["context"].attribution.factor_states,{f.value:TriValue.UNKNOWN for f in ConsistencyFactor})
        self.assertFalse(v["metadata"]["real_attribution_prediction_claimed"])
        self.assertTrue(v["metadata"]["placeholder_is_not_normal_or_real_unknown_prediction"])
        self.assertEqual(v["context"].belief.facts["grasped"].confidence,.9)
        self.assertFalse(any(c.reason=="consistent_block" for c in v["context"].changes))

    def test_mask_routes_all_zero_standalone_command_retained(self):
        for mode in ("masked","effect_only"):
            trace=variants(pool())[mode]["traces"][0]
            x=dict(zip(trace.feature_names,trace.features))
            self.assertTrue(all(value==0 for name,value in x.items() if any(name.startswith(r+".") for r in ROUTES)))
            self.assertEqual(x["command.has_close"],1.)
            self.assertEqual(x["command.has_upward"],1.)

    def test_common_objectives_preserved_without_fabricated_need(self):
        v=variants(pool());full=v["full"]["rows"][0]
        for mode in v:
            self.assertEqual(v[mode]["rows"][0]["requirements"],full["requirements"])
            self.assertEqual(v[mode]["rows"][0]["goals"],full["goals"])
        for mode in ("masked","effect_only"):
            self.assertTrue(all(n["weight"]==0 for row in v[mode]["rows"] for n in row["needs"]))

    def test_current_observation_can_refresh_in_every_control(self):
        p=replace(pool(),observations=(CurrentFactEvidence("grasped",TriValue.TRUE,.95,"actual:wrist:3",view="wrist"),))
        for mode,v in variants(p).items():
            self.assertIs(v["context"].belief.facts["grasped"].value,TriValue.TRUE)
            self.assertEqual(v["context"].belief.facts["grasped"].evidence_ids,("actual:wrist:3",))

    def test_bad_primary_reliable_wrist_still_refreshes(self):
        p=pool(quality=EvidenceQuality(False,True,True))
        p=replace(p,observations=(CurrentFactEvidence("grasped",TriValue.TRUE,.95,"actual:wrist",view="wrist"),))
        self.assertIs(variants(p)["masked"]["context"].belief.facts["grasped"].value,TriValue.TRUE)

    def test_bad_quality_does_not_create_physical_false(self):
        p=pool(cause=CoarseCause.UNKNOWN,quality=EvidenceQuality(False,False,True))
        v=variants(p)["masked"]
        self.assertIs(v["context"].belief.facts["target_visible"].value,TriValue.TRUE)
        self.assertIs(v["context"].belief.facts["grasped"].value,TriValue.TRUE)

    def test_quality_filter_matches_original_without_new_prior_invalidation(self):
        p=pool(cause=CoarseCause.NORMAL,quality=EvidenceQuality(False,False,True))
        obs=CurrentFactEvidence("receptacle_visible",TriValue.FALSE,.95,"unusable:actual",view="primary")
        p=replace(p,observations=(obs,))
        v=variants(p)
        for name in p.prior.facts:
            self.assertEqual(v["masked"]["context"].belief.facts[name],v["full"]["context"].belief.facts[name])
        self.assertEqual(v["masked"]["context"].changes,())
        self.assertFalse(v["masked"]["metadata"]["extra_quality_prior_invalidation_added"])
        self.assertFalse(v["masked"]["metadata"]["D18_perception_quality_removed"])

    def test_mask_keeps_real_measured_record_hard_false(self):
        obs=CurrentFactEvidence("execution_consistent",TriValue.FALSE,.99,"actual:commands",source="measured_execution",view="record")
        p=replace(pool(),observations=(obs,))
        for mode,v in variants(p).items():
            self.assertIs(v["context"].belief.facts["execution_consistent"].value,TriValue.FALSE)
            self.assertEqual(v["metadata"]["explicit_record_hard_evidence_count"],1)

    def test_unreliable_record_cannot_certify_grasp(self):
        p=pool();bad=CurrentFactEvidence("grasped",TriValue.TRUE,.99,"actual:command",source="measured_execution",view="record")
        with self.assertRaisesRegex(ValueError,"Command equality"):
            variants(replace(p,observations=(bad,)))

    def test_witnessed_false_not_erased_by_mask(self):
        p=pool();p.prior.facts["grasped"]=BeliefFact(TriValue.FALSE,.95,"observation",2,("actual:lost",))
        v=variants(p)["masked"]
        self.assertIs(v["context"].belief.facts["grasped"].value,TriValue.FALSE)
        self.assertFalse(v["gates"][0]["accepted"])

    def test_future_endpoint_never_repairs_before(self):
        p=pool();before=deepcopy(p.prior.snapshot())
        v=variants(p)
        self.assertIs(v["full"]["context"].belief.facts["grasped"].value,TriValue.UNKNOWN)
        self.assertFalse(v["full"]["gates"][0]["accepted"])
        self.assertEqual(p.prior.snapshot(),before)
        self.assertEqual(v["full"]["rows"][0]["requirements"][0]["deadline"],4)
        self.assertEqual(v["full"]["rows"][0]["goals"][0]["deadline"],16)

    def test_candidate_hard_violation_shared(self):
        p=pool();e=replace(p.effects[0],hard_violations={"trajectory_collision":.95})
        p=replace(p,effects=(e,p.effects[1]))
        for v in variants(p).values():
            self.assertIn("candidate_evidence:trajectory_collision",v["gates"][0]["rejection_reasons"])

    def test_budget_x_base_cap_wrapper_same(self):
        p=pool();v=variants(p)
        for mode,control in v.items():
            self.assertEqual(control["common_before_and_candidate_x_sha256"],v["full"]["common_before_and_candidate_x_sha256"])
            self.assertEqual(control["budget"],p.budget)
            self.assertEqual(control["backbone_scores"],p.backbone_scores)
            self.assertEqual(control["residual_cap"],.1)
            self.assertEqual(control["selection_wrapper_id"],p.selection_wrapper_id)
            self.assertEqual(control["fallback"],p.fallback)

    def test_shuffle_cross_task_state_and_split_local_real_rerun(self):
        pools=(pool(),pool(task=57,state=11,cause=CoarseCause.EXECUTION_CONTACT_DEVIATION),
               pool(task=0,state=12,split="val"),pool(task=9,state=13,split="val",cause=CoarseCause.EXECUTION_CONTACT_DEVIATION))
        run=build_complete_attribution_controls(pools)
        for item in run["pools"]:
            v=item["controls"]["shuffled"];donor=v["metadata"]["donor_identity"]
            self.assertEqual(donor["split"],item["split"])
            self.assertNotEqual((donor["task"],donor["state"]),(item["identity"]["task"],item["identity"]["state"]))
            receiver=next(p for p in pools if p.pool_id==item["pool_id"])
            source=next(p for p in pools if p.pool_id==donor["pool_id"] and p.split==donor["split"])
            self.assertEqual(v["context"].attribution.class_probs,source.attribution.class_probs)
            self.assertEqual(v["context"].attribution.source_block_id,receiver.attribution.source_block_id)
            self.assertEqual(v["context"].attribution.evidence_quality,receiver.attribution.evidence_quality)
        again=build_complete_attribution_controls(pools)
        self.assertEqual([x["controls"]["shuffled"]["metadata"] for x in run["pools"]],
                         [x["controls"]["shuffled"]["metadata"] for x in again["pools"]])

    def test_single_group_shuffle_not_self_donor(self):
        v=variants(pool())["shuffled"]
        self.assertIsNone(v["metadata"]["donor_identity"])
        self.assertTrue(v["metadata"]["masked_adapter_used"])
        self.assertEqual(v["metadata"]["shuffled_missing_donor"],"no_complete_split_local_cross_task_state_bijection_masked")

    def test_shuffle_normal_donor_confidence_uses_receiver_forced_unknown_direct_slot(self):
        receiver=pool(cause=CoarseCause.UNKNOWN,quality=EvidenceQuality(False,False,True,True))
        donor=pool(task=57,state=11,cause=CoarseCause.NORMAL)
        probs={c.value:.0025 for c in CoarseCause};probs["normal"]=.99
        donor=replace(donor,attribution=replace(donor.attribution,class_probs=probs,confidence=.99))
        run=build_complete_attribution_controls((receiver,donor))
        controlled=next(p for p in run["pools"] if p["pool_id"]==receiver.pool_id)["controls"]["shuffled"]
        self.assertIs(controlled["context"].attribution.projected_cause,CoarseCause.UNKNOWN)
        self.assertEqual(controlled["context"].attribution.confidence,.0025)
        self.assertEqual(controlled["context"].attribution.class_probs,probs)
        self.assertEqual(controlled["context"].attribution.evidence_quality,receiver.attribution.evidence_quality)
        self.assertTrue(controlled["metadata"]["shuffled_selected_confidence_from_original_direct_projected_class"])
        self.assertFalse(controlled["metadata"]["shuffled_confidence_calibrated"])
        self.assertFalse(controlled["metadata"]["shuffled_probability_values_modified"])
        self.assertFalse(controlled["metadata"]["shuffled_selected_confidence_increased_by_class_projection"])
        self.assertEqual(controlled["metadata"]["shuffled_original_selected_confidence"],.99)

    def test_shuffle_same_bare_task_state_in_different_suites_is_distinct_group(self):
        first=pool();second=replace(pool(),pool_id="same_ids_other_suite",
            identity=dict(first.identity,suite="fixture_other_suite"))
        run=build_complete_attribution_controls((first,second))
        self.assertTrue(run["shuffled_complete_prediction_bijection_available"])
        for recipient in run["pools"]:
            donor=recipient["controls"]["shuffled"]["metadata"]["donor_identity"]
            self.assertNotEqual(donor["suite"],recipient["identity"]["suite"])
            self.assertEqual((donor["task"],donor["state"]),
                (recipient["identity"]["task"],recipient["identity"]["state"]))

    def test_shuffle_multiple_blocks_is_bijection_not_donor_replication(self):
        a=pool();b=replace(pool(),pool_id="another_block",identity=dict(a.identity,observed_block_id="block4"))
        c=pool(task=57,state=11);d=replace(c,pool_id="other_second",identity=dict(c.identity,observed_block_id="block4"))
        run=build_complete_attribution_controls((a,b,c,d))
        donor_keys=[]
        for item in run["pools"]:
            donor=item["controls"]["shuffled"]["metadata"]["donor_identity"]
            donor_keys.append((donor["task"],donor["state"],donor["observed_block_id"],donor["pool_id"]))
        self.assertEqual(len(set(donor_keys)),4)
        self.assertTrue(run["shuffled_complete_prediction_bijection_available"])

    def test_unbalanced_groups_do_not_force_distribution_change(self):
        a=pool();b=replace(a,pool_id="second",identity=dict(a.identity,observed_block_id="block4"))
        c=pool(task=57,state=11)
        run=build_complete_attribution_controls((a,b,c))
        self.assertFalse(run["shuffled_complete_prediction_bijection_available"])
        self.assertTrue(all(item["controls"]["shuffled"]["metadata"]["donor_identity"] is None for item in run["pools"]))

    def test_prior_attribution_history_scope_is_not_concealed(self):
        p=pool();p.prior.facts["lifted"]=BeliefFact(TriValue.UNKNOWN,.5,"attribution",1,("old:block",))
        v=variants(p)["masked"]
        self.assertIn("lifted",v["metadata"]["prior_attribution_facts_retained"])
        self.assertTrue(v["metadata"]["not_a_history_reconstructed_closed_loop_ablation"])

    def test_scope_configs_bound_and_export_json(self):
        p=pool();a=variants(p)["full"]["common_before_and_candidate_x_sha256"]
        b=variants(replace(p,information_acquisition=True))["full"]["common_before_and_candidate_x_sha256"]
        self.assertNotEqual(a,b)
        record=control_factory_record(build_complete_attribution_controls((p,)))
        json.dumps(record,allow_nan=False)
        self.assertFalse(record["actual_dataset_fairness_pass"])

    def test_native_official_value_range_preserved_and_nonfinite_rejected(self):
        p=replace(pool(),official_values=(-1.3,4.2))
        record=control_factory_record(build_complete_attribution_controls((p,)))
        shared=record["pools"][0]["controls"]["full"]
        self.assertEqual(shared["official_values"],[-1.3,4.2])
        for values in ((float("nan"),.1),(float("inf"),.1)):
            with self.assertRaisesRegex(ValueError,"Finite official"):
                variants(replace(p,official_values=values))

    def test_unknown_seed_no_confirmation_no_duplicate_pools(self):
        with self.assertRaises(ValueError):build_complete_attribution_controls((pool(),),seed=1)
        with self.assertRaises(ValueError):variants(replace(pool(),split="confirmation"))
        p=pool()
        with self.assertRaises(ValueError):build_complete_attribution_controls((p,p))

    def test_identity_no_bool_string_or_empty_namespace_aliases(self):
        p=pool()
        for key,value in (("task",True),("state",False),("task","46"),("state",-1),
                          ("dataset",""),("suite"," "),("observed_block_id",3)):
            with self.assertRaisesRegex(ValueError,"Identity"):
                variants(replace(p,identity=dict(p.identity,**{key:value})))

    def test_no_y_api_or_hidden_outcome_input(self):
        self.assertNotIn("labels",inspect.signature(build_complete_attribution_controls).parameters)
        p=pool();e=replace(p.effects[0],evidence={"success":1.})
        with self.assertRaisesRegex(ValueError,"outcomes"):
            variants(replace(p,effects=(e,p.effects[1])))
        result=build_complete_attribution_controls((p,))
        self.assertFalse(result["training_ready"])
        self.assertFalse(result["actual_dataset_fairness_pass"])
        self.assertFalse(result["outcomes_read"])

    def test_assumptions_masked_not_supposed_current_observation(self):
        p=pool();p.prior.facts["grasped"]=BeliefFact(TriValue.TRUE,.99,"candidate_prediction",2,("forecast",))
        for mode,v in variants(p).items():
            self.assertIs(v["context"].belief.facts["grasped"].value,TriValue.UNKNOWN)


if __name__=="__main__":
    unittest.main()
