"""Synthetic source/ablation tests, not physical-certification or benefit evidence."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np

from wam_reranking.contracts import AttributionOutput, CoarseCause, ConsistencyFactor, EvidenceQuality, TriValue
from wam_reranking.shared_current_visual_baseline import build_shared_current_visual_evidence
from wam_reranking.source_scoped_forecast_features import (
    MAIN_NAMES, CHANNEL_NAMES, EXPOSURE_SCOPES, INTERACTION_NAMES, TrainChannelCentering, _digest,
    calibrate_train_channels, source_scoped_features,
)


def identity(state=0):
    return dict(dataset="synthetic-source", suite="test-suite", task=2, state=state,
                moment=0, condition="source-only", block_id=f"actual:block3:state{state}", block_index=3)


def attribution(source, world=.15, quality=None):
    factors = {f.value: .95 for f in ConsistencyFactor}
    factors[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = 1.-world
    cp = {c.value: .025 for c in CoarseCause}; cp[CoarseCause.UNKNOWN.value] = .9
    return AttributionOutput(factors, cp, CoarseCause.UNKNOWN, .9, .7,
        quality or EvidenceQuality(True, True, True), source,
        {k: TriValue.TRUE if v >= .5 else TriValue.FALSE for k,v in factors.items()},
        {k: max(v,1.-v) for k,v in factors.items()})


def peak(x,y):
    v = np.full((8,8),.01);v[y,x]=1.;return v


def visual(i):
    return build_shared_current_visual_evidence(current_subject_maps=(peak(2,3),peak(5,4)),
        current_anchor_maps=(peak(5,5),peak(2,4)), current_gripper_maps=(peak(2,2),peak(5,3)),
        predicted_subject_maps=(peak(2+i,3),peak(5-i,4)),
        predicted_anchor_maps=(peak(5,5),peak(2,4)),
        predicted_gripper_maps=(peak(2+i,2),peak(5-i,3)), relation="inside")


def inputs():
    ident = identity(); visuals, diagnostics = zip(*(visual(i) for i in range(2)))
    plans=[]
    for i in range(2):
        a=np.zeros((16,7));a[:8,6]=-1.;a[8:,6]=1.;a[:,2]=.02+.005*i;plans.append(a)
    sources={name:"a"*64 for name in ("current_primary","current_wrist","localizer_model",
        "localizer_code","shared_fusion_code","candidate_parser_code","attribution")}
    sources.update({f"candidate_{i}_{field}":"b"*64 for i in range(2)
                    for field in ("plan","predicted_primary","predicted_wrist")})
    center=calibrate_train_channels([dict(identity=ident,split="train",
        source_sha256={"attribution":"c"*64}, attribution=attribution(ident["block_id"],.1))])
    snapshot=dict(facts={},history=[])
    return dict(attribution=attribution(ident["block_id"],.3), plans=plans,
        visual_evidences=visuals,fusion_diagnostics=diagnostics,identity=ident,
        source_sha256=sources,split="train",variant="full",belief_snapshot=snapshot,
        history_context=dict(identity=ident,variant="full",no_dag=False,snapshot_sha256=_digest(snapshot)),
        centering=center,relation="inside")


def change(predicate="target_pose_current",value="unknown",confidence=.9,
           reason="world_state_consistent",evidence="old_world",path=None,direct=True):
    return dict(predicate=predicate,old_value="true",old_confidence=.7,new_value=value,
        new_confidence=confidence,reason=reason,direct=direct,
        propagation_path=path or [predicate],evidence_id=evidence)


def set_snapshot(kw,snapshot):
    kw["belief_snapshot"]=snapshot
    kw["history_context"]["snapshot_sha256"]=_digest(snapshot)


class SharedForecastFeatureTests(unittest.TestCase):
    def test_mask_keeps_every_ordinary_main_effect_identical(self):
        kw=inputs();main,interaction,audit=source_scoped_features(**kw)
        masked,removed,ma=source_scoped_features(**kw,mask_learned=True)
        np.testing.assert_array_equal(main,masked)
        self.assertTrue(interaction.any());self.assertFalse(removed.any())
        for name in ("contact_after","grasp_after","ordered_release_forecast","forecast_visibility",
                     "forecast_uncertainty","trajectory_risk","release_step_fraction"):
            self.assertIn(name,MAIN_NAMES)
        self.assertTrue(ma["full_ordinary_main_capacity_preserved"])
        self.assertFalse(audit["deployed"])

    def test_constant_training_mean_removes_ordinary_capacity_from_interactions(self):
        kw=inputs();kw["attribution"]=attribution(kw["identity"]["block_id"],.1)
        main,interaction,_=source_scoped_features(**kw)
        self.assertTrue(np.ptp(main,axis=0).any())
        self.assertFalse(interaction.any())

    def test_old_cache_schema_and_mixed_current_before_rejected(self):
        kw=inputs();bad=deepcopy(kw["fusion_diagnostics"]);bad[0]["schema"]="historical_cached_visual_v1"
        with self.assertRaisesRegex(ValueError,"legacy cached BEFORE"):
            source_scoped_features(**{**kw,"fusion_diagnostics":bad})
        bad=list(kw["visual_evidences"]);bad[1]=replace(bad[1],grasp_support_before=.999)
        with self.assertRaisesRegex(ValueError,"CURRENT baseline"):
            source_scoped_features(**{**kw,"visual_evidences":bad})

    def test_ordered_parser_cannot_infer_release_from_open_at_entry(self):
        kw=inputs();plan=np.zeros((16,7));plan[:4,6]=1.;plan[4:,6]=-1.;plan[:,2]=.02
        kw["plans"][0]=plan
        main,_,audit=source_scoped_features(**kw)
        self.assertEqual(main[0,MAIN_NAMES.index("ordered_release_present")],0.)
        self.assertEqual(main[0,MAIN_NAMES.index("ordered_release_forecast")],0.)
        self.assertEqual(audit["parser"],"ordered_release_v1")

    def test_no_belief_attribution_plan_or_map_mutation_and_no_certificate(self):
        kw=inputs();before=deepcopy(kw["belief_snapshot"]);a=deepcopy(kw["attribution"])
        plans=[p.copy() for p in kw["plans"]]
        main,interaction,audit=source_scoped_features(**kw)
        self.assertEqual(kw["belief_snapshot"],before);self.assertEqual(kw["attribution"],a)
        for actual,expected in zip(kw["plans"],plans):np.testing.assert_array_equal(actual,expected)
        self.assertEqual(audit["current_physical_certificates_generated"],0)
        self.assertFalse(audit["observation_registry_populated"])
        self.assertEqual(audit["prediction_writes"],0)
        self.assertFalse(audit["proxy_geometry_is_physical_truth"])
        with self.assertRaises(ValueError):main[0,0]=1.
        with self.assertRaises(ValueError):interaction[0,0]=1.

    def test_learned_quality_metadata_does_not_erase_ordinary_forecast_capacity(self):
        kw=inputs();main,_,_=source_scoped_features(**kw)
        kw["attribution"]=replace(kw["attribution"],
            evidence_quality=EvidenceQuality(False,False,True,True))
        unreliable,interaction,audit=source_scoped_features(**kw)
        np.testing.assert_array_equal(main,unreliable)
        view=interaction.reshape(2,len(EXPOSURE_SCOPES),len(CHANNEL_NAMES),len(MAIN_NAMES))
        self.assertFalse(view[:,:,[0,2,3],:].any())
        self.assertFalse(audit["learned_visual_channel_usable"])
        self.assertFalse(audit["ordinary_main_gated_by_learned_quality"])
        self.assertTrue(audit["independent_rgb_and_extractor_byte_authentication_required"])
        self.assertFalse(audit["proxy_geometry_is_physical_truth"])
        self.assertEqual(audit["prediction_writes"],0)

    def test_execution_soft_channel_uses_own_quality_not_visual_quality(self):
        kw=inputs();a=kw["attribution"]
        factors={**a.factor_probs,ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value:.2}
        kw["attribution"]=replace(a,factor_probs=factors,
            evidence_quality=EvidenceQuality(False,False,True))
        main,interaction,audit=source_scoped_features(**kw)
        view=interaction.reshape(2,2,len(CHANNEL_NAMES),len(MAIN_NAMES))
        self.assertTrue(view[:,:,1,:].any())
        self.assertTrue(audit["learned_execution_contact_probability_is_not_certificate"])
        kw["attribution"]=replace(kw["attribution"],evidence_quality=EvidenceQuality(False,False,False))
        same,removed,_=source_scoped_features(**kw)
        np.testing.assert_array_equal(main,same)
        self.assertFalse(removed.reshape(2,2,len(CHANNEL_NAMES),len(MAIN_NAMES))[:,:,1,:].any())

    def test_invalidation_and_unverified_unknown_have_distinct_fixed_scopes(self):
        kw=inputs();kw["relation"]="articulated"
        main,unknown,baseline=source_scoped_features(**kw)
        self.assertEqual(unknown.shape,(2,364));self.assertEqual(len(INTERACTION_NAMES),364)
        event=change()
        snapshot=dict(facts={"target_pose_current":dict(value="unknown",confidence=.9,
            source="attribution",updated_at_block=3,evidence_ids=["old_world"])},history=[event])
        set_snapshot(kw,snapshot)
        same,invalid,audit=source_scoped_features(**kw)
        np.testing.assert_array_equal(main,same);self.assertFalse(np.array_equal(unknown,invalid))
        self.assertEqual(audit["belief_context"]["target_pose_current"]["value"],"unknown")
        self.assertTrue(audit["invalidation_evidence_is_not_physical_FALSE"])
        self.assertEqual(audit["certified_prior_facts"],[])
        self.assertFalse(invalid[:,:182].tolist()==unknown[:,:182].tolist())

    def test_old_declared_true_stays_soft_context_never_current_certificate(self):
        kw=inputs();kw["relation"]="articulated"
        _,unknown,_=source_scoped_features(**kw)
        snapshot=dict(facts={"target_pose_current":dict(value="true",confidence=.9,
            source="observation",updated_at_block=1,evidence_ids=["old_observation"])},history=[])
        set_snapshot(kw,snapshot)
        _,soft,audit=source_scoped_features(**kw)
        self.assertFalse(np.array_equal(unknown,soft));self.assertEqual(audit["certified_prior_facts"],[])
        self.assertEqual(audit["belief_context"]["target_pose_current"]["retained_declared_TRUE_soft_prior"],.9)
        self.assertTrue(audit["retained_prior_cannot_pass_current_hard_gate"])
        snapshot["facts"]["target_pose_current"]["source"]="candidate_prediction";set_snapshot(kw,snapshot)
        _,forecast,_=source_scoped_features(**kw)
        np.testing.assert_array_equal(unknown,forecast)

    def test_current_verified_requires_source_time_history_and_original_threshold(self):
        from wam_reranking.reranker import PREDICATE_HARD_THRESHOLDS
        kw=inputs();kw["relation"]="articulated"
        event=change(value="true",confidence=.9,reason="current_observation",evidence="current_obs")
        snapshot=dict(facts={"target_pose_current":dict(value="true",confidence=.9,
            source="observation",updated_at_block=3,evidence_ids=["current_obs"])},history=[event])
        set_snapshot(kw,snapshot)
        _,_,audit=source_scoped_features(**kw)
        self.assertEqual(audit["certified_prior_facts"],["target_pose_current"])
        low=PREDICATE_HARD_THRESHOLDS["target_pose_current"]-.01
        snapshot["facts"]["target_pose_current"]["confidence"]=low
        event["new_confidence"]=low;set_snapshot(kw,snapshot)
        _,_,audit=source_scoped_features(**kw)
        self.assertEqual(audit["certified_prior_facts"],[])
        snapshot["facts"]["target_pose_current"]["updated_at_block"]=4;set_snapshot(kw,snapshot)
        with self.assertRaisesRegex(ValueError,"source/time"):source_scoped_features(**kw)

    def test_persistent_world_history_not_erased_by_new_consistent_or_candidate_true(self):
        kw=inputs();kw["relation"]="articulated"
        old=change(confidence=.95)
        latest=change(value="true",confidence=.9,reason="consistent_block",evidence="new_normal")
        snapshot=dict(facts={"target_pose_current":dict(value="true",confidence=.9,
            source="observation",updated_at_block=3,evidence_ids=["new_normal"])},history=[old,latest])
        set_snapshot(kw,snapshot)
        _,protected,audit=source_scoped_features(**kw)
        context=audit["belief_context"]["target_pose_current"]
        self.assertEqual(context["historical_invalidation_evidence"][0]["root_strength"],.95)
        self.assertFalse(context["historical_invalidation_evidence"][0]["historical_block_time_authenticated"])
        self.assertEqual(audit["certified_prior_facts"],[])
        snapshot["facts"]["target_pose_current"]["source"]="candidate_prediction";set_snapshot(kw,snapshot)
        _,candidate,_=source_scoped_features(**kw)
        np.testing.assert_array_equal(protected[:,:182],candidate[:,:182])
        snapshot["facts"]["target_pose_current"]["source"]="observation"
        latest["reason"]="current_observation";set_snapshot(kw,snapshot)
        _,resolved,audit=source_scoped_features(**kw)
        self.assertEqual(audit["certified_prior_facts"],["target_pose_current"])
        self.assertFalse(resolved[:,:182].any())

    def test_articulated_soft_requirements_do_not_require_carrying_the_joint(self):
        for relation in ("open","closed","articulated"):
            kw=inputs();kw["relation"]=relation
            _,_,audit=source_scoped_features(**kw)
            for req in audit["soft_required_facts"]:
                self.assertEqual(req,{"target_visible":.5,"target_pose_current":.6})
                self.assertNotIn("grasped",req);self.assertNotIn("lifted",req)
                self.assertNotIn("place_ready",req)

    def test_mismatched_task_or_latest_fact_history_is_rejected(self):
        kw=inputs();set_snapshot(kw,dict(task_id=9,facts={},history=[]))
        with self.assertRaisesRegex(ValueError,"source identity"):source_scoped_features(**kw)
        snapshot=dict(facts={"target_pose_current":dict(value="unknown",confidence=.9,
            source="attribution",updated_at_block=3,evidence_ids=["old_world"])},
            history=[change(confidence=.8)])
        set_snapshot(kw,snapshot)
        with self.assertRaisesRegex(ValueError,"history and live fact"):source_scoped_features(**kw)

    def test_no_dag_and_other_variant_history_cannot_be_borrowed(self):
        kw=inputs()
        with self.assertRaisesRegex(ValueError,"borrowed arm"):
            source_scoped_features(**{**kw,"variant":"no_dag","no_dag":True})
        kw["variant"]="no_dag";kw["no_dag"]=True
        kw["belief_snapshot"]["history"]=[dict(predicate="grasped",new_value="unknown",
            reason="world_state_consistent",propagation_path=["target_pose_current","target_reachable","grasped"])]
        kw["history_context"]=dict(identity=kw["identity"],variant="no_dag",no_dag=True,
            snapshot_sha256=_digest(kw["belief_snapshot"]))
        with self.assertRaisesRegex(ValueError,"propagated history"):
            source_scoped_features(**kw)

    def test_only_upstream_current_transition_closes_dependency_exposure(self):
        kw=inputs()
        snapshot=dict(facts={"lifted":dict(value="true",source="candidate_prediction",confidence=.9,
            updated_at_block=3,evidence_ids=["forecast"])},history=[])
        kw["belief_snapshot"]=snapshot;kw["history_context"]["snapshot_sha256"]=_digest(snapshot)
        _,_,audit=source_scoped_features(**kw)
        self.assertEqual(audit["certified_prior_facts"],[])

    def test_sources_and_terminal_metadata_must_not_enter_features(self):
        kw=inputs();del kw["source_sha256"]["candidate_0_predicted_wrist"]
        with self.assertRaisesRegex(ValueError,"own-plan/forecast"):
            source_scoped_features(**kw)
        kw=inputs();kw["belief_snapshot"]["outcomes"]=[True]
        kw["history_context"]["snapshot_sha256"]=_digest(kw["belief_snapshot"])
        with self.assertRaisesRegex(ValueError,"audited belief"):
            source_scoped_features(**kw)


class TrainCenteringTests(unittest.TestCase):
    def record(self,state=0):
        ident=identity(state)
        return dict(identity=ident,split="train",source_sha256={"attribution":"a"*64},
                    attribution=attribution(ident["block_id"],.1+.05*state))

    def test_train_only_order_independent_fingerprint_and_no_labels(self):
        records=[self.record(0),self.record(1)]
        a=calibrate_train_channels(records);b=calibrate_train_channels(records[::-1])
        self.assertEqual(a.fingerprint,b.fingerprint);self.assertEqual(a.mean,b.mean)
        self.assertEqual(a.source_count,2);self.assertEqual(len(a.mean),len(CHANNEL_NAMES))
        bad=self.record();bad["split"]="val"
        with self.assertRaisesRegex(ValueError,"TRAIN-only"):calibrate_train_channels([bad])
        bad=self.record();bad["outcome"]={"success":True}
        with self.assertRaisesRegex(ValueError,"no outcomes/labels"):calibrate_train_channels([bad])

    def test_missing_duplicate_or_cross_source_identity_rejected(self):
        bad=self.record();del bad["identity"]["dataset"]
        with self.assertRaisesRegex(ValueError,"source identity"):calibrate_train_channels([bad])
        bad=self.record();bad["source_sha256"]={}
        with self.assertRaisesRegex(ValueError,"source SHA256"):calibrate_train_channels([bad])
        with self.assertRaisesRegex(ValueError,"Duplicate"):calibrate_train_channels([self.record(),self.record()])
        bad=self.record();bad["attribution"]=attribution("borrowed:block2")
        with self.assertRaisesRegex(ValueError,"source block"):calibrate_train_channels([bad])

    def test_centering_rejects_nonfinite_nonhex_and_nonliteral_count(self):
        valid=dict(mean=(.5,)*len(CHANNEL_NAMES),fingerprint="a"*64,source_count=1)
        for value in (float("nan"),float("inf"),float("-inf"),-.1,1.1):
            with self.subTest(mean=value),self.assertRaises(ValueError):
                TrainChannelCentering(**{**valid,"mean":(value,)+valid["mean"][1:]})
        for value in ("G"*64,"A"*64,"a"*63,64,None):
            with self.subTest(fingerprint=value),self.assertRaises(ValueError):
                TrainChannelCentering(**{**valid,"fingerprint":value})
        for value in (True,1.,"1",0,-1,None):
            with self.subTest(source_count=value),self.assertRaises(ValueError):
                TrainChannelCentering(**{**valid,"source_count":value})
        mutable=[.5]*len(CHANNEL_NAMES)
        calibrated=TrainChannelCentering(**{**valid,"mean":mutable})
        mutable[0]=.1
        self.assertIsInstance(calibrated.mean,tuple);self.assertEqual(calibrated.mean[0],.5)


if __name__=="__main__":unittest.main()
