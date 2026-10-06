"""CPU regressions for finite full-boundary native observer probes; no model."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from wam_reranking import native_block_reobservation_supervision as labels


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);x=importlib.util.module_from_spec(spec)
    sys.modules[name]=x;spec.loader.exec_module(x);return x


collector=module("native_collector_test",ROOT/"scripts/native_block_reobservation_collect.py")
launcher=module("native_launcher_test",ROOT/"scripts/native_block_reobservation_launch.py")
prior=module("native_prior_test",ROOT/"tests/test_post_prefix_reobservation.py")
helpers=prior.helpers;prefix_helpers=prior.labels
SOURCE=dict(dataset="frozen_source",suite="libero90",task=9,state=10,candidate_id=0,observed_block_id="source/first_block")
NATIVE=dict(dataset=collector.VERSION,suite="libero90",task=9,state=10,candidate_id=0,observed_block_id="source_post16/native_first_block")
QUALITY=dict(schema="actual_before_deployment_observer_quality_v1",quality=dict(primary_reliable=False,wrist_reliable=True,
    execution_reliable=True,cross_view_conflict=False,observer_evidence_available=False),predicted_cause="visual_occlusion")


def frame(step,available=False):
    x=prior.frame(step,available)
    x["observation_evidence"]["private_collision_vs_visual_mapping"]={"target":[dict(body_name=labels.TARGET)]}
    return x


def proof():
    p=prior.proof();p.update(identity=SOURCE,reference_step=16,used_original_steps=list(range(1,17)),
        checks=[dict(step=s,primary=prior.rgb(),wrist=prior.rgb(),proprio_max_abs=0.,passed=True) for s in range(1,17)],
        full_native_block_boundary=True,complete_source_steps=16,restore_order_audit=dict(full_runtime_bitwise_equal=True))
    return p


def after_quality(available=True):
    record=dict(schema="actual_after_frozen_deployment_observer_quality_v1",role="offline_supervision_only",private_teacher_used=False,
        actual_after_used=True,identity=NATIVE,step=16,relative_step=16,predicted_cause="visual_occlusion",quality=dict(QUALITY["quality"],observer_evidence_available=available),
        frozen_attributor_model_sha256=labels.ATTRIBUTOR_SHA,query_observation_sha256="c"*64,
        full_source_prefix_runtime_sha256=proof()["new_runtime_snapshot_sha256"],
        actual_array_sources={v:dict(path=v+".npy",sha256="b"*64,frame_index=16) for v in ("primary","wrist")},
        actual_attribution_source=dict(path="actual_after_attribution.json",sha256="b"*64),
        subject_maps_source=dict(path="actual_after_target_maps.npy",sha256="b"*64),
        observer_code_source=dict(path="observed_recovery.py",sha256=labels.OBSERVER_CODE_SHA),
        observer_model_source=dict(path="frozen_d18.pt",sha256=labels.ATTRIBUTOR_SHA),
        actual_after_rgb_sources={v:dict(path="actual_after_"+v+"_rgb.npy",sha256="b"*64) for v in ("primary","wrist")})
    binding={k:record[k] for k in ("identity","step","actual_after_rgb_sources","subject_maps_source","observer_code_source","observer_model_source")}
    record["observer_input_lineage_sha256"]=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return record


def after_maps(available=True):
    maps=np.zeros((2,8,8),np.float32)
    if available:maps[:,2,3]=1.
    return maps


class LabelTests(unittest.TestCase):
    def derive(self,frames=None,before=None,p=None,stage="lift",quality=None,goal=None,after=None,maps=None):
        q=copy.deepcopy(quality or QUALITY)
        record=after or after_quality()
        return labels.derive_native_goal(frames or [frame(s,s>=2) for s in range(17)],before or frame(16),
            source_identity=SOURCE,native_identity=NATIVE,verified_prefix_evidence=p or proof(),helpers=helpers,prefix_helpers=prefix_helpers,
            native_stage=stage,before_deployment_quality=q,frozen_goal=goal or labels.secondary_goal(q["quality"],predicted_cause=q["predicted_cause"]),
            actual_after_deployment_quality=record,actual_after_quality_source=dict(path="actual_after_quality.json",sha256="b"*64),native_query_observation_sha256="c"*64,
            actual_after_subject_maps=after_maps(record["quality"]["observer_evidence_available"]) if maps is None else maps)
    def test_native_lift_stage_kept_secondary_goal_only16(self):
        x=self.derive();self.assertEqual(x["native_stage"],"lift");self.assertEqual(x["purpose"],"task_with_observation_recovery")
        self.assertEqual(x["secondary_observer_goal"]["deadline"],16);self.assertTrue(x["secondary_observer_goal"]["value"])
        self.assertFalse(x["before"]["value"]);self.assertEqual(x["before"]["original_source_step"],16)
        self.assertTrue(x["no_future_backfill_or_predicate_gate_unlock"]);self.assertFalse(x["ranking_training_ready"])
    def test_zero_recovery_is_retained_epistemic_negative(self):
        x=self.derive([frame(s) for s in range(17)],after=after_quality(False));self.assertFalse(x["secondary_observer_goal"]["value"])
        self.assertTrue(x["secondary_observer_goal"]["mask"]);self.assertFalse(x["physical_predicate_mapping_admitted"])
        self.assertIsNone(x["semantic_conflict_resolved"]["value"]);self.assertFalse(x["semantic_conflict_resolved"]["mask"])
    def test_no_after_backfill_of_before(self):
        fs=[frame(s,True) for s in range(17)];x=self.derive(fs);self.assertFalse(x["before"]["value"])
    def test_uncalculated_midframe_observer_Y_always_masked(self):
        x=self.derive()
        for row in x["frames"][1:16]:
            self.assertIsNone(row["value"]);self.assertFalse(row["mask"])
            self.assertIs(type(row["actual_rgb_certificate_available"]),bool)
        self.assertTrue(x["frames"][16]["value"]);self.assertTrue(x["frames"][16]["mask"])
    def test_actual_available_before_is_not_reselected(self):
        with self.assertRaises(ValueError):self.derive(before=frame(16,True))
    def test_cached_short_prefix_is_not_full_boundary(self):
        for reference in (0,1,2,4,15):
            p=proof();p["reference_step"]=reference
            with self.assertRaises(ValueError):self.derive(p=p)
    def test_other_target_body_cannot_fill_native_certificate(self):
        fs=[frame(s,s>=2) for s in range(17)]
        fs[16]["observation_evidence"]["private_collision_vs_visual_mapping"]["target"][0]["body_name"]="plate_1"
        with self.assertRaises(ValueError):self.derive(fs)
    def test_private_binding_only_cannot_fill_RGB_role_identity(self):
        fs=[frame(s,s>=2) for s in range(17)]
        for v in ("primary","wrist"):
            fs[16]["observation_evidence"]["views"][v]["roles"]["target"]["local_rgb"]["texture_sufficient"]=False
        self.assertIsNone(self.derive(fs)["secondary_observer_goal"]["value"]);self.assertFalse(self.derive(fs)["secondary_observer_goal"]["mask"])
    def test_forward_interval_rejected_before_label(self):
        fs=[frame(s,s>=2) for s in range(17)]
        fs[16]["observation_evidence"]["actual_rgb_temporal_intervals"][0]["end_step"]=17
        with self.assertRaises(ValueError):self.derive(fs)
    def test_missing_own_noise_not_zero_filled(self):
        fs=[frame(s,s>=2) for s in range(17)]
        for a in fs[16]["observation_evidence"]["actual_rgb_temporal_intervals"]:a["same_candidate_repeat_noise"]["target"]=None
        self.assertIsNone(self.derive(fs)["secondary_observer_goal"]["value"])
    def test_unregistered_view_stays_unavailable(self):
        fs=[frame(s,s>=2) for s in range(17)]
        for v in ("primary","wrist"):fs[16]["observation_evidence"]["views"][v]["fresh_geometry_registered_to_actual_input"]=False
        self.assertIsNone(self.derive(fs)["secondary_observer_goal"]["value"])
    def test_after_observer_false_with_positive_RGB_cert_is_conflict_masked(self):
        x=self.derive(after=after_quality(False));self.assertIsNone(x["secondary_observer_goal"]["value"])
        self.assertFalse(x["secondary_observer_goal"]["mask"])
    def test_after_model_identity_time_query_source_hash_must_bind(self):
        for key,value in (("identity",SOURCE),("step",15),("frozen_attributor_model_sha256","f"*64),
                ("query_observation_sha256","a"*64),("full_source_prefix_runtime_sha256","f"*64),
                ("subject_maps_source",{}),("actual_attribution_source",{}),("observer_code_source",{}),("observer_model_source",{}),
                ("observer_input_lineage_sha256","f"*64),("actual_after_rgb_sources",{})):
            record=after_quality();record[key]=value
            with self.assertRaises(ValueError):self.derive(after=record)
    def test_explicit_quality_flag_must_match_frozen_actual_maps(self):
        with self.assertRaises(ValueError):self.derive(maps=after_maps(False))
        with self.assertRaises(ValueError):self.derive([frame(s) for s in range(17)],after=after_quality(False),maps=after_maps(True))
    def test_literal_complete_after_quality_required(self):
        for key,value in (("primary_reliable",1),("observer_evidence_available",None)):
            record=after_quality();record["quality"][key]=value
            with self.assertRaises(ValueError):self.derive(after=record,maps=after_maps())
    def test_after_true_missing_certificate_is_never_false(self):
        x=self.derive([frame(s) for s in range(17)]);self.assertIsNone(x["secondary_observer_goal"]["value"])
        self.assertFalse(x["secondary_observer_goal"]["mask"])
    def test_QC_boolean_numeric_failclosed(self):
        for key,value in (("physics_passed",None),("all_contact_atoms_match",1),("qpos_qvel_max_abs",.00101),("qpos_qvel_p95",float("nan"))):
            fs=[frame(s,s>=2) for s in range(17)];fs[16]["observation_evidence"]["same_candidate_repeat_qc"][key]=value
            with self.assertRaises(ValueError):self.derive(fs)
    def test_shorts_not_padded(self):
        with self.assertRaises(ValueError):self.derive([frame(s) for s in range(16)])
    def test_goal_cannot_move_to_first_up_or_unlock_gate(self):
        goal=labels.secondary_goal(QUALITY["quality"],predicted_cause="visual_occlusion");goal["deadline"]=3
        with self.assertRaises(ValueError):self.derive(goal=goal)


class QualityAndContrastTests(unittest.TestCase):
    def test_deployment_quality_accepts_actual_maps_only(self):
        maps=np.zeros((2,8,8),np.float32)
        q=labels.deployment_observer_quality(maps,image_quality={k:v for k,v in QUALITY["quality"].items() if k!="observer_evidence_available"},predicted_cause="visual_occlusion")
        self.assertFalse(q["quality"]["observer_evidence_available"]);self.assertFalse(q["actual_after_or_private_teacher_used"])
        labels.secondary_goal(q["quality"],predicted_cause=q["predicted_cause"])
    def test_private_key_or_after_four_maps_not_allowed_in_quality(self):
        q={k:v for k,v in QUALITY["quality"].items() if k!="observer_evidence_available"};q["GT_target_visible"]=True
        with self.assertRaises(ValueError):labels.deployment_observer_quality(np.zeros((2,8,8)),image_quality=q,predicted_cause="unknown")
        with self.assertRaises(ValueError):labels.deployment_observer_quality(np.zeros((4,8,8)),image_quality={},predicted_cause="unknown")
    def test_no_goal_on_normal_available_unknown_before(self):
        for available,cause in ((False,"normal"),(True,"visual_occlusion"),(None,"unknown")):
            q=copy.deepcopy(QUALITY["quality"]);q["observer_evidence_available"]=available
            with self.assertRaises(ValueError):labels.secondary_goal(q,predicted_cause=cause)
    def rows(self,values=(True,False,True,False)):
        return [dict(candidate_id=i,identity=dict(NATIVE,candidate_id=i),query_observation_sha256="a"*64,
            complete=True,goal_mask=True,goal_value=v) for i,v in enumerate(values)]
    def test_true_same_pool_positive_negative_contrast(self):
        x=labels.pool_contrast(self.rows());self.assertTrue(x["qualified_goal_contrast"]);self.assertFalse(x["stop_before_next_pool"])
        self.assertFalse(x["training_ready"])
    def test_allpos_allneg_and_missing_mask_stop_no_expand(self):
        for values in ((True,)*4,(False,)*4):
            self.assertTrue(labels.pool_contrast(self.rows(values))["stop_before_next_pool"])
        r=self.rows();r[2]["goal_mask"]=False;self.assertTrue(labels.pool_contrast(r)["stop_before_next_pool"])
    def test_same_pool_string_does_not_alias_other_dataset_state_or_block(self):
        for field,value in (("dataset","other"),("state",13),("observed_block_id","otherblock")):
            r=self.rows();r[2]["identity"][field]=value
            with self.assertRaises(ValueError):labels.pool_contrast(r)
    def test_different_query_hash_duplicate_CID_not_paired(self):
        r=self.rows();r[1]["query_observation_sha256"]="b"*64
        with self.assertRaises(ValueError):labels.pool_contrast(r)
        r=self.rows();r[1]["candidate_id"]=0
        with self.assertRaises(ValueError):labels.pool_contrast(r)
    def test_model_not_loaded_until_both_source_gates_pass(self):
        source=dict(proof=proof(),quality=QUALITY,goal=labels.secondary_goal(QUALITY["quality"],predicted_cause="visual_occlusion"))
        loader=mock.Mock(return_value="model")
        with self.assertRaises(ValueError):collector.load_model_after_source_gates([source],labels,loader)
        self.assertEqual(loader.call_count,0)
        bad=copy.deepcopy(source);bad["proof"]["restore_order_audit"]["full_runtime_bitwise_equal"]=False
        with self.assertRaises(ValueError):collector.load_model_after_source_gates([source,bad],labels,loader)
        self.assertEqual(loader.call_count,0)
        self.assertEqual(collector.load_model_after_source_gates([source,copy.deepcopy(source)],labels,loader),"model")
        self.assertEqual(loader.call_count,1)


class ExecutionTests(unittest.TestCase):
    def test_budget_caps_are_literal_no_hidden_retry(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temporary:
            b=collector.Budget(Path(temporary)/"budget.json")
            for _ in range(2):b.begin_prefix()
            for _ in range(32):b.prefix_step()
            for _ in range(8):b.query();b.begin_native(False);b.begin_native(True)
            for _ in range(256):b.native_step()
            for f in (b.begin_prefix,b.prefix_step,b.query,b.native_step):
                with self.assertRaises(ValueError):f()
            with self.assertRaises(ValueError):b.begin_native(False)
            with self.assertRaises(ValueError):b.begin_native(True)
            self.assertEqual(b.prefix_steps+b.native_steps,288)
    def test_explicit_launcher_LIBERO_frozen_library_threads(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temporary:
            root=Path(temporary);code=root/"research_runs"/collector.VERSION
            with self.assertRaises(ValueError):launcher.execution_environment(root,code)
            (root/".libero").mkdir();(root/".libero/config.yaml").write_text("existing: true\n")
            e=launcher.execution_environment(root,code)
            self.assertIn(str(root/"research_runs"/launcher.LIBRARY),e["PYTHONPATH"])
            self.assertEqual(e["OPENBLAS_NUM_THREADS"],"1");self.assertEqual(e["OMP_NUM_THREADS"],"1")
            self.assertEqual(e["LIBERO_CONFIG_PATH"],str(root/".libero"));self.assertEqual(collector.VERSION,launcher.VERSION)
    def test_protocol_exact_two_full_sources_and_frozen_seeds(self):
        p=json.loads((ROOT/"reports/native_recovery_block_boundary_protocol_20261007_v1.json").read_text())
        r=json.loads((ROOT/"reports/native_recovery_block_boundary_protocol_20261007_v2.json").read_text())
        self.assertEqual(collector.sha(ROOT/"reports/native_recovery_block_boundary_protocol_20261007_v1.json"),collector.PROTOCOL_V1_SHA)
        self.assertEqual([s["prefix_steps"] for s in p["native_reobservation"]["sources"]],[16,16])
        self.assertEqual(r["seed_schedule"]["task9_state10_visual_occlusion"],[10910064,10910065,10910066,10910067])
        self.assertEqual(p["native_reobservation"]["budget"]["max_new_action_steps_total"],288)
        self.assertEqual(p["release_native_cache_reuse_readonly"]["status"],"deferred_no_execution_in_this_protocol")
    def test_dependency_pins_and_sourceX_pins_all_sha(self):
        for digest in [*collector.DEPENDENCIES.values(),*[v for row in collector.SOURCE_X_PINS.values() for v in row.values()]]:
            self.assertEqual(len(digest),64);self.assertEqual(set(digest)-set("0123456789abcdef"),set())
    def test_single_callback_main_plus_one_QC_exact32_native_actions(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temporary:
            folder=Path(temporary)/"candidate";folder.mkdir();actions=np.zeros((16,7));actions[:,0]=np.arange(16)/20;actions[:,6]=1.
            np.save(folder/"planned_actions.npy",actions)
            for v in ("primary","wrist"):np.save(folder/("query_input_"+v+".npy"),np.zeros((256,256,3),np.uint8))
            np.save(folder/"query_input_proprio.npy",np.zeros(9))
            class Env:
                def __init__(self):self.s=0;self.called=[]
                def reset(self):self.s=0
                def raw(self):
                    return dict(agentview_image=np.full((256,256,3),self.s,np.uint8),robot0_eye_in_hand_image=np.full((256,256,3),self.s,np.uint8),
                        robot0_gripper_qpos=np.zeros(2),robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.zeros(4))
                def step(self,a):self.called.append(a);self.s+=1;return self.raw(),0.,False,{}
            env=Env();snapshot={"sim_state":np.zeros(2)}
            class Fork:
                last_restore_audit={"full_runtime_bitwise_equal":True}
                def restore_runtime_snapshot(self,env,snapshot):env.s=0;return env.raw()
                def capture_runtime_snapshot(self,env):return snapshot
            base=types.SimpleNamespace(install_scoped_segmentation_decoder=lambda e:None,image_check=lambda a,b:prior.rgb(np.array_equal(a,b)),
                sim_copy=lambda e,n:np.zeros(2),capture=lambda e,r,a,b,s,c:(dict(step=s,physics={k:False for k in prior.CONTACTS}),{v:np.zeros((256,256),np.int32) for v in ("primary","wrist")}))
            post=types.SimpleNamespace(_equal=lambda a,b:True)
            intervention=types.SimpleNamespace(InterventionEnv=types.SimpleNamespace(_mask_primary=lambda raw:(raw,None)))
            fake_robot=types.ModuleType("robosuite");fake_robot.macros=types.SimpleNamespace(IMAGE_CONVENTION="opencv")
            fake_robot_env=types.ModuleType("robosuite.environments.robot_env");fake_robot_env.IMAGE_CONVENTION_MAPPING={"opencv":1}
            hand_calls=[]
            def hand_reader(environment,binding,source,step):
                hand_calls.append((environment.s,step))
                return dict(role="offline_supervision_only",supervision_mask=False,held_negative_certificate_issued=False,
                    release_certificate_issued=False,whole_training_gate_pass=False)
            budget=collector.Budget(Path(temporary)/"budget.json");callback=collector.NativeSingleExecution(base,post,budget,hand_reader)
            with mock.patch.dict(sys.modules,{"robosuite":fake_robot,"robosuite.environments.robot_env":fake_robot_env}):
                first=callback(env,Fork(),intervention,{},folder,snapshot,"visual_occlusion")
                second=callback(env,Fork(),intervention,{},folder,snapshot,"visual_occlusion",save_segments=False)
                with self.assertRaises(ValueError):callback(env,Fork(),intervention,{},folder,snapshot,"visual_occlusion")
            self.assertEqual(len(env.called),32);self.assertEqual(budget.main,1);self.assertEqual(budget.qc,1);self.assertEqual(budget.native_steps,32)
            self.assertEqual(len(first[0]),17);self.assertEqual(len(second[0]),17)
            self.assertEqual(hand_calls,[(i,i) for i in range(17)]*2)
            for f in first[0]+second[0]:
                h=f["private_full_hand_metadata"]
                for field in ("supervision_mask","held_negative_certificate_issued","release_certificate_issued","whole_training_gate_pass"):
                    self.assertIs(h[field],False)
            np.testing.assert_array_equal(np.asarray(env.called[:16]),np.asarray(actions,np.float32))
            np.testing.assert_array_equal(np.asarray(env.called[16:]),np.asarray(actions,np.float32))


if __name__=="__main__":unittest.main()
