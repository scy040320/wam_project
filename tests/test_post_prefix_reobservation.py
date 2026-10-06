"""CPU fail-closed tests for the new bounded actual post-prefix AUX protocol."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wam_reranking import post_prefix_reobservation_supervision as labels


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


collector=module("postprefix_collector_test",ROOT/"scripts"/"post_prefix_reobservation_collect.py")
launcher=module("postprefix_launcher_test",ROOT/"scripts"/"post_prefix_reobservation_launch.py")
diagnostic=module("postprefix_restore_diagnostic_test",ROOT/"scripts"/"post_prefix_reobservation_restore_diagnostic.py")
helpers=module("postprefix_real_helpers_test",ROOT/"scripts"/"build_observable_recovery_labels.py")


def rgb(passed=True):
    return dict(passed=passed,mean_abs=0.,p95=0.,fraction_gt5=0.,psnr=100.,
                texture_sufficient=True,actual_texture_range=20.)


def track():
    return dict(verified=True,geometry_motion_support=True,evidence_kind="actual_RGB_geometry_correspondence",count=3,
        tracks=[dict(start_pixel_xy=[float(i),0.],end_pixel_xy=[float(i)+3.,1.],
            projected_end_pixel_xy=[float(i)+3.,1.],camera_only_end_pixel_xy=[float(i)+2.,1.],
            forward_backward_error_px=0.,projected_endpoint_error_px=0.) for i in range(3)])


def frame(step,available=False):
    evidence=dict(step=step,views={v:dict(global_rgb=rgb(),fresh_geometry_registered_to_actual_input=True,
        roles=dict(target=dict(rendered_pixels=40,local_rgb=rgb(available))),
        repeat_noise=dict(target=dict(centroid_noise_px=0.,rgb_repeat=rgb()))) for v in ("primary","wrist")},
        actual_rgb_temporal_intervals=[],same_candidate_repeat_qc=dict(physics_passed=True,
            all_contact_atoms_match=True,qpos_qvel_max_abs=0.,qpos_qvel_p95=0.))
    if step>=2:
        evidence["actual_rgb_temporal_intervals"]=[dict(start_step=1,end_step=step,view=v,
            tracks=dict(target=track()),same_candidate_repeat_noise=dict(target=0.)) for v in ("primary","wrist")]
    return dict(step=step,observation_evidence=evidence,physics=dict(target_absent=False))


IDENTITY=collector.source_identity(dict(task=57,state=13,candidate_id=0))
AUX=dict(dataset=collector.VERSION,suite="libero90",task=57,state=13,candidate_id=0,observed_block_id="actual_prefix2/hold")


def proof():
    return dict(schema="verified_original_postaction_prefix_v1",identity=copy.deepcopy(IDENTITY),reference_step=2,
        source_hashes_verified=True,used_original_steps=[1,2],original_query_snapshot_used_as_new_start=False,
        actual_before_arrays_not_teacher_render=True,new_runtime_snapshot_sha256="a"*64,
        source_sha256={k:"b"*64 for k in ("original_snapshot","applied","primary","wrist","proprio","teacher","old_sidecar")},
        checks=[dict(step=s,primary=rgb(),wrist=rgb(),proprio_max_abs=0.,passed=True) for s in (1,2)])


class AvailabilityTests(unittest.TestCase):
    def derive(self,frames=None,before=None,p=None):
        return labels.derive_post_prefix_availability(frames or [frame(s,s>=2) for s in range(17)],before or frame(2),
            source_identity=IDENTITY,auxiliary_identity=AUX,arm="hold",original_reference_step=2,
            verified_prefix_evidence=p or proof(),certificate_helpers=helpers)

    def test_recovery_uses_historical_before_and_actual_after_only(self):
        x=self.derive();self.assertFalse(x["before"]["value"]);self.assertTrue(x["after"]["value"])
        self.assertTrue(x["availability_recovery"]["value"]);self.assertTrue(x["availability_recovery"]["mask"])
        self.assertFalse(x["before"]["cached_query_frame"]);self.assertEqual(x["before"]["original_source_step"],2)
        self.assertFalse(x["new_before_ranking_X_created"]);self.assertTrue(x["after_execution_observations_are_Y_only"])

    def test_zero_recovery_is_valid_retained_negative_availability_not_absence(self):
        x=self.derive([frame(s) for s in range(17)])
        self.assertFalse(x["availability_recovery"]["value"]);self.assertTrue(x["availability_recovery"]["mask"])
        self.assertFalse(x["target_absence_inferred"])

    def test_conflict_unresolved_is_never_false_semantic_label(self):
        x=self.derive();self.assertIsNone(x["cross_view_conflict_resolved"]["value"])
        self.assertFalse(x["cross_view_conflict_resolved"]["mask"])
        self.assertFalse(x["ranking_training_ready"]);self.assertFalse(x["native_candidate_pool_benefit_claimed"])

    def test_new_local0_available_does_not_backfill_historical_before(self):
        frames=[frame(s,s>=2) for s in range(17)];frames[0]["observation_evidence"]["views"]["primary"]["roles"]["target"]["local_rgb"]=rgb()
        self.assertFalse(self.derive(frames)["before"]["value"])

    def test_available_source_rejected_no_replacement(self):
        with self.assertRaises(ValueError):self.derive(before=frame(2,True))

    def test_original_query_used_as_start_rejected(self):
        p=proof();p["original_query_snapshot_used_as_new_start"]=True
        with self.assertRaises(ValueError):self.derive(p=p)

    def test_teacher_render_used_as_before_rejected(self):
        p=proof();p["actual_before_arrays_not_teacher_render"]=False
        with self.assertRaises(ValueError):self.derive(p=p)

    def test_runtime_snapshot_missing_hash_rejected(self):
        p=proof();p.pop("new_runtime_snapshot_sha256")
        with self.assertRaises(ValueError):self.derive(p=p)

    def test_source_hash_missing_rejected(self):
        p=proof();p["source_sha256"].pop("applied")
        with self.assertRaises(ValueError):self.derive(p=p)

    def test_prefix_mapping_not_complete_rejected(self):
        p=proof();p["used_original_steps"]=[2]
        with self.assertRaises(ValueError):self.derive(p=p)

    def test_prefix_RGB_or_proprio_gate_not_relaxed(self):
        for path,value in (("proprio_max_abs",.0011),("passed",False)):
            p=proof();p["checks"][0][path]=value
            with self.assertRaises(ValueError):self.derive(p=p)
        p=proof();p["checks"][0]["primary"]["fraction_gt5"]=.0601
        with self.assertRaises(ValueError):self.derive(p=p)

    def test_no_padding_or_alignment_repair(self):
        with self.assertRaises(ValueError):self.derive([frame(s) for s in range(16)])
        fs=[frame(s) for s in range(17)];fs[3]["observation_evidence"]["step"]=4
        with self.assertRaises(ValueError):self.derive(fs)

    def test_frame1_actual_but_no_cached_query_track_identity(self):
        fs=[frame(s,True) for s in range(17)];x=self.derive(fs)
        self.assertFalse(x["frames"][1]["value"]);self.assertFalse(x["frames"][1]["cached_query_frame"])

    def test_duplicate_corner_missing_camera_only_fail_identity(self):
        for mode in ("duplicate","camera_missing"):
            f=frame(3,True)
            for a in f["observation_evidence"]["actual_rgb_temporal_intervals"]:
                t=a["tracks"]["target"]["tracks"]
                if mode=="duplicate":t[1]["start_pixel_xy"]=t[0]["start_pixel_xy"][:]
                else:t[0].pop("camera_only_end_pixel_xy")
            self.assertFalse(labels.view_available(f,"primary",helpers))

    def test_nonconstant_own_role_three_tracks_required(self):
        for texture,count in ((False,3),(True,2)):
            f=frame(3,True);d=f["observation_evidence"]["views"]["primary"]["roles"]["target"]
            d["local_rgb"]["texture_sufficient"]=texture
            a=f["observation_evidence"]["actual_rgb_temporal_intervals"][0]["tracks"]["target"]
            a["tracks"]=a["tracks"][:count];a["count"]=count
            self.assertFalse(labels.view_available(f,"primary",helpers))

    def test_own_repeat_noise_missing_or_nonfinite_not_zero_filled(self):
        for value in (None,-.1,float("nan")):
            f=frame(3,True);f["observation_evidence"]["actual_rgb_temporal_intervals"][0]["same_candidate_repeat_noise"]["target"]=value
            self.assertFalse(labels.view_available(f,"primary",helpers))

    def test_fresh_registration_and_role_pixels_frozen(self):
        f=frame(3,True);f["observation_evidence"]["views"]["primary"]["fresh_geometry_registered_to_actual_input"]=False
        self.assertFalse(labels.view_available(f,"primary",helpers))
        f=frame(3,True);f["observation_evidence"]["views"]["primary"]["roles"]["target"]["rendered_pixels"]=15
        self.assertFalse(labels.view_available(f,"primary",helpers))

    def test_all_intervals_audited_before_accepting_first(self):
        f=frame(3,True);a=copy.deepcopy(f["observation_evidence"]["actual_rgb_temporal_intervals"][0]);a["end_step"]=4
        f["observation_evidence"]["actual_rgb_temporal_intervals"].append(a)
        with self.assertRaises(ValueError):labels.view_available(f,"primary",helpers)

    def test_cached_source0_interval_rejected(self):
        f=frame(3,True);f["observation_evidence"]["actual_rgb_temporal_intervals"][0]["start_step"]=0
        with self.assertRaises(ValueError):labels.view_available(f,"primary",helpers)

    def test_same_arm_qc_contact_or_numeric_failure_not_relabelled(self):
        for key,value in (("all_contact_atoms_match",False),("qpos_qvel_max_abs",.00101),("qpos_qvel_p95",.000101)):
            fs=[frame(s,s>=2) for s in range(17)];fs[3]["observation_evidence"]["same_candidate_repeat_qc"][key]=value
            with self.assertRaises(ValueError):self.derive(fs)


CONTACTS=("left_finger_target_contact","right_finger_target_contact","target_support_contact","target_anchor_contact")


class ProtocolTests(unittest.TestCase):
    def test_launcher_environment_explicit_existing_config_paths_and_threads(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temp:
            root=Path(temp);code=root/"research_runs"/collector.VERSION
            with self.assertRaises(ValueError):launcher.execution_environment(root,code)
            (root/".libero").mkdir();(root/".libero"/"config.yaml").write_text("existing: configuration\n")
            env=launcher.execution_environment(root,code)
            self.assertEqual(env["LIBERO_CONFIG_PATH"],str(root/".libero"))
            self.assertIn(str(code),env["PYTHONPATH"]);self.assertIn(str(root),env["PYTHONPATH"])
            self.assertEqual(env["OPENBLAS_NUM_THREADS"],"1");self.assertEqual(env["OMP_NUM_THREADS"],"1")
            self.assertEqual(env["MUJOCO_GL"],"egl");self.assertEqual(launcher.VERSION,collector.VERSION)

    def test_exact_last_applied_prefix_gripper_continuity_all_arms(self):
        applied=np.zeros((16,7));applied[0,6]=-1.01;applied[3,6]=1.00125
        for arm in collector.ARMS:
            p=collector.planned_probe(arm,applied,4)
            self.assertEqual(p.shape,(16,7));self.assertTrue(np.all(p[:,6]==1.))
            self.assertTrue(np.all(p[:2,:6]==0));self.assertTrue(np.all(p[10:,:6]==0))
        applied[1,6]=-1.0085
        self.assertTrue(np.all(collector.planned_probe("hold",applied,2)[:,6]==-1.))

    def test_arm_displacements_fixed_before_outcome(self):
        applied=np.zeros((16,7));applied[3,6]=.5
        for arm,axis,sign in (("peek_left",0,-1),("peek_right",0,1),("peek_up",2,1)):
            p=collector.planned_probe(arm,applied,4);self.assertTrue(np.all(p[2:10,axis]==np.float32(sign*.2)))
            for other in set(range(6))-{axis}:self.assertTrue(np.all(p[:,other]==0))
        self.assertTrue(np.all(collector.planned_probe("hold",applied,4)[:,:6]==0))

    def test_invalid_plan_not_repaired(self):
        for source,prefix,arm in ((np.zeros((15,7)),2,"hold"),(np.full((16,7),np.nan),2,"hold"),
                                  (np.zeros((16,7)),1,"hold"),(np.zeros((16,7)),2,"new_arm")):
            with self.assertRaises(ValueError):collector.planned_probe(arm,source,prefix)

    def histories(self):
        t=dict(frames=[dict(step=s,physics={k:False for k in CONTACTS},observation_evidence={}) for s in range(17)])
        a=dict(qpos=np.zeros((17,2)),qvel=np.zeros((17,2)))
        return t,copy.deepcopy(t),a,copy.deepcopy(a)

    def test_prefix_QC_future_noise_contact_not_used(self):
        t,r,a,b=self.histories();b["qpos"][3:]=100.;r["frames"][3]["physics"][CONTACTS[0]]=True
        before=collector.prefix_only_before(t,r,a,b,2)
        self.assertEqual(before["step"],2);self.assertEqual(before["observation_evidence"]["same_candidate_repeat_qc"]["used_original_steps"],[1,2])
        self.assertFalse(before["observation_evidence"]["same_candidate_repeat_qc"]["future_source_steps_used"])

    def test_missing_or_int_contact_not_equal_none(self):
        for value in (None,0,"False"):
            t,r,a,b=self.histories();t["frames"][1]["physics"][CONTACTS[0]]=value;r["frames"][1]["physics"][CONTACTS[0]]=value
            with self.assertRaises(ValueError):collector.prefix_only_before(t,r,a,b,2)
        t,r,a,b=self.histories();t["frames"][1]["physics"].pop(CONTACTS[0]);r["frames"][1]["physics"].pop(CONTACTS[0])
        with self.assertRaises(ValueError):collector.prefix_only_before(t,r,a,b,2)

    def test_prefix_contact_mismatch_and_noise_fail_closed(self):
        t,r,a,b=self.histories();r["frames"][2]["physics"][CONTACTS[0]]=True
        with self.assertRaises(ValueError):collector.prefix_only_before(t,r,a,b,2)
        t,r,a,b=self.histories();b["qpos"][1,0]=.002
        with self.assertRaises(ValueError):collector.prefix_only_before(t,r,a,b,2)

    def test_full_runtime_equality_covers_controller_model_data(self):
        a=dict(sim_state=np.zeros(2),timestep=2,sim_model=dict(body_pos=np.zeros(3)),
            sim_data=dict(qacc_warmstart=np.zeros(2)),robots=[dict(controller=dict(goal_pos=np.zeros(3)))])
        self.assertTrue(collector._equal(a,copy.deepcopy(a)))
        for area in ("controller","warmstart","model"):
            b=copy.deepcopy(a)
            if area=="controller":b["robots"][0]["controller"]["goal_pos"][0]=1.
            elif area=="warmstart":b["sim_data"]["qacc_warmstart"][0]=1.
            else:b["sim_model"]["body_pos"][0]=1.
            self.assertFalse(collector._equal(a,b))

    def test_restore_difference_report_preserves_each_exact_field(self):
        a=dict(sim_state=np.zeros(2),robots=[dict(controller=dict(goal_pos=np.zeros(3)))],sim_data=dict(qacc=np.zeros(2)))
        b=copy.deepcopy(a);b["robots"][0]["controller"]["goal_pos"][1]=.01;b["sim_data"]["qacc"][0]=2.
        d=diagnostic.differences(a,b)
        self.assertEqual({x["path"] for x in d},{"runtime.robots.0.controller.goal_pos","runtime.sim_data.qacc"})
        self.assertTrue(all(x["finite"] for x in d));self.assertTrue(all(x["changed_values"]==1 for x in d))
        self.assertEqual(diagnostic.differences(a,copy.deepcopy(a)),[])


class FakeEnv:
    def __init__(self):self.index=0;self.actions=[];self.terminal_at=None
    def reset(self):self.index=0
    def get_sim_state(self):return np.array([self.index],float)
    def raw(self):
        return dict(agentview_image=np.full((256,256,3),self.index,np.uint8),
            robot0_eye_in_hand_image=np.full((256,256,3),self.index,np.uint8),
            robot0_gripper_qpos=np.zeros(2),robot0_eef_pos=np.array([self.index*.01,0.,0.]),robot0_eef_quat=np.array([0.,0.,0.,1.]))
    def step(self,action):
        self.actions.append(action);self.index+=1
        return self.raw(),0.,self.index==self.terminal_at,{}


def snapshot(env):
    return dict(sim_state=env.get_sim_state(),timestep=env.index,sim_model=dict(body_pos=np.zeros(3)),
        sim_data=dict(qacc_warmstart=np.zeros(2)),robots=[dict(controller=dict(goal_pos=np.ones(3)*env.index))])


def restore(env,value):env.index=int(value["timestep"]);return env.raw()


FORK=types.SimpleNamespace(capture_runtime_snapshot=snapshot,restore_runtime_snapshot=restore)
INTERVENTION=types.SimpleNamespace(InterventionEnv=types.SimpleNamespace(_shift_primary_view=lambda raw:(raw,{})))


def image_check(a,b):
    d=np.abs(a.astype(float)-b.astype(float));out=rgb(bool(d.max()<=2));out["mean_abs"]=float(d.mean());return out


BASE=types.SimpleNamespace(image_check=image_check)


class ActualStateTests(unittest.TestCase):
    def cache_fixture(self,break_other=False):
        env=FakeEnv();env.index=2
        env.sim=types.SimpleNamespace(data=types.SimpleNamespace(qacc=np.array([1.,2.]),qacc_warmstart=np.array([1.,2.]),ctrl=np.array([3.])))
        env.forward_calls=0
        def capture(e):
            s=snapshot(e);s["sim_data"]={k:getattr(e.sim.data,k).copy() for k in ("qacc","qacc_warmstart","ctrl")};return s
        def old_restore(e,s):
            e.index=s["timestep"];e.forward_calls+=3
            # Matches the frozen last raw_observation.forward cache overwrite.
            e.sim.data.qacc[:]=-100.;e.sim.data.qacc_warmstart[:]=-100.
            e.sim.data.ctrl[:]=999. if break_other else s["sim_data"]["ctrl"]
            return e.raw()
        return env,types.SimpleNamespace(capture_runtime_snapshot=capture,restore_runtime_snapshot=old_restore),capture(env)

    def test_final_solver_reapply_keeps_full_equality_after_forward_overwrite(self):
        env,old,saved=self.cache_fixture();fork=collector.PostPrefixFork(old)
        raw=fork.restore_runtime_snapshot(env,saved)
        self.assertEqual(env.forward_calls,3);self.assertEqual(raw["robot0_eef_pos"][0],.02)
        self.assertTrue(collector._equal(saved,fork.capture_runtime_snapshot(env)))
        np.testing.assert_array_equal(env.sim.data.qacc_warmstart,[1.,2.]);self.assertTrue(fork.last_restore_audit["full_runtime_bitwise_equal"])
        self.assertEqual(env.actions,[])

    def test_solver_repair_does_not_mask_other_runtime_change(self):
        env,old,saved=self.cache_fixture(True);fork=collector.PostPrefixFork(old)
        with self.assertRaises(ValueError):fork.restore_runtime_snapshot(env,saved)
        self.assertFalse(fork.last_restore_audit["full_runtime_bitwise_equal"]);self.assertEqual(env.actions,[])

    def test_missing_saved_solver_array_not_reconstructed(self):
        env,old,saved=self.cache_fixture();saved["sim_data"].pop("qacc_warmstart")
        with self.assertRaises(ValueError):collector.PostPrefixFork(old).restore_runtime_snapshot(env,saved)
        self.assertEqual(env.actions,[])

    def test_record_arm_restores_postprefix_and_keeps_actual_before_not_render(self):
        env=FakeEnv();env.index=2;s=snapshot(env);before=collector.actual_observation(env.raw(),INTERVENTION)
        before["primary"][:]=99
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temp:
            folder=Path(temp)/"hold";n=collector.record_arm(env,FORK,INTERVENTION,BASE,s,before,np.zeros((16,7),np.float32),folder)
            self.assertEqual(n,16);self.assertEqual(env.index,18)
            actual=np.load(folder/"primary.npy");self.assertTrue(np.all(actual[0]==99));self.assertTrue(np.all(actual[1]==3))
            audit=json.loads((folder/"capture_audit.json").read_text())
            self.assertFalse(audit["cached_query_frame0"]);self.assertTrue(audit["fresh_restored_render_never_replaces_actual_before"])

    def test_full_restore_failure_stops_before_any_action(self):
        env=FakeEnv();env.index=2;s=snapshot(env);s["sim_data"]["qacc_warmstart"][0]=1.
        before=collector.actual_observation(env.raw(),INTERVENTION)
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temp:
            with self.assertRaises(ValueError):collector.record_arm(env,FORK,INTERVENTION,BASE,s,before,np.zeros((16,7),np.float32),Path(temp)/"arm")
        self.assertEqual(env.actions,[])

    def test_early_terminal_retained_no_padding(self):
        env=FakeEnv();env.index=2;s=snapshot(env);before=collector.actual_observation(env.raw(),INTERVENTION);env.terminal_at=4
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temp:
            folder=Path(temp)/"hold";n=collector.record_arm(env,FORK,INTERVENTION,BASE,s,before,np.zeros((16,7),np.float32),folder)
            self.assertEqual(n,2);self.assertEqual(np.load(folder/"primary.npy").shape[0],3)
            self.assertTrue(json.loads((folder/"capture_audit.json").read_text())["terminal_seen"])

    def test_reconstruct_exact_applied_prefix_before_new_full_snapshot(self):
        env=FakeEnv();actions=np.zeros((16,7),np.float32);actions[:2,0]=[.125,-.25];actions[1,6]=-.875
        old=snapshot(env)
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temp:
            root=Path(temp);pd=root/"pool";original=pd/"candidate_0";original.mkdir(parents=True)
            with (pd/"snapshot.pkl").open("wb") as stream:pickle.dump(old,stream)
            np.save(original/"exact_start_state.npy",old["sim_state"]);np.save(original/"applied.npy",actions)
            arrays={v:[] for v in ("primary","wrist","proprio")}
            for step in range(17):
                env.index=step;a=collector.actual_observation(env.raw(),INTERVENTION)
                for key in arrays:arrays[key].append(a[key])
            for key in arrays:np.save(original/(key+".npy"),np.stack(arrays[key]))
            (root/"observed").mkdir();(root/"sidecar.json").write_text(json.dumps(dict(evidence_sha256=dict(actual_proprio="a"*64,private_projection="b"*64))))
            entry=dict(task=57,state=13,candidate_id=0,prefix=2,source_directory="pool",original_directory="pool/candidate_0",
                observation_directory="observed",old_sidecar="sidecar.json",snapshot_sha256=collector.sha(pd/"snapshot.pkl"),
                applied_sha256=collector.sha(original/"applied.npy"),primary_sha256=collector.sha(original/"primary.npy"),
                wrist_sha256=collector.sha(original/"wrist.npy"),old_sidecar_sha256=collector.sha(root/"sidecar.json"))
            env.actions=[];new,before,p=collector.reconstruct_prefix(env,FORK,INTERVENTION,BASE,entry,root,root/"newprefix")
            self.assertEqual(new["timestep"],2);self.assertEqual(len(env.actions),2)
            np.testing.assert_array_equal(np.asarray(env.actions),actions[:2])
            self.assertTrue(np.all(before["primary"]==2));self.assertEqual(p["used_original_steps"],[1,2])
            self.assertFalse(p["original_query_snapshot_used_as_new_start"])
            self.assertNotEqual(new["timestep"],old["timestep"])
            self.assertEqual(set(new),{"sim_state","timestep","sim_model","sim_data","robots"})

    def test_reuse_verified_old_prefix_zero_reexecution_source_hash_bound(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as temp:
            root=Path(temp);previous=root/"outputs"/collector.PREVIOUS_VERSION;previous.mkdir(parents=True)
            (previous/"status.json").write_text(json.dumps(dict(phase="failed",captured_arms=0,prefix_reconstructions=1)))
            (previous/"failure.json").write_text(json.dumps(dict(message="Full post-prefix runtime restore differs; no repair/relaxation")))
            source=previous/"verified_prefixes"/"task57_state10_unknown";source.mkdir(parents=True)
            entry=dict(task=57,state=10,candidate_id=0,prefix=4,pool_id="task57_state10_unknown",original_directory="original",
                **{k+"_sha256":"b"*64 for k in ("snapshot","applied","primary","wrist","old_sidecar")})
            original=root/"original";original.mkdir()
            for name in ("primary","wrist","proprio"):np.save(original/(name+".npy"),np.arange(17))
            s=snapshot(FakeEnv())
            with (source/"post_prefix_runtime_snapshot.pkl").open("wb") as stream:pickle.dump(s,stream)
            digest=collector.sha(source/"post_prefix_runtime_snapshot.pkl")
            p=proof();p.update(identity=collector.source_identity(entry),reference_step=4,used_original_steps=[1,2,3,4],
                checks=[dict(step=step,primary=rgb(),wrist=rgb(),proprio_max_abs=0.,passed=True) for step in range(1,5)],new_runtime_snapshot_sha256=digest)
            (source/"verified_prefix_proof.json").write_text(json.dumps(p));pdigest=collector.sha(source/"verified_prefix_proof.json")
            with mock.patch.object(collector,"REUSED_SNAPSHOT_SHA",digest),mock.patch.object(collector,"REUSED_PROOF_SHA",pdigest):
                reused,actual,verified=collector.reuse_verified_prefix(entry,root,root/"new",helpers,labels)
            self.assertTrue(collector._equal(reused,s));self.assertEqual(actual["primary"],4)
            self.assertEqual(collector.sha(source/"post_prefix_runtime_snapshot.pkl"),digest)
            self.assertEqual(verified["reference_step"],4)
            audit=json.loads((root/"new"/"reuse_audit.json").read_text());self.assertTrue(audit["original_prefix_not_reexecuted"])


if __name__=="__main__":unittest.main()
