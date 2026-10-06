"""Finite postaction-prefix AUX branches; no native WAM queries or fitting.

Two fixed historical insufficient references, four fixed arms, one capture and
two QC executions per arm. Prefix reconstruction is separate preparation, not
an extra arm. All outcomes, including zero availability recovery, are retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np

VERSION="known_task_post_prefix_reobservation_auxiliary_20261007_v3"
PREVIOUS_VERSION="known_task_post_prefix_reobservation_auxiliary_20261007_v2"
REUSED_SNAPSHOT_SHA="33ea49036d62526fc286ec5736d0a4131f1705f6085ba8eb812db2176d1dc778"
REUSED_PROOF_SHA="9303378a70e5f3eebc007c4fe4c283d060e36c3d1c36147ac79443b8aa674c6e"
SOURCE="known_task_recovery_v9_20261006_v5_paired_recovery"
OBSERVATION="known_task_recovery_observation_audit_20261006_v5_motion_typed_support_full"
OLD_LABELS=OBSERVATION+"_labels_v3_perspective"
ADAPTER="known_task_recovery_observation_audit_20261006_v5_motion_typed_support"
BASE="known_task_temporal_recovery_supervision_20261006_v2_scoped_segmentation"
REFERENCE="candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"
ARMS=("hold","peek_left","peek_right","peek_up")
OBSERVATION_MANIFEST_SHA="a514c79aaa5e9e1018460a6ec81b0488fef1a21d96cb2e5dce1d550ad0d66f36"
DEPENDENCIES={
 f"research_runs/{ADAPTER}/collect_recovery_observation_evidence.py":"957e14e0e60bb4e2cae3db99311e2f2b4ee59d31b4fd49951b33c3f047a9d99e",
 f"research_runs/{ADAPTER}/observation_geometry.py":"f2eeca8e17d43f5939889a635a04252e4ee78bcd15632e9179a843e6843c7991",
 f"research_runs/{ADAPTER}/build_observable_recovery_labels.py":"7104ffa1faac715c63fc955d773896dac74efbcb836883bd356022334ca0c211",
 f"research_runs/{BASE}/replay_temporal_recovery_evidence.py":"3684243f669a27430b27f45673c138859548e160dff757de7269eb25128e8069",
 f"research_runs/{BASE}/temporal_recovery_teacher.py":"1c85d2b4667594d7a419501950bdcebbec134b6bc9fa2f4ea05ba476834d5053",
 f"research_runs/{REFERENCE}/collect_snapshot_fork.py":"e8588c5ba3718d6a9d1b1f64832bb66d9a835c3b935fd0ee8a450ac90e9c9070",
 f"research_runs/{REFERENCE}/base_collector_shift_calibrated.py":"681bea34e08b25ac4928c334b593d93921e44ef1067d985ee529998e3695a7c9",
}
BINDINGS=(
 dict(task=57,state=10,candidate_id=0,cause="unknown",split="train",prefix=4,
  snapshot_sha256="1469c584015586ed875f8d0647f991ee6c7568381fe177beb8e8273b3253df59",
  applied_sha256="3ab9b26d0b39d96105c1b168a50e11ec3a1a4c3b4b3fe896f298f897aa27f337",
  planned_sha256="cd6d0daeedb8e934911db266f0f8c531a06f738332da1b6b209db0f61699c492",
  primary_sha256="8dd585ba45b24e423df095f9a99078c927c40e2ce8268ff4c4b9b8d945a83cb9",
  wrist_sha256="8dac149fa1bdac6d98aba5e2316644391e75b01a34920841764a3f8dad4dd26c",
  old_sidecar_sha256="7c94f609fc9eed64a70303c4335a2ab2eaf2f643aa74a1a2d24d22967ca765f7"),
 dict(task=57,state=13,candidate_id=0,cause="unknown",split="val",prefix=2,
  snapshot_sha256="d0aad671a4423882f2649dbdd5e64e32f1d18c7b639b04f1584356b175029176",
  applied_sha256="6aa7d67f7d7b9741c2981186384c5497e72edd4d9fb405e8f23e5965268de624",
  planned_sha256="043d62546e3cdb5e271a9e59dda344372a7f534fcca5ad7e7a06e661aff0d258",
  primary_sha256="7216517dd56f2c55ad83078ff938c5290171a6f1147dcb593cf96eb55249e026",
  wrist_sha256="32af09432ff5e778c0e91577cd07ffaad8299c3b5882a4aa2d0d18ad654e43b8",
  old_sidecar_sha256="518c1d699df341f322f85c1f2870cb30adfd5c13ee8f404e85132be44cdf8fd2"),
)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1048576),b""):h.update(part)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def planned_probe(arm,original_applied,prefix):
    source=np.asarray(original_applied)
    if (arm not in ARMS or source.shape!=(16,7) or not np.isfinite(source).all() or
            type(prefix) is not int or not 2<=prefix<=16):
        raise ValueError("Predeclared arm and finite exact applied prefix required")
    # Continuity at the actual postaction start, never the old query first step.
    result=np.zeros((16,7),np.float32);result[:,6]=np.clip(source[prefix-1,6],-1.,1.)
    if arm=="peek_left":result[2:10,0]=-.2
    if arm=="peek_right":result[2:10,0]=.2
    if arm=="peek_up":result[2:10,2]=.2
    return result


def source_identity(entry):
    return dict(dataset=SOURCE,suite="libero90",task=entry["task"],state=entry["state"],
        candidate_id=entry["candidate_id"],observed_block_id=f"task57_state{entry['state']}_unknown/first_block")


def actual_observation(raw,intervention):
    observed,_=intervention.InterventionEnv._shift_primary_view(copy.deepcopy(raw))
    return dict(primary=np.flipud(np.asarray(observed["agentview_image"])),
        wrist=np.flipud(np.asarray(observed["robot0_eye_in_hand_image"])),
        proprio=np.r_[raw["robot0_gripper_qpos"],raw["robot0_eef_pos"],raw["robot0_eef_quat"]])


def prefix_only_before(teacher,repeat_teacher,first_arrays,repeat_arrays,prefix):
    """Use only historical steps <= prefix, not whole-window/future QC."""
    a=np.c_[first_arrays["qpos"][1:prefix+1],first_arrays["qvel"][1:prefix+1]]
    b=np.c_[repeat_arrays["qpos"][1:prefix+1],repeat_arrays["qvel"][1:prefix+1]]
    delta=np.abs(a-b)
    names=("left_finger_target_contact","right_finger_target_contact","target_support_contact","target_anchor_contact")
    if any(type(t["frames"][s]["physics"].get(k)) is not bool
           for t in (teacher,repeat_teacher) for s in range(1,prefix+1) for k in names):
        raise ValueError("Literal bool contact QC required at every historical prefix step")
    contacts=all(teacher["frames"][s]["physics"].get(k)==repeat_teacher["frames"][s]["physics"].get(k)
        for s in range(1,prefix+1) for k in names)
    if not np.isfinite(delta).all() or delta.max()>1e-3 or np.quantile(delta,.95)>1e-4 or not contacts:
        raise ValueError("Historical prefix-only physical repeat QC failed")
    before=copy.deepcopy(teacher["frames"][prefix])
    before["observation_evidence"]["same_candidate_repeat_qc"]=dict(physics_passed=True,
        all_contact_atoms_match=True,qpos_qvel_max_abs=float(delta.max()),qpos_qvel_p95=float(np.quantile(delta,.95)),
        used_original_steps=list(range(1,prefix+1)),future_source_steps_used=False)
    return before


def freeze(root,helpers,labeler):
    manifest=root/"outputs"/OBSERVATION/"SHA256SUMS.txt"
    if sha(manifest)!=OBSERVATION_MANIFEST_SHA:raise ValueError("Frozen observation SHA manifest changed")
    manifest_entries={line.split("  ",1)[1]:line.split("  ",1)[0] for line in manifest.read_text().splitlines()}
    report=json.loads((root/"outputs"/SOURCE/"completion_audit.json").read_text())
    if report.get("passed") is not True or report.get("candidate_first_blocks")!=160:
        raise ValueError("Complete frozen development source required")
    entries=[];touched={str(manifest.relative_to(root)):OBSERVATION_MANIFEST_SHA}
    for expected in BINDINGS:
        item=dict(expected);pool=f"task57_state{item['state']}_unknown";pd=root/"outputs"/SOURCE/"pools"/pool
        original=pd/"candidate_0";observed=root/"outputs"/OBSERVATION/"supervision_only"/pool/"candidate_0"
        oldpath=root/"outputs"/OLD_LABELS/"certificates"/pool/"candidate_0"/"observability_certificates.json"
        required={pd/"snapshot.pkl":item["snapshot_sha256"],original/"applied.npy":item["applied_sha256"],
            original/"planned_actions.npy":item["planned_sha256"],original/"primary.npy":item["primary_sha256"],
            original/"wrist.npy":item["wrist_sha256"],oldpath:item["old_sidecar_sha256"]}
        old=json.loads(oldpath.read_text());teacher=json.loads((observed/"temporal_teacher.json").read_text())
        if old["identity"]!=source_identity(item) or teacher["identity"]!=source_identity(item) or teacher["split"]!=item["split"]:
            raise ValueError("Historical candidate/split scope mismatch")
        required.update({original/"proprio.npy":old["evidence_sha256"]["actual_proprio"],
            observed/"temporal_teacher.json":old["evidence_sha256"]["private_projection"],
            observed/"replay_audit.json":old["evidence_sha256"]["replay_audit"],
            observed/"physical_teacher_arrays.npz":old["evidence_sha256"]["private_segmentation_arrays"]})
        for name in ("repeat_teacher.json","repeat_teacher_arrays.npz"):
            relative=(observed/name).relative_to(manifest.parent).as_posix()
            required[observed/name]=manifest_entries[relative]
        for path,digest in required.items():
            if sha(path)!=digest:raise ValueError("Frozen source changed: "+str(path))
            touched[str(path.relative_to(root))]=digest
        meta=json.loads((pd/"pool.json").read_text())
        if (meta["task"],meta["state"],meta["cause"],meta["split"])!=(57,item["state"],"unknown",item["split"]):
            raise ValueError("Pool metadata scope mismatch")
        touched[str((pd/"pool.json").relative_to(root))]=sha(pd/"pool.json")
        planned=np.load(original/"planned_actions.npy",allow_pickle=False);applied=np.load(original/"applied.npy",allow_pickle=False)
        if applied.shape!=(16,7) or not np.isfinite(applied).all():raise ValueError("Original exact applied action array required")
        repeat=json.loads((observed/"repeat_teacher.json").read_text())
        with np.load(observed/"physical_teacher_arrays.npz") as first,np.load(observed/"repeat_teacher_arrays.npz") as second:
            before=prefix_only_before(teacher,repeat,first,second,item["prefix"])
        if any(labeler.view_available(before,v,helpers) for v in ("primary","wrist")):
            raise ValueError("Fixed source no longer insufficient; no replacement permitted")
        item.update(pool_id=pool,source_directory=str(pd.relative_to(root)),original_directory=str(original.relative_to(root)),
            observation_directory=str(observed.relative_to(root)),old_sidecar=str(oldpath.relative_to(root)),
            before_available=False,actual_applied_prefix=applied[:item["prefix"]].tolist(),
            fixed_gripper_command=float(np.clip(applied[item["prefix"]-1,6],-1.,1.)),
            fixed_gripper_source=dict(array="original/applied.npy",sha256=item["applied_sha256"],
                zero_based_row=item["prefix"]-1,one_based_original_step=item["prefix"],column=6,
                literal_applied_value=float(applied[item["prefix"]-1,6]),controller_clip=[-1.,1.],
                selected_before_any_auxiliary_execution=True),
            planned_arms={a:planned_probe(a,applied,item["prefix"]).tolist() for a in ARMS})
        entries.append(item)
    return dict(schema="post_prefix_reobservation_protocol_v1",version=VERSION,entries=entries,source_hashes=touched,
        qualification_report_sha256="cc5049c43aa855a828a567ddce7814d638bc0dbccbc2eb6e7aa376b5ef1948a7",
        fixed_membership_not_selected_by_future_outcomes=True,source_prefix_reconstruction_passes=2,
        source_prefix_action_steps=6,auxiliary_main_executions=8,auxiliary_QC_executions=16,
        previous_verified_prefix_passes=1,new_prefix_reconstruction_passes=1,
        reuse_existing_verified_state10_prefix_snapshot=True,reused_snapshot_sha256=REUSED_SNAPSHOT_SHA,
        reused_prefix_proof_sha256=REUSED_PROOF_SHA,
        restore_qacc_warmstart_after_last_frozen_observation_forward=True,
        complete_runtime_bitwise_equality_assertion_retained=True,
        auxiliary_16step_execution_cap=24,auxiliary_action_step_cap=384,
        reset_restore_new_full_runtime_snapshot_before_every_arm=True,original_query_snapshot_is_not_branch_start=True,
        source_gripper_last_verified_applied_prefix_command_clip_minus1_plus1=True,all_arm_gripper_commands_equal=True,
        gripper_not_selected_from_old_query_first_command=True,gripper_not_selected_by_probe_outcome=True,
        pure_camera_only_sensing_claimed=False,gripper_and_target_state_change_possible=True,
        persistent_primary_corruption_unchanged=True,semantic_conflict_resolution_claimed=False,
        native_candidate_pool=False,model_queries=0,training_started=False,after_probe_observations_are_Y_only=True,
        zero_recovery_retained=True,no_outcome_based_retries_or_source_replacement=True,dependencies=DEPENDENCIES)


def reconstruct_prefix(env,fork,intervention,base,entry,root,destination):
    """Original snapshot -> EXACT original applied prefix -> NEW full snapshot."""
    pd=root/entry["source_directory"];original=root/entry["original_directory"];prefix=entry["prefix"]
    if sha(pd/"snapshot.pkl")!=entry["snapshot_sha256"]:raise ValueError("Trusted snapshot hash changed")
    with (pd/"snapshot.pkl").open("rb") as stream:snapshot=pickle.load(stream)
    if not np.array_equal(snapshot["sim_state"],np.load(original/"exact_start_state.npy",allow_pickle=False)):
        raise ValueError("Original source query snapshot binding differs")
    actions=np.load(original/"applied.npy",allow_pickle=False);images={v:np.load(original/(v+".npy"),allow_pickle=False) for v in ("primary","wrist")}
    proprio=np.load(original/"proprio.npy",allow_pickle=False)
    if any(x.shape!=(17,256,256,3) for x in images.values()) or proprio.shape!=(17,9):
        raise ValueError("Complete original actual image/proprio arrays required")
    env.reset();raw=fork.restore_runtime_snapshot(env,copy.deepcopy(snapshot))
    if not np.array_equal(env.get_sim_state(),snapshot["sim_state"]):raise ValueError("Original snapshot restore failed")
    checks=[]
    for step in range(1,prefix+1):
        raw,_,done,_=env.step(np.asarray(actions[step-1],np.float32).tolist());actual=actual_observation(raw,intervention)
        check=dict(step=step,primary=base.image_check(actual["primary"],images["primary"][step]),
            wrist=base.image_check(actual["wrist"],images["wrist"][step]),
            proprio_max_abs=float(np.abs(actual["proprio"]-proprio[step]).max()))
        check["passed"]=check["primary"]["passed"] and check["wrist"]["passed"] and check["proprio_max_abs"]<=1e-3
        checks.append(check)
        if done or not check["passed"]:
            dump(destination/"prefix_failure.json",dict(checks=checks,terminal_seen=bool(done),no_auto_retry=True))
            raise ValueError("Fixed original applied prefix replay failed; source retained, no replacement")
    new=fork.capture_runtime_snapshot(env)
    if set(new)!={"sim_state","timestep","sim_model","sim_data","robots"}:
        raise ValueError("Complete simulator/controller runtime snapshot required")
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/"post_prefix_runtime_snapshot.pkl").open("wb") as stream:pickle.dump(new,stream,protocol=pickle.HIGHEST_PROTOCOL)
    actual_before={v:images[v][prefix].copy() for v in images};actual_before["proprio"]=proprio[prefix].copy()
    for name,array in actual_before.items():np.save(destination/(name+"_actual_before.npy"),array)
    observed=root/entry["observation_directory"];old=json.loads((root/entry["old_sidecar"]).read_text())
    proof=dict(schema="verified_original_postaction_prefix_v1",identity=source_identity(entry),reference_step=prefix,
        source_hashes_verified=True,used_original_steps=list(range(1,prefix+1)),checks=checks,
        original_query_snapshot_used_as_new_start=False,actual_before_arrays_not_teacher_render=True,
        new_runtime_snapshot_sha256=sha(destination/"post_prefix_runtime_snapshot.pkl"),
        source_sha256=dict(original_snapshot=entry["snapshot_sha256"],applied=entry["applied_sha256"],
            primary=entry["primary_sha256"],wrist=entry["wrist_sha256"],proprio=old["evidence_sha256"]["actual_proprio"],
            teacher=old["evidence_sha256"]["private_projection"],old_sidecar=entry["old_sidecar_sha256"]))
    dump(destination/"verified_prefix_proof.json",proof)
    return new,actual_before,proof


def _equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b))


class PostPrefixFork:
    """New wrapper, not a modification to any historical fork file.

    The frozen restore writes solver state and then raw_observation.forward()
    overwrites it. Reapply both saved arrays AFTER that last forward, and keep
    the full runtime bitwise check (including controllers/model/all data).
    Main and both QC executions call this exact same wrapper.
    """
    def __init__(self,frozen):self.frozen=frozen;self.last_restore_audit=None
    def __getattr__(self,name):return getattr(self.frozen,name)
    def restore_runtime_snapshot(self,env,snapshot):
        raw=self.frozen.restore_runtime_snapshot(env,snapshot)
        inner=getattr(env,"env",env)
        for name in ("qacc","qacc_warmstart"):
            if name not in snapshot.get("sim_data",{}) or not hasattr(inner.sim.data,name):
                raise ValueError("Literal saved solver state and runtime arrays required")
            getattr(inner.sim.data,name)[:]=np.asarray(snapshot["sim_data"][name])
        # Inside this wrapper only: no extra forward/render/getobs after writes.
        # The normal env.step() forward/control/physics sequence is unchanged.
        current=self.frozen.capture_runtime_snapshot(env)
        self.last_restore_audit=dict(saved_solver_fields_reapplied_after_last_observation_forward=["qacc","qacc_warmstart"],
            full_runtime_bitwise_equal=_equal(snapshot,current),
            no_extra_forward_or_observation_call_inside_wrapper_after_solver_reapply=True)
        if not self.last_restore_audit["full_runtime_bitwise_equal"]:
            raise ValueError("Full post-prefix runtime restore still differs after solver-state order repair")
        return raw


def reuse_verified_prefix(entry,root,destination,helpers,labeler):
    """No re-execution: preserve the already verified four original steps."""
    previous=root/"outputs"/PREVIOUS_VERSION
    status=json.loads((previous/"status.json").read_text());failure=json.loads((previous/"failure.json").read_text())
    if (entry["state"]!=10 or entry["prefix"]!=4 or status.get("phase")!="failed" or status.get("captured_arms")!=0 or
            status.get("prefix_reconstructions")!=1 or failure.get("message")!="Full post-prefix runtime restore differs; no repair/relaxation"):
        raise ValueError("Only the fixed V2 verified state10 prefix, before any AUX action, may be reused")
    source=previous/"verified_prefixes"/entry["pool_id"];p=source/"post_prefix_runtime_snapshot.pkl"
    pp=source/"verified_prefix_proof.json"
    if sha(p)!=REUSED_SNAPSHOT_SHA or sha(pp)!=REUSED_PROOF_SHA:
        raise ValueError("Previously verified prefix snapshot/proof changed")
    proof=json.loads(pp.read_text());labeler.validate_prefix_proof(proof,source_identity(entry),4,helpers)
    if proof["new_runtime_snapshot_sha256"]!=REUSED_SNAPSHOT_SHA:raise ValueError("Reused snapshot does not bind proof")
    expected=dict(original_snapshot=entry["snapshot_sha256"],applied=entry["applied_sha256"],
        primary=entry["primary_sha256"],wrist=entry["wrist_sha256"],old_sidecar=entry["old_sidecar_sha256"])
    if any(proof["source_sha256"].get(k)!=v for k,v in expected.items()):raise ValueError("Reused source identity/hash mismatch")
    with p.open("rb") as stream:snapshot=pickle.load(stream)
    actual={v:np.load(root/entry["original_directory"]/(v+".npy"),allow_pickle=False)[4].copy() for v in ("primary","wrist","proprio")}
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(p,destination/p.name);shutil.copyfile(pp,destination/pp.name)
    for name,array in actual.items():np.save(destination/(name+"_actual_before.npy"),array)
    dump(destination/"reuse_audit.json",dict(source=str(source.relative_to(root)),snapshot_sha256=REUSED_SNAPSHOT_SHA,
        proof_sha256=REUSED_PROOF_SHA,original_prefix_not_reexecuted=True,prior_original_action_steps=4,
        AUX_action_steps_in_previous_attempt=0,shared_before_original_actual_arrays=True))
    return snapshot,actual,proof


def record_arm(env,fork,intervention,base,snapshot,actual_before,actions,destination):
    env.reset();raw=fork.restore_runtime_snapshot(env,copy.deepcopy(snapshot))
    current=fork.capture_runtime_snapshot(env)
    if not _equal(snapshot,current):raise ValueError("Full post-prefix runtime restore differs; no repair/relaxation")
    initial=actual_observation(raw,intervention)
    if np.abs(initial["proprio"]-actual_before["proprio"]).max()>1e-3:raise ValueError("Post-prefix proprio start differs")
    values={v:[actual_before[v].copy()] for v in ("primary","wrist","proprio")};length=0;terminal=False
    for action in actions:
        raw,_,done,_=env.step(action.tolist());actual=actual_observation(raw,intervention)
        for key in values:values[key].append(actual[key])
        length+=1;terminal=bool(done)
        if terminal:break
    destination.mkdir(parents=True,exist_ok=False)
    for name,data in values.items():np.save(destination/(name+".npy"),np.stack(data))
    for name in ("applied","requested","planned_actions"):np.save(destination/(name+".npy"),actions[:length])
    np.save(destination/"exact_start_state.npy",snapshot["sim_state"])
    dump(destination/"capture_audit.json",dict(valid_length=length,terminal_seen=terminal,no_padding=True,
        actual_local_frame0_is_verified_postaction_source_reference=True,cached_query_frame0=False,
        frame0_input_from_original_actual_RGB_not_teacher_render=True,
        restored_fresh_render_comparison_diagnostic_only={v:base.image_check(initial[v],actual_before[v]) for v in ("primary","wrist")},
        fresh_restored_render_never_replaces_actual_before=True,full_runtime_restored_exactly=True))
    if getattr(fork,"last_restore_audit",None) is not None:
        dump(destination/"restore_order_audit.json",fork.last_restore_audit)
    return length


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--collect",action="store_true")
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out.exists() or root not in out.parents or out.name!=VERSION or out.parent!=root/"outputs":
        raise ValueError("New exact-version output required; no overwrite, resume, or broad path")
    for name,digest in DEPENDENCIES.items():
        if sha(root/name)!=digest:raise ValueError("Frozen dependency changed: "+name)
    code=Path(__file__).resolve().parent
    labeler=load("post_prefix_reobservation_supervision",code/"post_prefix_reobservation_supervision.py")
    helpers=load("postprefix_frozen_certificate_helpers",root/"research_runs"/ADAPTER/"build_observable_recovery_labels.py")
    frozen_code=json.loads((code/"post_prefix_reobservation_code_sha256.json").read_text())
    if set(frozen_code)!={"post_prefix_reobservation_collect.py","post_prefix_reobservation_supervision.py","post_prefix_reobservation_launch.py"}:
        raise ValueError("Exact frozen collector/supervision code manifest required")
    for name,digest in frozen_code.items():
        if sha(code/name)!=digest:raise ValueError("New frozen code changed: "+name)
    protocol=freeze(root,helpers,labeler);protocol["deployed_code_sha256"]=frozen_code
    out.mkdir(parents=True);dump(out/"protocol.json",protocol)
    dump(out/"status.json",dict(phase="protocol_frozen",captured_arms=0,source_prefix_reconstructions=1,
        previous_verified_prefix_passes=1,new_prefix_reconstruction_passes=0,training_started=False))
    if not args.collect:return
    completed=[];env=None;started=time.time();prefix_count=1;attempts=[]
    try:
        adapter_dir=root/"research_runs"/ADAPTER;sys.path.insert(0,str(adapter_dir))
        load("observation_geometry",adapter_dir/"observation_geometry.py")
        load("temporal_recovery_teacher",root/"research_runs"/BASE/"temporal_recovery_teacher.py")
        base=load("postprefix_frozen_replay",root/"research_runs"/BASE/"replay_temporal_recovery_evidence.py")
        adapter=load("postprefix_frozen_adapter",adapter_dir/"collect_recovery_observation_evidence.py")
        historical_fork=load("postprefix_frozen_fork",root/"research_runs"/REFERENCE/"collect_snapshot_fork.py")
        fork=PostPrefixFork(historical_fork)
        intervention=load("postprefix_frozen_intervention",root/"research_runs"/REFERENCE/"base_collector_shift_calibrated.py")
        adapter.attach(base,root)
        from libero.libero import benchmark
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
        env,_=get_libero_env(benchmark.get_benchmark_dict()["libero_90"]().get_task(57),"cosmos",resolution=256)
        for entry in protocol["entries"]:
            prefix_dir=out/"verified_prefixes"/entry["pool_id"]
            if entry["state"]==10:
                snapshot,actual_before,proof=reuse_verified_prefix(entry,root,prefix_dir,helpers,labeler)
            else:
                prefix_count+=1
                # Historical prefix replay remains exactly the original restore.
                snapshot,actual_before,proof=reconstruct_prefix(env,historical_fork,intervention,base,entry,root,prefix_dir)
            observed=root/entry["observation_directory"]
            teacher=json.loads((observed/"temporal_teacher.json").read_text());repeat=json.loads((observed/"repeat_teacher.json").read_text())
            with np.load(observed/"physical_teacher_arrays.npz") as first,np.load(observed/"repeat_teacher_arrays.npz") as second:
                before=prefix_only_before(teacher,repeat,first,second,entry["prefix"])
            labeler.validate_prefix_proof(proof,source_identity(entry),entry["prefix"],helpers)
            if any(labeler.view_available(before,v,helpers) for v in ("primary","wrist")):
                raise ValueError("Fixed before evidence changed; no replacement or event fabrication")
            for arm,planned in entry["planned_arms"].items():
                dest=out/"auxiliary"/entry["pool_id"]/arm;actions=np.asarray(planned,np.float32)
                attempts.append(dict(pool_id=entry["pool_id"],arm=arm,main_attempt_reserved=1,QC_attempts_reserved=0))
                dump(out/"execution_budget.json",dict(prefix_reconstructions_attempted=prefix_count,branch_attempts=attempts,
                    main_cap=8,QC_cap=16,branch_execution_cap=24,no_retry=True))
                n=record_arm(env,fork,intervention,base,snapshot,actual_before,actions,dest)
                if n!=16:
                    completed.append(dict(pool_id=entry["pool_id"],arm=arm,complete=False,valid_length=n,retained=True))
                    raise ValueError("Fixed branch terminated early; retain, no padding/replacement")
                binding=base.bind(env,57);base.install_scoped_segmentation_decoder(env)
                attempts[-1]["QC_attempts_reserved"]=2
                dump(out/"execution_budget.json",dict(prefix_reconstructions_attempted=prefix_count,branch_attempts=attempts,
                    main_cap=8,QC_cap=16,branch_execution_cap=24,no_retry=True))
                result=base.replay(env,fork,intervention,binding,dest,snapshot,"unknown")
                # This wrapper makes exactly two QC executions on first use.
                frames,checks,segments,qpos,qvel=result
                frames[0]["observation_evidence"].update(cached_entry_is_diagnostic_only=False,
                    actual_post_prefix_source_reference=True,original_source_step=entry["prefix"],
                    legacy_extractor_entry_rule_does_not_define_new_before_semantics=True)
                checks[0].update(actual_post_prefix_reference=True,original_source_step=entry["prefix"],
                    legacy_entry_diagnostic_rule_not_used_for_before_label=True)
                identity=dict(dataset=VERSION,suite="libero90",task=57,state=entry["state"],candidate_id=ARMS.index(arm),
                    observed_block_id=f"{entry['pool_id']}/real_prefix_{entry['prefix']}/{arm}")
                labels=labeler.derive_post_prefix_availability(frames,before,source_identity=source_identity(entry),
                    auxiliary_identity=identity,arm=arm,original_reference_step=entry["prefix"],verified_prefix_evidence=proof,certificate_helpers=helpers)
                np.savez_compressed(dest/"private_teacher_arrays.npz",**{v:np.stack(x) for v,x in segments.items()},qpos=qpos,qvel=qvel)
                dump(dest/"actual_observation_evidence.json",dict(role="offline_supervision_only",identity=identity,frames=frames,
                    original_source_identity=source_identity(entry),local_frame0_is_actual_post_prefix_not_cached_query=True))
                dump(dest/"QC_replay_checks.json",dict(checks=checks,QC_executions=2,frozen_adapter_sha256=DEPENDENCIES[f"research_runs/{ADAPTER}/collect_recovery_observation_evidence.py"]))
                dump(dest/"observation_labels.json",labels)
                completed.append(dict(identity=identity,source_identity=source_identity(entry),split=entry["split"],arm=arm,
                    prefix=entry["prefix"],complete=True,directory=str(dest.relative_to(out)),availability_recovered=labels["observed_availability_recovered"],
                    conflict_resolved_mask=labels["cross_view_conflict_resolved"]["mask"],before_views=labels["before"]["views"],after_views=labels["after"]["views"]))
                dump(out/"status.json",dict(phase="collecting",captured_arms=len(completed),total_arms=8,prefix_reconstructions=prefix_count,
                    auxiliary_executions=len(completed)*3,elapsed_seconds=time.time()-started,training_started=False))
                print(json.dumps(completed[-1]),flush=True)
        for name,digest in protocol["source_hashes"].items():
            if sha(root/name)!=digest:raise ValueError("Original source changed: "+name)
        if len(completed)!=8 or prefix_count!=2:raise ValueError("Predeclared finite membership incomplete")
        counts={split:sum(x["availability_recovered"] for x in completed if x["split"]==split) for split in ("train","val")}
        audit=dict(passed=True,source_prefix_reconstruction_passes=2,source_prefix_action_steps=6,auxiliary_main_executions=8,
            previous_verified_prefix_passes=1,new_prefix_reconstruction_passes=1,
            auxiliary_QC_executions=16,auxiliary_16step_executions=24,availability_recovery_positive_arms=counts,
            conflict_resolved_certificates=0,original_sources_unchanged=True,source_reference_semantics="actual_postaction_prefix_not_query_snapshot",
            keep_zero_recovery_results=True,auxiliary_availability_silver_only=True,pure_camera_sensing_not_claimed=True,
            no_native_candidate_pool_benefit_claim=True,model_queries=0,training_started=False,ranking_training_ready=False)
        dump(out/"records.json",completed);dump(out/"completion_audit.json",audit);dump(out/"status.json",dict(phase="complete",**audit))
        files=[p for p in out.rglob("*") if p.is_file() and p.name not in ("status.json","SHA256SUMS.txt")]
        (out/"SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in sorted(files)),encoding="utf8")
    except Exception as error:
        dump(out/"failure.json",dict(error_type=type(error).__name__,message=str(error),retained_completed_arms=completed,
            prefix_reconstructions_attempted=prefix_count,branch_attempts=attempts,
            no_auto_retry_or_source_replacement=True,training_started=False))
        dump(out/"status.json",dict(phase="failed",captured_arms=len(completed),prefix_reconstructions=prefix_count,training_started=False))
        raise
    finally:
        if env is not None:env.close()


if __name__=="__main__":main()
