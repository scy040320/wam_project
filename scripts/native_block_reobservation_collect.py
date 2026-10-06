"""Two finite, complete-boundary native K4 probes; source-audit first, no fit."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np

VERSION = "known_task_native_block_boundary_reobservation_20261007_v3_python310_compat"
REFERENCE = "candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"
ADAPTER = "known_task_recovery_observation_audit_20261006_v5_motion_typed_support"
BASE = "known_task_temporal_recovery_supervision_20261006_v2_scoped_segmentation"
POST = "known_task_post_prefix_reobservation_auxiliary_20261007_v3"
LEGACY = "candidate_reranking_known_task_closedloop_mechanism_20261004_v3_shared_source"
LIBRARY = "known_task_recovery_observation_input_overlay_20261006_v2_namespace_scoped/lib"
PROTOCOL_V1_SHA = "db810caf0b3df00e2a193c21570e91ea172c1758b06fa3d8c23abe76e9e9f3b5"
SOURCE_X_PINS={
    10:{"planned_actions.npy":"29ac503b74d792b21da331a383468a691248016b5e4736dd13703d1f6e8d4fd4",
        "requested.npy":"74621db5c45f3519ac8d8392837d528c5e931e1824164746cf9b3b741024f708",
        "predicted_primary.npy":"654c16e3ee1cc4d32d185b3727ad6ef39e1318d457df33f2e658cc48a32ae368",
        "predicted_wrist.npy":"9b5b5fbda19a2bc76819956034bdc3f94d94ecefa75822b35734c0eb03f79045"},
    13:{"planned_actions.npy":"c2669f40fdc12ec18af8cc4f3facc56771221c3eb6fe7f714600f6d96776943f",
        "requested.npy":"51e592a645c496bf54121de484e87fcd4d552bb52818848af9b949baf9119c2a",
        "predicted_primary.npy":"7387dea89648dd4d5cf3c815fa51ddb5ef9fe3d4e6f88c3918b3b5198599dcbc",
        "predicted_wrist.npy":"9873f829415a850f2f6ecc477e6d02e50e2a5baeab780b4df46e1201816c2346"}
}
DEPENDENCIES = {
    "research_runs/candidate_reranking_task16_multiblock_closedloop_20261004_v3_scoped_precision/frozen_online.py":"db4772c548a84d56ef8458b02bff093fcd8c893f8ddbe2e365096d59e09c8049",
    "outputs/candidate_reranking_d18_main20_joint_training_20261004_v8_supervision_consistent/subject_anchor_attributor_v18i.pt":"68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec",
    "research_runs/candidate_reranking_d18_main20_joint_training_20261004_v8_supervision_consistent/materialized_training.py":"0e787344ce3c1b62f724a884540d291ecdb97502800129243273776cf28b8826",
    "research_runs/candidate_reranking_d18_v4_training_resume_20260927_v2/d18_v3_lib.py":"27a3ad0c04257b10d29996a5e50f2304f2e89d978b4ffa3aaa1a3090ee21baf3",
    "research_runs/candidate_reranking_d18_v8_decoupled_balanced_factors_20260927_v1/feature_lib.py":"ba031a1c6e7216a8d86e45e850ed7dedab1dcccadb7cbab49df911475f745da4",
    "outputs/candidate_reranking_d18_v2_training_20260927_v1/feature_names_v2.json":"63c59db408e87dea6b06211e5f08c153aa4f8397c2510ca906f45601a9f310c3",
    "outputs/candidate_reranking_d18_main20_joint_training_20261004_v8_supervision_consistent/effective_dataset_raw.npz":"038ab4b4ec143a8e3d6741301bff8fa222d5dab3d9f78fe5d222c10f58f461bb",
    "outputs/candidate_reranking_d18_main20_joint_training_20261004_v8_supervision_consistent/validation_predictions.npz":"5ee7cb9ce4481520d74d9ed2bee3bba37d8190f62fb8e4f78432ccd144f83f0e",
    "outputs/candidate_reranking_d18_main20_joint_training_20261004_v8_supervision_consistent/training_manifest.json":"73405c8b5b26ea5340099a06bcad403c27787222761f30e2fb89dc35963e89f8",
    f"research_runs/{LIBRARY}/wam_reranking/direct_recovery.py":"289ef1c73d70fca2c1d3cc56f74f84037b341c4b2308de64944c60ce0e7bf12b",
    f"research_runs/{LIBRARY}/wam_reranking/recovery_contract.py":"26c13d935ecc5f77c94695a3af700e845a19d8270e6ba5bd77ce0bacba78dbc3",
    f"research_runs/{LIBRARY}/wam_reranking/observed_recovery.py":"c6213b301b98fc90fadb3dfe0fe90f4e2d40b84f04c9b422dbce4072720b3654",
    f"research_runs/{LIBRARY}/wam_reranking/evidence_residual.py":"51cb2fba7e266435ba6cc5d5c74e0c266e7db1b42c2413ae5d279aef7ec3f5c2",
    f"research_runs/{LIBRARY}/wam_reranking/candidate_effects.py":"55b11811e36269ac08927d53fe66f260a84ba0fed401766ffd25b633dd4e1c38",
    f"research_runs/{LIBRARY}/wam_reranking/target_localization.py":"72a2c025ec84536e1fbd52e50d4195e58d9bc7eed9f121dad71f6fe536a658ca",
    f"research_runs/{LIBRARY}/wam_reranking/relation_evidence.py":"3cf583f75a34308442bfadf65511740bf9a7be20433b5ea233ab312b058820b3",
    f"research_runs/{ADAPTER}/collect_recovery_observation_evidence.py":"957e14e0e60bb4e2cae3db99311e2f2b4ee59d31b4fd49951b33c3f047a9d99e",
    f"research_runs/{ADAPTER}/observation_geometry.py":"f2eeca8e17d43f5939889a635a04252e4ee78bcd15632e9179a843e6843c7991",
    f"research_runs/{ADAPTER}/build_observable_recovery_labels.py":"7104ffa1faac715c63fc955d773896dac74efbcb836883bd356022334ca0c211",
    f"research_runs/{BASE}/replay_temporal_recovery_evidence.py":"3684243f669a27430b27f45673c138859548e160dff757de7269eb25128e8069",
    f"research_runs/{BASE}/temporal_recovery_teacher.py":"1c85d2b4667594d7a419501950bdcebbec134b6bc9fa2f4ea05ba476834d5053",
    f"research_runs/{REFERENCE}/collect_snapshot_fork.py":"e8588c5ba3718d6a9d1b1f64832bb66d9a835c3b935fd0ee8a450ac90e9c9070",
    f"research_runs/{REFERENCE}/base_collector_shift_calibrated.py":"681bea34e08b25ac4928c334b593d93921e44ef1067d985ee529998e3695a7c9",
    f"research_runs/{REFERENCE}/candidate_outcome_rollout.py":"82d482ca0f943429aeac5472bf6e55efdcc804812b3b02050d08750ed0b78312",
    f"research_runs/{REFERENCE}/prepare_d21_training_bundle_v7.py":"5e1d7078e2a804c9dc2f864e072f15bbc548604ce536e9beede065ec3acac51a",
    f"research_runs/{POST}/post_prefix_reobservation_collect.py":"7238a4200e70132f9ec7b64a175802620cdde1001ea24e3c5bb2cdc530354e6d",
    f"research_runs/{POST}/post_prefix_reobservation_supervision.py":"6e96942373da528600696b841944e0457dbb0c000ed94c4df202682eddcebbc9",
    f"research_runs/{LEGACY}/run_mechanism.py":"25cd9cfc8aacab782aa94807babf4ccd96bebb16e4575cd50af3d11047bf98b8",
    "research_runs/candidate_reranking_task16_multiblock_closedloop_20261004_v3_scoped_precision/run_closedloop.py":"6517b31df8151ac16ee9305422d3513956db1f54bdb5d46a9329a3d1d4ec55da"
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1048576),b""): h.update(part)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf8")


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def source_identity(entry,source):
    return dict(dataset=source,suite="libero90",task=entry["task"],state=entry["state"],candidate_id=0,
        observed_block_id=entry["observed_block_id"])


def actual_observation(raw,intervention):
    observed,_=intervention.InterventionEnv._mask_primary(copy.deepcopy(raw))
    return dict(primary=np.flipud(np.asarray(observed["agentview_image"])),
        wrist=np.flipud(np.asarray(observed["robot0_eye_in_hand_image"])),
        proprio=np.r_[raw["robot0_gripper_qpos"],raw["robot0_eef_pos"],raw["robot0_eef_quat"]])


def query_observation(actual):
    if (np.asarray(actual["primary"]).shape!=(256,256,3) or np.asarray(actual["wrist"]).shape!=(256,256,3)
            or np.asarray(actual["proprio"]).shape!=(9,) or any(not np.isfinite(actual[k]).all() for k in actual)):
        raise ValueError("Actual before dual256RGB and9D proprio required")
    return dict(primary_image=np.asarray(actual["primary"]).copy(),wrist_image=np.asarray(actual["wrist"]).copy(),
        proprio=np.asarray(actual["proprio"]).copy())


class Budget:
    def __init__(self,path):
        self.path=path;self.prefix_passes=0;self.prefix_steps=0;self.main=0;self.qc=0;self.native_steps=0;self.queries=0
    def save(self):
        dump(self.path,dict(prefix_passes=self.prefix_passes,prefix_action_steps=self.prefix_steps,
            native_main_executions=self.main,native_QC_executions=self.qc,native_action_steps=self.native_steps,
            Cosmos_queries=self.queries,caps=dict(prefix_passes=2,prefix_steps=32,main=8,QC=8,native_steps=256,queries=8),no_retry=True))
    def begin_prefix(self):
        if self.prefix_passes>=2:raise ValueError("Fixed source reconstruction cap")
        self.prefix_passes+=1;self.save()
    def prefix_step(self):
        if self.prefix_steps>=32:raise ValueError("Fixed32 prefix actions cap")
        self.prefix_steps+=1;self.save()
    def begin_native(self,repeat):
        key="qc" if repeat else "main"
        if getattr(self,key)>=8:raise ValueError("Fixed native execution cap")
        setattr(self,key,getattr(self,key)+1);self.save()
    def native_step(self):
        if self.native_steps>=256:raise ValueError("Fixed native256 action cap")
        self.native_steps+=1;self.save()
    def query(self):
        if self.queries>=8:raise ValueError("Fixed8 Cosmos query cap")
        self.queries+=1;self.save()


def verify_source(root,entry,protocol,helpers,prefix_helpers,post):
    pool=root/entry["pool_path"];folder=pool/"candidate_0"
    tracked={pool/"pool.json":entry["pool_json_sha256"],pool/"snapshot.pkl":entry["snapshot_sha256"],
        folder/"applied.npy":entry["candidate_applied_sha256"],folder/"primary.npy":entry["actual_primary_sha256"],
        folder/"wrist.npy":entry["actual_wrist_sha256"]}
    tracked.update({folder/name:digest for name,digest in SOURCE_X_PINS[entry["state"]].items()})
    labels=root/"outputs"/protocol["old_labels"]/"certificates"/pool.name/"candidate_0"/"observability_certificates.json"
    tracked[labels]=entry["old_sidecar_sha256"]
    for path,digest in tracked.items():
        if sha(path)!=digest:raise ValueError("Frozen source changed: "+str(path))
    old=json.loads(labels.read_text());observed=root/"outputs"/protocol["observation_dataset"]/"supervision_only"/pool.name/"candidate_0"
    manifest=root/"outputs"/protocol["observation_dataset"]/"SHA256SUMS.txt"
    if sha(manifest)!=protocol["observation_source_manifest_sha256"]:raise ValueError("Frozen full observation manifest changed")
    manifest_entries={line.split("  ",1)[1]:line.split("  ",1)[0] for line in manifest.read_text().splitlines()}
    tracked[manifest]=protocol["observation_source_manifest_sha256"]
    hashes=old["evidence_sha256"]
    for path,key in ((folder/"proprio.npy","actual_proprio"),(observed/"temporal_teacher.json","private_projection"),
            (observed/"replay_audit.json","replay_audit"),(observed/"physical_teacher_arrays.npz","private_segmentation_arrays")):
        if sha(path)!=hashes[key]:raise ValueError("Source old certificate evidence changed")
        tracked[path]=hashes[key]
    for name in ("repeat_teacher.json","repeat_teacher_arrays.npz"):
        path=observed/name;digest=manifest_entries[path.relative_to(manifest.parent).as_posix()]
        if sha(path)!=digest:raise ValueError("Frozen prefix repeat evidence changed")
        tracked[path]=digest
    teacher=json.loads((observed/"temporal_teacher.json").read_text());repeat=json.loads((observed/"repeat_teacher.json").read_text())
    with np.load(observed/"physical_teacher_arrays.npz",allow_pickle=False) as first,np.load(observed/"repeat_teacher_arrays.npz",allow_pickle=False) as second:
        before=post.prefix_only_before(teacher,repeat,first,second,16)
    identity=source_identity(entry,protocol["source_dataset"])
    metadata=json.loads((pool/"pool.json").read_text())
    if (teacher.get("identity")!=identity or old.get("identity")!=identity or teacher.get("split")!=entry["split"] or
            (metadata.get("task"),metadata.get("state"),metadata.get("cause"),metadata.get("split"),metadata.get("k"))!=
            (9,entry["state"],"visual_occlusion",entry["split"],4)):
        raise ValueError("Full source identity/split/task binding mismatch")
    # Target binding + exact observed source role, not a model input.
    return dict(entry=entry,pool=pool,folder=folder,old=old,before=before,identity=identity,
        source_hashes={str(p.relative_to(root)):d for p,d in tracked.items()},observed=observed)


def reconstruct_source(env,historical_fork,post,intervention,base,source,dest,budget):
    entry=source["entry"];folder=source["folder"]
    with (source["pool"]/"snapshot.pkl").open("rb") as stream:original=pickle.load(stream)
    actions=np.load(folder/"applied.npy",allow_pickle=False)
    arrays={v:np.load(folder/(v+".npy"),allow_pickle=False) for v in ("primary","wrist","proprio")}
    if actions.shape!=(16,7) or any(not np.isfinite(x).all() for x in [actions,*arrays.values()]):raise ValueError("Complete finite actual prefix required")
    if arrays["primary"].shape!=(17,256,256,3) or arrays["wrist"].shape!=arrays["primary"].shape or arrays["proprio"].shape!=(17,9):raise ValueError("17 real original prefix observations required")
    budget.begin_prefix();env.reset();raw=historical_fork.restore_runtime_snapshot(env,copy.deepcopy(original));checks=[]
    for step in range(1,17):
        budget.prefix_step();raw,_,done,_=env.step(np.asarray(actions[step-1],np.float32).tolist())
        actual=actual_observation(raw,intervention)
        check=dict(step=step,primary=base.image_check(actual["primary"],arrays["primary"][step]),
            wrist=base.image_check(actual["wrist"],arrays["wrist"][step]),
            proprio_max_abs=float(np.abs(actual["proprio"]-arrays["proprio"][step]).max()))
        check["passed"]=check["primary"]["passed"] and check["wrist"]["passed"] and check["proprio_max_abs"]<=1e-3
        checks.append(check)
        if done or not check["passed"]:
            dump(dest/"prefix_failure.json",dict(checks=checks,terminal_seen=bool(done),no_retry=True))
            raise ValueError("Predeclared full prefix failed; no source replacement")
    snapshot=historical_fork.capture_runtime_snapshot(env)
    if set(snapshot)!={"sim_state","timestep","sim_model","sim_data","robots"}:raise ValueError("Complete runtime snapshot required")
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/"post_block_runtime_snapshot.pkl").open("xb") as stream:pickle.dump(snapshot,stream,protocol=pickle.HIGHEST_PROTOCOL)
    exact=post.PostPrefixFork(historical_fork);raw=exact.restore_runtime_snapshot(env,copy.deepcopy(snapshot));fresh=actual_observation(raw,intervention)
    restore_checks={v:base.image_check(fresh[v],arrays[v][16]) for v in ("primary","wrist")}
    if not all(x["passed"] for x in restore_checks.values()) or np.abs(fresh["proprio"]-arrays["proprio"][16]).max()>1e-3:
        raise ValueError("Same actual post-block source restore image/proprio gate failed")
    # New actual observation, not an old cached query substituted as X.
    for v,x in fresh.items():np.save(dest/(v+"_actual_before.npy"),x)
    old=source["old"]["evidence_sha256"]
    proof=dict(schema="verified_original_postaction_prefix_v1",identity=source["identity"],reference_step=16,
        source_hashes_verified=True,used_original_steps=list(range(1,17)),checks=checks,
        original_query_snapshot_used_as_new_start=False,actual_before_arrays_not_teacher_render=True,
        new_runtime_snapshot_sha256=sha(dest/"post_block_runtime_snapshot.pkl"),
        source_sha256=dict(original_snapshot=entry["snapshot_sha256"],applied=entry["candidate_applied_sha256"],
            primary=entry["actual_primary_sha256"],wrist=entry["actual_wrist_sha256"],proprio=old["actual_proprio"],
            teacher=old["private_projection"],old_sidecar=entry["old_sidecar_sha256"]),
        full_native_block_boundary=True,complete_source_steps=16,restore_order_audit=exact.last_restore_audit,
        restored_actual_source_RGB_checks=restore_checks)
    dump(dest/"verified_prefix_proof.json",proof)
    return snapshot,fresh,proof


class NativeSingleExecution:
    """Adapter's first call IS main, second IS single QC; no extra replay."""
    def __init__(self,base,post,budget,hand_reader=None):
        self.base=base;self.post=post;self.budget=budget;self.calls={};self.hand_reader=hand_reader
    def __call__(self,env,fork,intervention,binding,folder,snapshot,cause,*,save_segments=True):
        folder=Path(folder);key=str(folder);count=self.calls.get(key,0)
        if count>=2:raise ValueError("Exactly one main and one QC, no third execution")
        self.calls[key]=count+1;repeat=count==1;self.budget.begin_native(repeat)
        if cause!="visual_occlusion":raise ValueError("Fixed persistent source occlusion required")
        actions=np.load(folder/"planned_actions.npy",allow_pickle=False)
        if actions.shape!=(16,7) or not np.isfinite(actions).all():raise ValueError("Original complete native action plan required")
        env.reset();self.base.install_scoped_segmentation_decoder(env)
        raw=fork.restore_runtime_snapshot(env,copy.deepcopy(snapshot))
        if not self.post._equal(snapshot,fork.capture_runtime_snapshot(env)):raise ValueError("Full source runtime differs")
        from robosuite import macros
        from robosuite.environments.robot_env import IMAGE_CONVENTION_MAPPING
        convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        frames=[];checks=[];segments={v:[] for v in ("primary","wrist")};qpos=[];qvel=[]
        observations={v:[] for v in ("primary","wrist","proprio")};applied=[];done=False
        query_actual={v:np.load(folder/("query_input_"+v+".npy"),allow_pickle=False) for v in observations}
        expected={v:np.load(folder/(v+".npy"),allow_pickle=False) for v in observations} if repeat else None
        for step in range(17):
            if step:
                self.budget.native_step();action=np.asarray(actions[step-1],np.float32)
                raw,_,done,_=env.step(action.tolist());applied.append(action.copy())
            actual=actual_observation(raw,intervention)
            if repeat:
                if step>=len(expected["proprio"]):raise ValueError("QC length differs from main")
                check=dict(step=step,primary=self.base.image_check(actual["primary"],expected["primary"][step]),
                    wrist=self.base.image_check(actual["wrist"],expected["wrist"][step]),
                    proprio_max_abs=float(np.abs(actual["proprio"]-expected["proprio"][step]).max()))
            else:
                reference=query_actual if step==0 else actual
                check=dict(step=step,primary=self.base.image_check(actual["primary"],reference["primary"]),
                    wrist=self.base.image_check(actual["wrist"],reference["wrist"]),
                    proprio_max_abs=float(np.abs(actual["proprio"]-reference["proprio"]).max()))
            check["passed"]=check["primary"]["passed"] and check["wrist"]["passed"] and check["proprio_max_abs"]<=1e-3
            if not check["passed"]:raise ValueError("Same native candidate actual RGB/proprio QC failed")
            frame,seg=self.base.capture(env,copy.deepcopy(raw),actual,binding,step,convention)
            if self.hand_reader is not None:
                hand=self.hand_reader(env,binding,folder,step)
                if (hand.get("role")!="offline_supervision_only" or hand.get("supervision_mask") is not False or
                        any(hand.get(k) is not False for k in ("held_negative_certificate_issued","release_certificate_issued","whole_training_gate_pass"))):
                    raise ValueError("Supplemental fullhand metadata is private Y only; all masks/certificates false")
                frame["private_full_hand_metadata"]=hand
            atoms=("left_finger_target_contact","right_finger_target_contact","target_support_contact","target_anchor_contact")
            if any(type(frame["physics"].get(k)) is not bool for k in atoms):raise ValueError("Literal contact QC booleans required")
            frames.append(frame);checks.append(check)
            for v in observations:observations[v].append(actual[v].copy())
            for v in segments:segments[v].append(seg[v])
            qpos.append(self.base.sim_copy(env,"qpos"));qvel.append(self.base.sim_copy(env,"qvel"))
            if not np.isfinite(qpos[-1]).all() or not np.isfinite(qvel[-1]).all():raise ValueError("Finite actual simulator QC arrays required")
            if done:break
        if repeat and len(frames)!=len(expected["proprio"]):raise ValueError("Same-candidate QC terminal length mismatch")
        dest=folder/"QC_repeat" if repeat else folder
        dest.mkdir(parents=True,exist_ok=True)
        for v,x in observations.items():np.save(dest/(v+".npy"),np.stack(x))
        for name in ("requested","applied"):np.save(dest/(name+".npy"),np.asarray(applied,np.float32))
        dump(dest/"capture_audit.json",dict(valid_length=len(applied),complete=len(applied)==16,terminal_seen=bool(done),
            actual_native_capture=True,QC_repeat=repeat,actual_local0_is_post_full_block=True,cached_query_frame0=False,
            same_full_restore_wrapper=True,restore_order_audit=fork.last_restore_audit,no_padding=True))
        return frames,checks,segments,np.stack(qpos),np.stack(qvel)


def load_model_after_source_gates(sources,labels,loader):
    """The model loader is never called on a partial or failed source gate."""
    if not isinstance(sources,list) or len(sources)!=2:raise ValueError("Both fixed source gates required before Cosmos load")
    for source in sources:
        proof=source.get("proof",{})
        if (proof.get("full_native_block_boundary") is not True or proof.get("complete_source_steps")!=16 or
                proof.get("restore_order_audit",{}).get("full_runtime_bitwise_equal") is not True):
            raise ValueError("Complete actual-boundary exact runtime proof required")
        if source.get("goal")!=labels.secondary_goal(source["quality"]["quality"],predicted_cause=source["quality"]["predicted_cause"]):
            raise ValueError("Before-frozen deployable goal mismatch")
    return loader()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--collect",action="store_true")
    args=parser.parse_args();root=args.root.resolve();code=Path(__file__).resolve().parent;out=root/"outputs"/VERSION
    if out.exists() or code!=root/"research_runs"/VERSION:raise ValueError("Exact new version; no overwrite/resume")
    manifest=json.loads((code/"native_block_reobservation_code_sha256.json").read_text())
    if set(manifest)!={"native_block_reobservation_collect.py","native_block_reobservation_supervision.py","native_block_reobservation_launch.py",
            "native_recovery_block_boundary_protocol_20261007_v1.json","native_recovery_block_boundary_protocol_20261007_v2.json","hand_contact_supervision.py"}:
        raise ValueError("Exact finite code/protocol manifest required")
    for name,digest in manifest.items():
        if sha(code/name)!=digest:raise ValueError("Frozen new code changed")
    if sha(code/"native_recovery_block_boundary_protocol_20261007_v1.json")!=PROTOCOL_V1_SHA:raise ValueError("V1 source protocol changed")
    original=json.loads((code/"native_recovery_block_boundary_protocol_20261007_v1.json").read_text())
    revision=json.loads((code/"native_recovery_block_boundary_protocol_20261007_v2.json").read_text())
    if revision["version"]!=VERSION or revision["base_protocol_sha256"]!=PROTOCOL_V1_SHA:raise ValueError("Exact V2 frozen semantic revision required")
    for name,digest in DEPENDENCIES.items():
        if sha(root/name)!=digest:raise ValueError("Frozen dependency changed: "+name)
    snapshotdir=root/"outputs"/(REFERENCE+"_snapshot");pins=original["frozen_Cosmos"]
    for name,key in (("Cosmos-Policy-LIBERO-Predict2-2B.pt","checkpoint_sha256"),("libero_t5_embeddings.pkl","T5_cache_sha256"),("libero_dataset_statistics.json","dataset_statistics_sha256")):
        if sha(snapshotdir/name)!=pins[key]:raise ValueError("Frozen model/embedding/statistics changed")
    out.mkdir();dump(out/"protocol.json",dict(base=original,revision=revision,code_sha256=manifest,dependencies=DEPENDENCIES))
    dump(out/"status.json",dict(phase="protocol_frozen",training_started=False,Cosmos_loaded=False))
    if not args.collect:return
    started=time.time();budget=Budget(out/"execution_budget.json");env=None;completed=[];sources=[]
    try:
        sys.path.insert(0,str(root/"research_runs"/REFERENCE));sys.path.insert(0,str(root/"research_runs"/ADAPTER))
        helpers=load("native_frozen_certificate_helpers",root/"research_runs"/ADAPTER/"build_observable_recovery_labels.py")
        prefix_helpers=load("native_prefix_certificate_helpers",root/"research_runs"/POST/"post_prefix_reobservation_supervision.py")
        post=load("native_frozen_restore_helpers",root/"research_runs"/POST/"post_prefix_reobservation_collect.py")
        labels=load("native_secondary_observer_labels",code/"native_block_reobservation_supervision.py")
        hand=load("native_Yonly_full_hand_metadata",code/"hand_contact_supervision.py")
        historical_fork=load("native_historical_fork",root/"research_runs"/REFERENCE/"collect_snapshot_fork.py")
        intervention=load("native_frozen_intervention",root/"research_runs"/REFERENCE/"base_collector_shift_calibrated.py")
        load("observation_geometry",root/"research_runs"/ADAPTER/"observation_geometry.py")
        load("temporal_recovery_teacher",root/"research_runs"/BASE/"temporal_recovery_teacher.py")
        base=load("native_frozen_measurements",root/"research_runs"/BASE/"replay_temporal_recovery_evidence.py")
        adapter=load("native_frozen_rgb_adapter",root/"research_runs"/ADAPTER/"collect_recovery_observation_evidence.py")
        from libero.libero import benchmark
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
        env,language=get_libero_env(benchmark.get_benchmark_dict()["libero_90"]().get_task(9),"cosmos",resolution=256)
        if language!="put the black bowl on the plate":raise ValueError("Frozen exact task language mismatch")
        source_protocol=original["native_reobservation"]
        for entry in source_protocol["sources"]:
            source=verify_source(root,entry,source_protocol,helpers,prefix_helpers,post)
            prefix_helpers._identity(source["identity"]);labels.check_target_binding(source["before"])
            if any(prefix_helpers.view_available(source["before"],v,helpers) for v in ("primary","wrist")):raise ValueError("Predeclared before no longer insufficient")
            dest=out/"sources"/source["pool"].name
            source["snapshot"],source["actual_before"],source["proof"]=reconstruct_source(env,historical_fork,post,intervention,base,source,dest,budget)
            prefix_helpers.validate_prefix_proof(source["proof"],source["identity"],16,helpers)
            source["dest"]=dest;sources.append(source)
            dump(out/"status.json",dict(phase="physical_source_audit",verified_sources=len(sources),Cosmos_loaded=False,training_started=False))
        # Deployment attributor/CLIPSeg may load here; Cosmos weights may not.
        sys.path.insert(0,str(root/"research_runs"/LEGACY))
        legacy=load("native_frozen_known_runtime",root/"research_runs"/LEGACY/"run_mechanism.py")
        bundle=load("native_frozen_task_bindings",root/"research_runs"/REFERENCE/"prepare_d21_training_bundle_v7.py")
        attributor=legacy.KnownAttributor();relation=bundle.FROZEN_RELATIONS[(9,0)]
        attributor.set_context(9,labels.TARGET,relation)
        if not attributor.validation_replay_exact:raise ValueError("Frozen deployment attributor replay failed")
        if attributor.dev.UNKNOWN_THRESHOLD!=.64 or attributor.dev.FACTOR_THRESHOLD!=.5:
            raise ValueError("Frozen actual V8 unknown/factor thresholds changed")
        dump(out/"attributor_provenance.json",dict(model_path=str(legacy.CK.relative_to(root)),model_sha256=legacy.CK_SHA,
            producer="candidate_reranking_d18_main20_joint_training_20261004_v8_supervision_consistent",
            artifact_filename_v18i_does_not_change_V8_provenance=True,input_dim=2298,unknown_threshold=.64,factor_threshold=.5,
            feature_encoder_model=attributor.base.MODEL_NAME,validation_replay_exact=True,
            same_frozen_V8_deployment_attributor=True,GT_cause_used_for_need=False))
        from PIL import Image
        for source in sources:
            folder=source["folder"];arrays={v:np.load(folder/(v+".npy"),allow_pickle=False) for v in ("primary","wrist","proprio")}
            result=dict(actions=np.load(folder/"planned_actions.npy",allow_pickle=False),future_image_predictions=dict(
                future_image=np.load(folder/"predicted_primary.npy",allow_pickle=False),future_wrist_image=np.load(folder/"predicted_wrist.npy",allow_pickle=False)))
            ep=legacy.parent.saved_evidence(source["dest"]/"deployable_prefix",query_observation({v:x[0] for v,x in arrays.items()}),result,
                query_observation(source["actual_before"]),np.load(folder/"requested.npy",allow_pickle=False),np.load(folder/"applied.npy",allow_pickle=False),3)
            meta=json.loads((ep/"episode.json").read_text());meta.update(active_subgoal="put_black_bowl_on_plate",task_language=language,benchmark_suite="libero_90");dump(ep/"episode.json",meta)
            record=attributor.infer(ep);dump(source["dest"]/"before_predicted_attribution.json",record)
            attr=bundle.attribution_output(record,f"libero90_task9_block3")
            with legacy.frozen_precision():
                maps=attributor.localizer.relevance([Image.fromarray(source["actual_before"][v].astype(np.uint8)) for v in ("primary","wrist")],relation.subject)
            quality=labels.deployment_observer_quality(maps,image_quality=asdict(attr.evidence_quality),predicted_cause=attr.projected_cause.value)
            np.save(source["dest"]/"before_actual_target_maps.npy",np.asarray(maps))
            dump(source["dest"]/"before_deployable_quality.json",quality)
            source["quality"]=quality;source["goal"]=labels.secondary_goal(quality["quality"],predicted_cause=quality["predicted_cause"])
            dump(source["dest"]/"frozen_secondary_goal.json",source["goal"])
        dump(out/"source_audit.json",dict(passed=True,verified_sources=2,all_source_gates_before_Cosmos_load=True,
            before_quality_deployable_no_teacher=True,old_source_hashes={k:v for s in sources for k,v in s["source_hashes"].items()},training_started=False))
        dump(out/"status.json",dict(phase="loading_Cosmos_once",verified_sources=2,Cosmos_loaded=False,training_started=False))
        cfg=intervention.build_config(snapshotdir,1909);cfg.task_suite_name="libero_90";cfg.num_queries_best_of_n=1
        parent=legacy.parent;stats=parent.load_dataset_stats(cfg.dataset_stats_path);parent.init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        model,conf=load_model_after_source_gates(sources,labels,lambda:parent.get_model(cfg))
        if conf.dataloader_train.dataset.chunk_size!=16:raise ValueError("Actual Cosmos nativeH16 required")
        query_helper=load("native_frozen_query",root/"research_runs"/REFERENCE/"candidate_outcome_rollout.py")
        def hand_reader(environment,bound,folder,step):
            state=next(s["entry"]["state"] for s in sources if folder.parent.name==s["pool"].name+"_post_native16")
            ident=dict(dataset=VERSION,suite="libero90",task=9,state=state,candidate_id=int(folder.name.split("_")[-1]),
                observed_block_id=folder.parent.name+"/native_first_block")
            value=hand.capture_hand_contact_metadata(environment,bound["target_body"],source_identity=ident)
            value.update(relative_step=step,no_additional_step_reset_or_render=True)
            return value
        single=NativeSingleExecution(base,post,budget,hand_reader=hand_reader);base.replay=single;adapter.attach(base,root)
        fork=post.PostPrefixFork(historical_fork);binding=base.bind(env,9)
        if binding["target"]!=labels.TARGET:raise ValueError("Actual task target mismatch")
        from wam_reranking.candidate_effects import localize_candidate_visual_evidence,parse_candidate_effect
        from wam_reranking.evidence_residual import typed_gate_effect
        from wam_reranking.recovery_contract import VISUAL_EVIDENCE_KEYS,planned_command_timing
        from wam_reranking.direct_recovery import RawCandidateFeatures
        stopped=None
        for source in sources:
            pool_id=source["pool"].name+"_post_native16";pool_dir=out/"native_pools"/pool_id;pool_dir.mkdir(parents=True,exist_ok=False)
            obs=query_observation(source["actual_before"]);obs_hash=intervention.obs_hash(obs)
            candidates=[]
            for cid,seed in enumerate(revision["seed_schedule"][source["pool"].name]):
                budget.query();dump(out/"status.json",dict(phase="native_query",pool_id=pool_id,candidate_id=cid,queries=budget.queries,Cosmos_loaded=True,training_started=False))
                result,seconds=query_helper.query(cfg,model,stats,obs,language,seed)
                actions=np.asarray(result["actions"]);future=result["future_image_predictions"]
                if actions.shape!=(16,7) or not np.isfinite(actions).all() or not np.isfinite(float(result["value_prediction"])):raise ValueError("Complete finite native Cosmos result required")
                dest=pool_dir/f"candidate_{cid}";dest.mkdir()
                np.save(dest/"planned_actions.npy",actions)
                for v,a in source["actual_before"].items():np.save(dest/("query_input_"+v+".npy"),a)
                for v,k in (("primary","future_image"),("wrist","future_wrist_image")):
                    predicted=np.asarray(future[k])
                    if predicted.ndim!=3 or predicted.shape[2]!=3 or not np.isfinite(predicted).all():raise ValueError("Same-query finite native RGB prediction required")
                    np.save(dest/("predicted_"+v+".npy"),predicted)
                with legacy.frozen_precision():
                    visual=localize_candidate_visual_evidence(localizer=attributor.localizer,
                        current_primary=Image.fromarray(obs["primary_image"].astype(np.uint8)),current_wrist=Image.fromarray(obs["wrist_image"].astype(np.uint8)),
                        predicted_primary=Image.fromarray(np.asarray(future["future_image"]).astype(np.uint8)),predicted_wrist=Image.fromarray(np.asarray(future["future_wrist_image"]).astype(np.uint8)),
                        target_prompt=relation.subject,anchor_prompt=relation.anchor,relation=relation.relation)
                effect=typed_gate_effect(parse_candidate_effect(cid,actions,visual_evidence=visual,close_when_negative=False),relation.relation)
                raw=np.concatenate((planned_command_timing(actions),np.asarray([item for key in VISUAL_EVIDENCE_KEYS for item in (float(np.clip(effect.evidence[key],-1,1)) if key in effect.evidence else 0.,float(key in effect.evidence))],dtype=float)))
                raw46=RawCandidateFeatures(raw,True);np.save(dest/"canonical_raw46.npy",raw46.values)
                dump(dest/"candidate_X.json",dict(schema="preexecution_native_actual_before_own_prediction_v1",query_observation_sha256=obs_hash,
                    before_source=str(source["dest"].relative_to(root)),before_quality=source["quality"],stage=effect.stage.value,typed_relation=relation.relation,
                    planned_action_sha256=sha(dest/"planned_actions.npy"),predicted_primary_sha256=sha(dest/"predicted_primary.npy"),predicted_wrist_sha256=sha(dest/"predicted_wrist.npy"),
                    raw46_fingerprint=raw46.fingerprint(),raw46_schema=raw46.schema,candidate_visual=asdict(visual),frozen_secondary_goal=source["goal"],
                    value=float(result["value_prediction"]),seed=seed,seconds=seconds,actual_after_or_teacher_in_X=False,native_stage_not_rewritten=True,created_before_any_candidate_execution=True))
                candidates.append(dict(cid=cid,dest=dest,effect=effect))
            rows=[]
            for candidate in candidates:
                cid=candidate["cid"];dest=candidate["dest"]
                dump(out/"status.json",dict(phase="native_main_and_single_QC",pool_id=pool_id,candidate_id=cid,queries=budget.queries,main=budget.main,QC=budget.qc,Cosmos_loaded=True,training_started=False))
                frames,checks,segments,qpos,qvel=base.replay(env,fork,intervention,binding,dest,source["snapshot"],"visual_occlusion")
                n=len(frames)-1;native_id=dict(dataset=VERSION,suite="libero90",task=9,state=source["entry"]["state"],candidate_id=cid,observed_block_id=pool_id+"/native_first_block")
                dump(dest/"actual_observation_evidence.json",dict(role="offline_supervision_only",identity=native_id,frames=frames,main_plus_single_QC=True))
                np.savez_compressed(dest/"private_teacher_arrays.npz",**{v:np.stack(x) for v,x in segments.items()},qpos=qpos,qvel=qvel)
                if n==16:
                    actual_end={v:np.load(dest/(v+".npy"),allow_pickle=False)[16] for v in ("primary","wrist","proprio")}
                    actual_result=dict(actions=np.load(dest/"planned_actions.npy",allow_pickle=False),future_image_predictions=dict(
                        future_image=np.load(dest/"predicted_primary.npy",allow_pickle=False),future_wrist_image=np.load(dest/"predicted_wrist.npy",allow_pickle=False)))
                    after_ep=parent.saved_evidence(dest/"after_deployment_observer",obs,actual_result,query_observation(actual_end),
                        np.load(dest/"requested.npy",allow_pickle=False),np.load(dest/"applied.npy",allow_pickle=False),4)
                    after_meta=json.loads((after_ep/"episode.json").read_text());after_meta.update(active_subgoal="put_black_bowl_on_plate",task_language=language,benchmark_suite="libero_90");dump(after_ep/"episode.json",after_meta)
                    after_attr=bundle.attribution_output(attributor.infer(after_ep),"libero90_task9_block4")
                    with legacy.frozen_precision():
                        after_maps=attributor.localizer.relevance([Image.fromarray(actual_end[v].astype(np.uint8)) for v in ("primary","wrist")],relation.subject)
                    after_quality=labels.deployment_observer_quality(after_maps,image_quality=asdict(after_attr.evidence_quality),predicted_cause=after_attr.projected_cause.value,boundary="after")
                    after_quality.update(identity=native_id,step=16,relative_step=16,role="offline_supervision_only",
                        actual_array_sources={v:dict(path=str((dest/(v+".npy")).relative_to(root)),sha256=sha(dest/(v+".npy")),frame_index=16) for v in ("primary","wrist")},
                        frozen_attributor_model_sha256=legacy.CK_SHA,all_future_output_Y_only=True,query_observation_sha256=obs_hash,
                        full_source_prefix_runtime_sha256=source["proof"]["new_runtime_snapshot_sha256"],
                        actual_attribution_source=dict(path=str((after_ep/"attribution.json").relative_to(root)),sha256=sha(after_ep/"attribution.json")))
                    after_quality["observer_code_source"]=dict(path=f"research_runs/{LIBRARY}/wam_reranking/observed_recovery.py",sha256=labels.OBSERVER_CODE_SHA)
                    after_quality["observer_model_source"]=dict(path=str(legacy.CK.relative_to(root)),sha256=legacy.CK_SHA)
                    after_quality["actual_after_rgb_sources"]={}
                    for v in ("primary","wrist"):
                        endpoint=dest/("actual_after_"+v+"_rgb.npy");np.save(endpoint,actual_end[v])
                        after_quality["actual_after_rgb_sources"][v]=dict(path=str(endpoint.relative_to(root)),sha256=sha(endpoint))
                    np.save(dest/"actual_after_target_maps.npy",np.asarray(after_maps))
                    after_quality["subject_maps_source"]=dict(path=str((dest/"actual_after_target_maps.npy").relative_to(root)),sha256=sha(dest/"actual_after_target_maps.npy"))
                    observer_binding={k:after_quality[k] for k in ("identity","step","actual_after_rgb_sources","subject_maps_source","observer_code_source","observer_model_source")}
                    after_quality["observer_input_lineage_sha256"]=hashlib.sha256(json.dumps(observer_binding,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
                    dump(dest/"actual_after_deployment_quality.json",after_quality)
                    after_source=dict(path=str((dest/"actual_after_deployment_quality.json").relative_to(root)),sha256=sha(dest/"actual_after_deployment_quality.json"))
                    derived=labels.derive_native_goal(frames,source["before"],source_identity=source["identity"],native_identity=native_id,
                        verified_prefix_evidence=source["proof"],helpers=helpers,prefix_helpers=prefix_helpers,native_stage=candidate["effect"].stage.value,
                        frozen_goal=source["goal"],before_deployment_quality=source["quality"],actual_after_deployment_quality=after_quality,
                        actual_after_quality_source=after_source,native_query_observation_sha256=obs_hash,actual_after_subject_maps=np.asarray(after_maps))
                    dump(dest/"observation_labels.json",derived);goal=derived["secondary_observer_goal"]
                else:
                    goal=dict(value=None,mask=False);dump(dest/"short_retained.json",dict(valid_length=n,endpoint16_unobserved=True,no_padding_or_replacement=True))
                rows.append(dict(candidate_id=cid,identity=native_id,query_observation_sha256=obs_hash,complete=n==16,
                    native_stage=candidate["effect"].stage.value,goal_mask=goal["mask"],goal_value=goal["value"],split=source["entry"]["split"],retained=True))
            audit=labels.pool_contrast(rows);dump(pool_dir/"native_contrast_audit.json",audit);dump(pool_dir/"pool.json",dict(pool_id=pool_id,k=4,query_observation_sha256=obs_hash,candidates=rows))
            completed.append(dict(pool_id=pool_id,rows=rows,audit=audit));dump(out/"records.json",completed)
            if audit["stop_before_next_pool"]:stopped="no_qualified_within_pool_secondary_goal_contrast";break
        for source in sources:
            for path,digest in source["source_hashes"].items():
                if sha(root/path)!=digest:raise ValueError("Original source mutated")
        result=dict(passed=True,completed_pools=len(completed),unattempted_pools=2-len(completed),stop_reason=stopped,
            source_prefix_actions=budget.prefix_steps,Cosmos_queries=budget.queries,native_main=budget.main,native_QC=budget.qc,native_actions=budget.native_steps,
            elapsed_seconds=time.time()-started,no_fit=True,training_started=False,whole_contract_pass_claimed=False,ranking_training_ready=False,original_sources_unchanged=True)
        dump(out/"completion_audit.json",result);dump(out/"status.json",dict(phase="complete",**result))
    except Exception as error:
        dump(out/"failure.json",dict(error_type=type(error).__name__,message=str(error),retained_completed_pools=completed,
            no_retry_source_or_seed_change=True,training_started=False));dump(out/"status.json",dict(phase="failed",error=str(error),queries=budget.queries,
                main=budget.main,QC=budget.qc,prefix_actions=budget.prefix_steps,native_actions=budget.native_steps,training_started=False));raise
    finally:
        if env is not None:env.close()
        files=sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("status.json","SHA256SUMS.txt"))
        (out/"SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in files),encoding="utf8")


if __name__=="__main__":main()
