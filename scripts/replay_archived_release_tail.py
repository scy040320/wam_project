"""Two immutable terminal tails: real lengths only, offline AUX, no policy call.

The original 797 complete blocks and old release pilot are never modified.
Future actual observations are supervision-only, never pre-execution inputs.
The unchanged 17-frame physics teacher is evaluated on real rolling windows;
each window retains original block, candidate and frame lineage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

SCHEMA = "archived_terminal_short_auxiliary_replay_v1"
SOURCE = "known_task_trusted_evidence_arbitration_20261005_v8"
PREVIOUS = "known_task_archived_release_evidence_20261006_v1"
CONTRACT_SHA = "750d7706f586f69e9adf81b7463f8fb787ae8161410effcba0dc776e1a132605"
ARCHIVE_SHA = "55cb3651e63927a23cf3ce351bbd5673b13c42a8da1bd968a880915fefa25fce"
TEACHER_SHA = "1c85d2b4667594d7a419501950bdcebbec134b6bc9fa2f4ea05ba476834d5053"
PREVIOUS_MANIFEST_SHA = "86ed9dcb6d485510ac252f8997ad591215739ae3fc758ded676b29e8c45dad64"
PREVIOUS_AUDIT_SHA = "3e6da74259519b418e26c60f19c190cbe3768649e0963268ef6af38168bd003f"
FIXED = {
    (46,10,"execution_contact_deviation","train"): dict(length=3,cid=1,previous_cid=3,
        trajectory="99f7aec690466922a5ea952648b281e6fcf63d7cd55869e67acc631cdcec2057",
        terminal="d3d161a3ccc146501dd53426c9414094055ed16604b8463244e718858ea85ba3"),
    (46,13,"clean","val"): dict(length=7,cid=3,previous_cid=0,
        trajectory="38c3e93d378cf9b615d0d16cc4faeaeb36e86d2236824c177d4a049712d683f0",
        terminal="4ddc3aed9ff7e380bb493ae70ea66457427255d790fb4f00aa718fa90d7ad154"),
}
CONTACTS = ("left_finger_target_contact","right_finger_target_contact",
            "target_support_contact","target_anchor_contact")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def import_verified(name,path,expected):
    if not Path(path).is_file() or sha(path)!=expected:raise ValueError("Exact frozen code SHA required: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def validate_contract(contract):
    if contract.get("schema")!="archived_terminal_short_auxiliary_scope_v1" or contract.get("source")!=SOURCE:
        raise ValueError("Wrong frozen terminal-tail contract")
    entries=contract.get("entries",[]);seen=set()
    if len(entries)!=2:raise ValueError("Exactly the two approved tails, not an expanded set")
    for e in entries:
        key=(e.get("task"),e.get("state"),e.get("context"),e.get("split"))
        f=FIXED.get(key)
        if f is None or key in seen or e.get("suite")!="libero90":raise ValueError("Fixed source identity/split mismatch")
        seen.add(key)
        if e.get("observed_block")!=7 or e.get("valid_length")!=f["length"] or e.get("candidate_id")!=f["cid"]:
            raise ValueError("Short length or own candidate identity changed")
        if e.get("trajectory_sha256")!=f["trajectory"] or e.get("terminal_metadata_sha256")!=f["terminal"]:
            raise ValueError("Short source hash changed")
        p=e.get("previous_block_source",{})
        if p.get("observed_block")!=6 or p.get("candidate_id")!=f["previous_cid"] or p.get("same_actual_execution_chain") is not True:
            raise ValueError("Explicit previous event lineage required")
    return entries


def validate_arrays(arrays,length):
    if set(arrays)!={"primary","wrist","requested","applied"}:raise ValueError("Unexpected original tail fields")
    for v in ("primary","wrist"):
        if arrays[v].shape!=(length,256,256,3) or arrays[v].dtype!=np.uint8:
            raise ValueError("Original real-length uint8 RGB, never padding")
    for k in ("requested","applied"):
        if arrays[k].shape!=(length,7) or not np.isfinite(arrays[k]).all():
            raise ValueError("Real-length finite command records required")


def check_terminal(done,step,length):
    # Only technical alignment. This boolean is not passed to physics teacher.
    if bool(done)!=(step==length):raise ValueError("Original terminal step changed")


def frame_identity(row,block,step):
    return dict(dataset=SOURCE,suite="libero90",task=row["task"],state=row["state"],
        candidate_id=block["candidate_id"],observed_block_id=block["directory"],
        local_step=step,chain_step=(block["block"]-3)*16+step,
        original_rgb_index=step-1 if step else None,cached_reference_only=step==0 and block["block"]==3)


def history_from_prefix(row,prefix):
    history=[];lineage=[]
    for b in prefix:
        for step,frame in enumerate(b["frames"]):
            if step==0 and history:continue
            history.append(copy.deepcopy(frame));lineage.append(frame_identity(row,b["block"],step))
    return history,lineage


def derive_tail_windows(teacher,history,lineage,tail_length):
    if len(history)!=len(lineage) or len(history)<17+tail_length:raise ValueError("Actual contiguous past window required")
    reports=[];first_event=None
    for index in range(len(history)-tail_length,len(history)):
        window=copy.deepcopy(history[index-16:index+1]);sources=copy.deepcopy(lineage[index-16:index+1])
        for step,frame in enumerate(window):frame["step"]=step
        measured=teacher.derive_temporal_recovery(window,entity_kind="rigid",joint_unit=None)
        current=measured["frames"][-1]["physics"]
        carrying=[i for i,f in enumerate(measured["frames"][:-1])
            if f["physics"]["carried_sufficient_evidence"]["value"] is True and
            f["physics"]["carried_sufficient_evidence"]["measurement_valid"]]
        event=dict(current_source=sources[-1],physics=current,window_sources=sources,
            prior_carried_source=sources[carrying[-1]] if carrying else None,
            event_source="actual_executed_contiguous_chain_window_only",
            supervision_only=True,source_candidate_id_not_replaced=True)
        if current["released_sufficient_evidence"]["value"] is True and first_event is None:
            first_event=copy.deepcopy(event)
        reports.append(event)
    return dict(schema="archived_release_real_tail_measurements_v1",frames=reports,
        first_release_event=first_event,physical_release_exists=first_event is not None,
        physical_teacher_thresholds_unchanged=True,future_used_for_pre_execution_X=False,
        observation_certificates_pending=True,training_ready=False)


def repeat_check(first,second):
    if len(first["frames"])!=len(second["frames"]):raise ValueError("Repeat length mismatch")
    delta=np.abs(np.c_[first["qpos"],first["qvel"]]-np.c_[second["qpos"],second["qvel"]])
    if not np.isfinite(delta).all():raise ValueError("Nonfinite repeat physics")
    contacts=all(all(a["physics"][k]==b["physics"][k] for k in CONTACTS)
        for a,b in zip(first["frames"],second["frames"]))
    report=dict(sim_max_abs=float(delta.max()),sim_p95=float(np.quantile(delta,.95)),
        fraction_gt_1e3=float(np.mean(delta>1e-3)),contact_atoms_match=contacts,
        terminal_sequence_match=first["terminal_steps"]==second["terminal_steps"])
    report["passed"]=(report["sim_max_abs"]<=1e-3 and report["sim_p95"]<=1e-4 and
        report["fraction_gt_1e3"]==0 and contacts and report["terminal_sequence_match"])
    if not report["passed"]:raise ValueError("Frozen same-source repeat QC failed: "+json.dumps(report))
    return report


def event_signature(measurements):
    names=("carried_sufficient_evidence","released_sufficient_evidence","holding_after_measured_release")
    return [(x["current_source"],x["prior_carried_source"],
        {n:(x["physics"][n]["value"],x["physics"][n]["measurement_valid"]) for n in names})
        for x in measurements["frames"]]


def freeze(root,contract_path,archive):
    if sha(contract_path)!=CONTRACT_SHA:raise ValueError("Frozen local short-tail contract SHA changed")
    contract=read(contract_path);entries=validate_contract(contract)
    if sha(contract_path)!=CONTRACT_SHA:raise ValueError("Contract changed while reading")
    old=root/"outputs"/PREVIOUS
    if sha(old/"frozen_membership.json")!=PREVIOUS_MANIFEST_SHA or sha(old/"pilot_audit.json")!=PREVIOUS_AUDIT_SHA:
        raise ValueError("The previously passed fixed two-chain pilot changed")
    previous=read(old/"frozen_membership.json");audit=read(old/"pilot_audit.json")
    if audit.get("technical_replay_passed") is not True:raise ValueError("Prior prefix pilot must have passed")
    archive.verify_source_table(root,previous["source_sha256"])
    rows=[];hashes=dict(previous["source_sha256"])
    for entry in entries:
        scenario=f'task{entry["task"]}_state{entry["state"]}_{entry["context"]}'
        found=[x for x in previous["pilot"] if x["scenario"]==scenario and x["split"]==entry["split"]]
        if len(found)!=1:raise ValueError("Tail does not inherit exact previously passed chain")
        row=copy.deepcopy(found[0]);folder=root/"outputs"/SOURCE/"scenarios"/scenario/"full_repaired/block_7"
        if str(folder)!=entry["cloud_directory"]:raise ValueError("Fixed absolute source folder disagrees")
        if row["chain"][-1]["block"]!=6 or row["chain"][-1]["candidate_id"]!=entry["previous_block_source"]["candidate_id"]:
            raise ValueError("Previous block candidate lineage changed")
        for name,digest in (("trajectory.npz",entry["trajectory_sha256"]),("terminal_short_block.json",entry["terminal_metadata_sha256"])):
            path=folder/name
            if sha(path)!=digest:raise ValueError("Frozen short-tail file changed")
            hashes[str(path.relative_to(root))]=digest
        decision=folder.parent/"decision_7.json";decision_sha=sha(decision);meta=read(decision)
        if meta.get("selected_candidate_id")!=entry["candidate_id"] or meta.get("fallback") is not None:
            raise ValueError("Recorded selected short arm identity changed")
        hashes[str(decision.relative_to(root))]=decision_sha
        terminal=read(folder/"terminal_short_block.json")
        if terminal.get("executed")!=entry["valid_length"] or terminal.get("not_fed_to_attributor") is not True or terminal.get("success") is not True:
            raise ValueError("Fixed terminal alignment metadata changed")
        if (folder/"evidence/block/actual_observed_proprio.npy").exists():raise ValueError("Frozen missing endpoint proprio state changed")
        with np.load(folder/"trajectory.npz",allow_pickle=False) as saved:
            validate_arrays({k:saved[k] for k in saved.files},entry["valid_length"])
        row["tail"]=dict(block=7,candidate_id=entry["candidate_id"],directory=str(folder.relative_to(root)),
            valid_length=entry["valid_length"],original_endpoint_proprio="not_available",
            original_terminal_metadata_is_not_label=True)
        rows.append(row)
    archive.verify_source_table(root,hashes)
    return dict(schema=SCHEMA,source=SOURCE,contract_sha256=CONTRACT_SHA,prefix_source=PREVIOUS,
        source_sha256=hashes,chains=rows,complete_blocks_count_unchanged=797,
        short_tails=2,short_actual_steps=10,role="offline_historical_selected_arm_auxiliary_only",
        new_cosmos_queries=0,new_candidate_actions=0,training_ready=False)


def replay_tail(env,base,binding,row,prefix):
    folder=Path(row["tail"]["absolute_directory"]);length=row["tail"]["valid_length"]
    with np.load(folder/"trajectory.npz",allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    validate_arrays(arrays,length)
    from robosuite import macros
    from robosuite.environments.robot_env import IMAGE_CONVENTION_MAPPING
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    last=prefix[-1];frames=[copy.deepcopy(last["frames"][-1])];frames[0]["step"]=0
    proprio=[last["proprio"][-1]];qpos=[last["qpos"][-1]];qvel=[last["qvel"][-1]]
    segments={v:[last["segments"][v][-1]] for v in ("primary","wrist")};checks=[];terminal_steps=[]
    for step in range(1,length+1):
        raw,_,done,_=env.step(np.asarray(arrays["applied"][step-1],np.float32).tolist())
        actual={v:arrays[v][step-1] for v in ("primary","wrist")}
        check=dict(step=step,**{v:base.image_check(np.flipud(np.asarray(raw[k])),actual[v])
            for v,k in (("primary","agentview_image"),("wrist","robot0_eye_in_hand_image"))})
        check["passed"]=all(check[v]["passed"] for v in actual)
        if not check["passed"]:raise ValueError("Original short-tail RGB mismatch: "+json.dumps(check))
        check_terminal(done,step,length);terminal_steps.append(bool(done))
        frame,seg=base.capture(env,raw,actual,binding,step,convention);frames.append(frame);checks.append(check)
        pr=np.r_[raw["robot0_gripper_qpos"],raw["robot0_eef_pos"],raw["robot0_eef_quat"]]
        if pr.shape!=(9,) or not np.isfinite(pr).all():raise ValueError("Finite replay-measured proprio required")
        proprio.append(pr);qpos.append(base.sim_copy(env,"qpos"));qvel.append(base.sim_copy(env,"qvel"))
        for v in segments:segments[v].append(seg[v])
    return dict(frames=frames,checks=checks,terminal_steps=terminal_steps,arrays=arrays,
        proprio=np.stack(proprio),qpos=np.stack(qpos),qvel=np.stack(qvel),segments={v:np.stack(x) for v,x in segments.items()})


def execute(root,out,manifest,archive):
    code=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
    teacher=import_verified("temporal_recovery_teacher",code/"temporal_recovery_teacher.py",TEACHER_SHA)
    base=archive.import_file("_tail_capture_base",archive.verified_base_path(code))
    fork_path=root/"research_runs"/archive.REFERENCE/"collect_snapshot_fork.py"
    prior_audit=read(root/"outputs"/PREVIOUS/"pilot_audit.json")
    fork=import_verified("_tail_snapshot_restore",fork_path,prior_audit["snapshot_loader_sha256"])
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    import pickle
    env=None;reports=[]
    try:
        for frozen in manifest["chains"]:
            archive.verify_source_table(root,manifest["source_sha256"])
            row=copy.deepcopy(frozen)
            for b in row["chain"]:b["absolute_directory"]=str(archive.within(root,b["directory"]))
            row["tail"]["absolute_directory"]=str(archive.within(root,row["tail"]["directory"]))
            if env is not None:env.close()
            task=benchmark.get_benchmark_dict()["libero_90"]().get_task(row["task"])
            env,_=get_libero_env(task,"cosmos",resolution=256);env.reset();binding=base.bind(env,row["task"])
            archive.verify_chain_sources(root,row)
            snapshot_path=archive.within(root,row["root_provenance"]["root_directory"])/"root_runtime_snapshot.pkl"
            # Guarded exact generated-project snapshot bytes only, verified above.
            snapshot=pickle.loads(snapshot_path.read_bytes())
            runs=[];windows=[]
            for repeat in range(2):
                prefix=archive.run_chain(env,fork,base,binding,snapshot,row)
                tail=replay_tail(env,base,binding,row,prefix)
                history,lineage=history_from_prefix(row,prefix)
                for step,frame in enumerate(tail["frames"][1:],1):
                    history.append(frame);lineage.append(frame_identity(row,row["tail"],step))
                windows.append(derive_tail_windows(teacher,history,lineage,row["tail"]["valid_length"]))
                runs.append((prefix,tail))
            for a,b in zip(runs[0][0],runs[1][0]):
                a["terminal_steps"]=[];b["terminal_steps"]=[];repeat_check(a,b)
            qc=repeat_check(runs[0][1],runs[1][1])
            if event_signature(windows[0])!=event_signature(windows[1]):raise ValueError("Repeat causal event measurements changed")
            prefix,tail=runs[0];dest=out/"auxiliary"/row["scenario"]/'block_7'
            dump(dest/"temporal_teacher.json",dict(role="offline_supervision_only",identity={
                k:frame_identity(row,row["tail"],0)[k] for k in ("dataset","suite","task","state","candidate_id","observed_block_id")},
                split=row["split"],valid_length=row["tail"]["valid_length"],frames=tail["frames"],
                start_frame_source=frame_identity(row,row["chain"][-1],16),
                original_endpoint_proprio="not_available",original_complete_block_count_unchanged=797,
                physical_gt_never_deployment_feature=True,terminal_metadata_never_release_label=True))
            dump(dest/"temporal_measurements.json",windows[0])
            dump(dest/"alignment_audit.json",dict(passed=True,checks=tail["checks"],repeat=qc,
                original_endpoint_proprio=dict(available=False,max_abs=None,reason="source_did_not_save_terminal_proprio"),
                original_short_length=row["tail"]["valid_length"],padding_applied=False,training_ready=False))
            np.savez_compressed(dest/"physical_teacher_arrays.npz",proprio=tail["proprio"],qpos=tail["qpos"],qvel=tail["qvel"],
                **tail["segments"],requested=tail["arrays"]["requested"],applied=tail["arrays"]["applied"],
                actual_primary=tail["arrays"]["primary"],actual_wrist=tail["arrays"]["wrist"])
            np.savez_compressed(dest/"repeat_physics_arrays.npz",proprio=runs[1][1]["proprio"],qpos=runs[1][1]["qpos"],qvel=runs[1][1]["qvel"])
            reports.append(dict(scenario=row["scenario"],split=row["split"],candidate_id=row["tail"]["candidate_id"],
                actual_length=row["tail"]["valid_length"],technical_replay_passed=True,
                physical_release_exists=windows[0]["physical_release_exists"],first_release_event=windows[0]["first_release_event"],
                original_endpoint_proprio_available=False,observation_certificate_pending=True))
            archive.verify_source_table(root,manifest["source_sha256"])
        report=dict(schema=SCHEMA,technical_replay_passed=True,chains=reports,role="historical_short_auxiliary_only",
            new_cosmos_queries=0,new_candidate_actions=0,complete_blocks_count_unchanged=797,
            padding_applied=False,source_unchanged=True,training_ready=False,training_started=False,
            code_sha256={p.name:sha(p) for p in (Path(__file__),code/'replay_archived_release_evidence.py',
                code/'replay_temporal_recovery_evidence.py',code/'temporal_recovery_teacher.py')})
        dump(out/"pilot_audit.json",report);return report
    finally:
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    root=args.root.resolve();out=args.output.resolve();code=Path(__file__).resolve().parent
    if out.exists() or root not in out.parents or (root/'outputs'/SOURCE) in out.parents or (root/'outputs'/PREVIOUS) in out.parents:
        raise ValueError('New root-local output required; sources never overwritten')
    out.mkdir(parents=True,exist_ok=False)
    try:
        archive=import_verified('_tail_previous_replay',code/'replay_archived_release_evidence.py',ARCHIVE_SHA)
        manifest=freeze(root,args.contract.resolve(),archive);dump(out/'frozen_membership.json',manifest)
        if args.replay:execute(root,out,manifest,archive)
        dump(out/'status.json',dict(phase='technical_pilot_complete' if args.replay else 'manifest_frozen',
            training_ready=False,training_started=False,new_cosmos_queries=0))
    except Exception as error:
        dump(out/'failure.json',dict(error_type=type(error).__name__,message=str(error),source_retained=True,
            training_started=False,new_cosmos_queries=0));raise
    finally:
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'}
        (out/'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(files.items())),encoding='utf-8')


if __name__=='__main__':main()
