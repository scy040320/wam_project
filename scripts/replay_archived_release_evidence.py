"""Freeze and optionally replay immutable selected-arm release AUX evidence.

Default is a CPU-only frozen manifest. --pilot is explicit and executes two
predeclared historical chains, without a policy query or model fitting. This
is not whole-pool counterfactual supervision. Simulator measurements, geometric
visibility diagnostics and learnable certificates remain separate layers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

SOURCE = "known_task_trusted_evidence_arbitration_20261005_v8"
ROOT_SOURCE = "candidate_reranking_known_task_closedloop_mechanism_20261004_v3_shared_source"
AUDIT_SOURCE = "known_task_recovery_contract_audit_20261006_v9_r2"
REFERENCE = "candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"
SCHEMA = "archived_release_auxiliary_replay_v1"
BASE_SHA = "3684243f669a27430b27f45673c138859548e160dff757de7269eb25128e8069"
VAL_STATES = {9:13,46:13,57:13}
REQUIRED_ROOT_FILES = ("root_runtime_snapshot.pkl", "shared_root_pool/pool.json",
    "source_block/evidence/block/actual_observed_primary.npy",
    "source_block/evidence/block/actual_observed_wrist.npy",
    "source_block/evidence/block/actual_observed_proprio.npy")


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda:stream.read(1024*1024),b""):
            result.update(part)
    return result.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def within(root, relative):
    root=Path(root).resolve()
    if not isinstance(relative,str) or Path(relative).is_absolute():
        raise ValueError("Source path must be root-relative")
    path=(root/relative).resolve()
    if root not in path.parents:raise ValueError("Source path escapes root")
    return path


def close_before_open(actions):
    """Frozen scheduling criterion ONLY, not a physical release annotation.

    Official Panda command +1 closes, -1 opens. Eligibility reads requested
    actions available before execution, never contacts, predictions or success.
    """
    a=np.asarray(actions)
    if a.shape!=(16,7) or not np.isfinite(a).all():
        raise ValueError("Complete finite requested 16x7 required")
    return any(bool((a[:i,6]>.5).any()) for i in np.flatnonzero(a[:,6]<-.5))


def chain_block_numbers(directory, last_block):
    if type(last_block) is not int or last_block<3:raise ValueError("Historical source begins at block3")
    by_number={}
    for path in Path(directory).glob("block_*"):
        match=re.fullmatch(r"block_(\d+)",path.name)
        if match:
            number=int(match[1])
            if number in by_number:raise ValueError("Duplicate block number")
            by_number[number]=path
    wanted=list(range(3,last_block+1))
    if any(n not in by_number for n in wanted):raise ValueError("Missing prefix block; partial-state substitute forbidden")
    return [by_number[n] for n in wanted]


def validate_root_provenance(root, scenario, frozen_inputs):
    directory=root/"outputs"/ROOT_SOURCE/"scenarios"/scenario
    hashes={}
    for name in REQUIRED_ROOT_FILES:
        path=directory/name;relative=str(path.relative_to(root)).replace("\\","/")
        digest=sha(path)
        if frozen_inputs.get(relative)!=digest:
            raise ValueError("V8 frozen root dependency absent or changed: "+relative)
        hashes[relative]=digest
    pool_path=directory/"shared_root_pool/pool.json"
    child_path=root/"outputs"/SOURCE/"scenarios"/scenario/"scenario.json"
    reads={str(p.relative_to(root)).replace("\\","/"):sha(p) for p in (pool_path,child_path)}
    pool=load(pool_path)
    child=load(child_path)
    verify_source_table(root,reads)
    if child.get("root_observation_sha256")!=pool.get("observation_sha256"):
        raise ValueError("V8 did not inherit this immutable root observation")
    expected_suite="libero90"
    return dict(root_source=ROOT_SOURCE,source=SOURCE,suite=expected_suite,
        root_directory=str(directory.relative_to(root)).replace("\\","/"),
        frozen_inputs_sha256=hashes,source_read_sha256=reads,
        root_observation_sha256=pool["observation_sha256"],
        provenance_basis="V8 run_arbitrated_closedloop ORIGINAL/root_runtime_snapshot and frozen_inputs")


def freeze(root):
    source=root/"outputs"/SOURCE
    audit_path=root/"outputs"/AUDIT_SOURCE/"historical_selected_arm_transport_audit.json"
    protocol_path=source/"frozen_protocol.json"
    touched={str(p.relative_to(root)).replace("\\","/"):sha(p) for p in (audit_path,protocol_path)}
    accepted=load(audit_path);protocol=load(protocol_path)
    def remember(path,digest=None):
        relative=str(path.relative_to(root)).replace("\\","/")
        digest=sha(path) if digest is None else digest
        if relative in touched and touched[relative]!=digest:raise ValueError("Source changed while freezing membership")
        touched[relative]=digest
    if (accepted.get("source",SOURCE)!=SOURCE or protocol.get("version")!=SOURCE or
        protocol.get("benchmark_suite") not in ("libero90","libero_90") or
        protocol.get("budget",{}).get("total_policy_steps")!=400):
        raise ValueError("Frozen historical source/suite/budget mismatch")
    raw=[x for x in accepted["entries"] if x.get("potential_auxiliary")]
    if len(raw)!=797:raise ValueError("Historical complete count must remain797")
    groups={};eligible=[]
    for row in raw:
        task,state=row["task"],row["state"]
        if task not in VAL_STATES:continue
        path=within(root,row["path"])
        if root/"outputs"/SOURCE not in path.parents or path.name!="trajectory.npz":
            raise ValueError("Historical trajectory is outside immutable V8")
        match=re.fullmatch(r"task(\d+)_state(\d+)_(.+)",path.parent.parent.parent.name)
        if not match or int(match[1])!=task or int(match[2])!=state or path.parent.parent.name!="full_repaired":
            raise ValueError("Historical source-scoped task/state/method mismatch")
        remember(path)
        with np.load(path,allow_pickle=False) as trajectory:
            requested=trajectory["requested"];applied=trajectory["applied"]
            if applied.shape!=(16,7) or not np.isfinite(applied).all():raise ValueError("Incomplete applied block")
            scheduled=close_before_open(requested)
        remember(path)
        if not scheduled:continue
        block=int(path.parent.name.split("_")[-1]);scenario=path.parent.parent.parent.name
        split="val" if state==VAL_STATES[task] else "train"
        if scenario not in groups:
            groups[scenario]=validate_root_provenance(root,scenario,protocol["frozen_inputs"])
        provenance=groups[scenario]
        for relative,digest in provenance["frozen_inputs_sha256"].items():remember(within(root,relative),digest)
        for relative,digest in provenance["source_read_sha256"].items():remember(within(root,relative),digest)
        chain=[]
        for folder in chain_block_numbers(path.parent.parent,block):
            evidence=folder/"evidence/block";files=[folder/"trajectory.npz",
                evidence/"O_t_primary.npy",evidence/"O_t_wrist.npy",
                evidence/"O_t_proprio.npy",evidence/"actual_observed_proprio.npy",
                folder.parent/("decision_"+folder.name.split("_")[-1]+".json")]
            block_hashes={str(p.relative_to(root)).replace("\\","/"):sha(p) for p in files}
            for relative,digest in block_hashes.items():remember(within(root,relative),digest)
            with np.load(folder/"trajectory.npz",allow_pickle=False) as saved:
                if any(saved[k].shape!=(16,256,256,3) for k in ("primary","wrist")) or any(
                    saved[k].shape!=(16,7) or not np.isfinite(saved[k]).all() for k in ("requested","applied")):
                    raise ValueError("Prefix contains a short block; retain source and stop this pilot")
                applied_sha=hashlib.sha256(np.ascontiguousarray(saved["applied"]).tobytes()).hexdigest()
            decision=load(files[-1]);cid=decision.get("selected_candidate_id")
            if type(cid) is not int or cid not in range(4) or decision.get("fallback") is not None:
                raise ValueError("Only recorded selected candidate arms are eligible")
            chain.append(dict(block=int(folder.name.split("_")[-1]),candidate_id=cid,
                directory=str(folder.relative_to(root)).replace("\\","/"),files_sha256=block_hashes,
                applied_c_order_bytes_sha256=applied_sha))
        eligible.append(dict(source_scope=SOURCE,suite="libero90",task=task,state=state,
            scenario=scenario,split=split,selected_block=block,path=row["path"],chain=chain,
            root_provenance=provenance,role="historical_selected_arm_auxiliary_not_counterfactual_pool"))
    eligible.sort(key=lambda x:x["path"])
    counts={s:sum(x["split"]==s for x in eligible) for s in ("train","val")}
    if len(eligible)!=81 or counts!={"train":64,"val":17}:raise ValueError("Frozen requested schedule membership must be81/64/17")
    pilot=[next(x for x in eligible if x["split"]==s) for s in ("train","val")]
    verify_source_table(root,touched)
    return dict(schema=SCHEMA,source=SOURCE,role="offline_selected_arm_auxiliary_only",membership=eligible,
        blocks=81,split=counts,pilot=pilot,pilot_selection="lexicographically first source path per existing split; no outcome criterion",
        frozen_protocol_sha256=sha(protocol_path),historical_transport_audit_sha256=sha(audit_path),
        source_sha256=touched,freeze_source_unchanged=True,
        old797_unchanged=True,new_queries=0,new_candidate_actions=0,training_started=False,training_ready=False,
        eligibility_is_requested_close_before_open_schedule_not_release_truth=True)


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def verify_chain_sources(root,row):
    for relative,digest in row["root_provenance"]["frozen_inputs_sha256"].items():
        if sha(within(root,relative))!=digest:raise ValueError("Frozen root dependency changed")
    verify_source_table(root,row["root_provenance"].get("source_read_sha256",{}))
    for block in row["chain"]:
        for relative,digest in block["files_sha256"].items():
            if sha(within(root,relative))!=digest:raise ValueError("Frozen historical chain source changed")


def verify_source_table(root,table):
    for relative,digest in table.items():
        if sha(within(root,relative))!=digest:raise ValueError("Immutable historical source changed: "+relative)


def verified_base_path(directory):
    path=Path(directory)/"replay_temporal_recovery_evidence.py"
    if not path.is_file() or sha(path)!=BASE_SHA:
        raise ValueError("Only the exact SHA-frozen flat CODE replay base is permitted")
    return path


def run_chain(env,fork,base,binding,snapshot,row):
    env.reset();base.install_scoped_segmentation_decoder(env)
    raw=fork.restore_runtime_snapshot(env,copy.deepcopy(snapshot))
    if not np.array_equal(np.asarray(env.get_sim_state()),snapshot["sim_state"]):
        raise ValueError("Complete root snapshot did not restore exactly")
    from robosuite import macros
    from robosuite.environments.robot_env import IMAGE_CONVENTION_MAPPING
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    output=[]
    for block in row["chain"]:
        directory=Path(block["absolute_directory"])
        with np.load(directory/"trajectory.npz",allow_pickle=False) as saved:
            expected={v:saved[v].copy() for v in ("primary","wrist")}
            requested,applied=saved["requested"].copy(),saved["applied"].copy()
        if hashlib.sha256(np.ascontiguousarray(applied).tobytes()).hexdigest()!=block["applied_c_order_bytes_sha256"]:
            raise ValueError("Recorded applied sequence changed")
        evidence=directory/"evidence/block"
        first={v:np.load(evidence/("O_t_"+v+".npy"),allow_pickle=False) for v in expected}
        # Cached frame0 is diagnostic, not a physical/observable release label.
        frames=[];checks=[];qpos=[];qvel=[];segments={v:[] for v in expected};proprio=[]
        for step in range(17):
            done=False
            if step:raw,_,done,_=env.step(np.asarray(applied[step-1],np.float32).tolist())
            actual={v:first[v] if step==0 else expected[v][step-1] for v in expected}
            pr=np.concatenate([raw["robot0_gripper_qpos"],raw["robot0_eef_pos"],raw["robot0_eef_quat"]])
            checks_at=dict(step=step,cached_entry_diagnostic_only=step==0,
                **{v:base.image_check(np.flipud(np.asarray(raw[key])),actual[v]) for v,key in
                    (("primary","agentview_image"),("wrist","robot0_eye_in_hand_image"))})
            checks_at["passed"]=step==0 or all(checks_at[v]["passed"] for v in expected)
            if step==16:
                old=np.load(evidence/"actual_observed_proprio.npy",allow_pickle=False)
                checks_at["endpoint_proprio_max_abs"]=float(np.abs(pr-old).max())
                checks_at["passed"] &= checks_at["endpoint_proprio_max_abs"]<=1e-3
            if not checks_at["passed"]:raise ValueError("Frozen original RGB/endpoint mismatch: "+json.dumps(checks_at))
            if done and (step<16 or block is not row["chain"][-1]):
                raise ValueError("Original prefix became terminal before the declared end")
            frame,seg=base.capture(env,raw,actual,binding,step,convention)
            frames.append(frame);checks.append(checks_at);proprio.append(pr)
            qpos.append(base.sim_copy(env,"qpos"));qvel.append(base.sim_copy(env,"qvel"))
            for v in expected:segments[v].append(seg[v])
        output.append(dict(block=block,frames=frames,checks=checks,proprio=np.stack(proprio),
            qpos=np.stack(qpos),qvel=np.stack(qvel),segments={v:np.stack(x) for v,x in segments.items()}))
    return output


def pilot(root,out,manifest):
    sys.path.insert(0,str(root))
    directory=Path(__file__).resolve().parent
    base_path=verified_base_path(directory)
    base=import_file("_historical_release_base",base_path)
    fork_path=root/"research_runs"/REFERENCE/"collect_snapshot_fork.py"
    fork=import_file("_historical_release_snapshot",fork_path)
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    import pickle
    reports=[];env=None
    try:
        verify_source_table(root,manifest["source_sha256"])
        for row in manifest["pilot"]:
            if env is not None:env.close()
            task=benchmark.get_benchmark_dict()["libero_90"]().get_task(row["task"])
            env,_=get_libero_env(task,"cosmos",resolution=256);env.reset();binding=base.bind(env,row["task"])
            verify_chain_sources(root,row)
            prepared=copy.deepcopy(row)
            for block in prepared["chain"]:
                for relative,digest in block["files_sha256"].items():
                    if sha(within(root,relative))!=digest:raise ValueError("Frozen chain source changed before replay")
                block["absolute_directory"]=str(within(root,block["directory"]))
            # Only the exact, provenance-verified runtime data generated by the
            # audited project is loaded, after byte SHA matches frozen V8 inputs.
            snap=root/row["root_provenance"]["root_directory"]/"root_runtime_snapshot.pkl"
            snapshot=pickle.loads(snap.read_bytes())
            first=run_chain(env,fork,base,binding,snapshot,prepared)
            repeated=run_chain(env,fork,base,binding,snapshot,prepared)
            chain_report=[]
            for original,repeat in zip(first,repeated):
                a,b=original["frames"],repeat["frames"]
                delta=np.abs(np.c_[original["qpos"],original["qvel"]]-np.c_[repeat["qpos"],repeat["qvel"]])
                contacts=("left_finger_target_contact","right_finger_target_contact","target_support_contact","target_anchor_contact")
                physical=base.derive_temporal_recovery(a,entity_kind=binding["kind"],joint_unit=binding["joint_unit"])
                repeated_physical=base.derive_temporal_recovery(b,entity_kind=binding["kind"],joint_unit=binding["joint_unit"])
                check=dict(sim_max_abs=float(delta.max()),sim_p95=float(np.quantile(delta,.95)),
                    contact_atoms_match=all(all(x["physics"][n]==y["physics"][n] for n in contacts) for x,y in zip(a,b)),
                    first_measured_events_match=physical["first_measured_events"]==repeated_physical["first_measured_events"])
                check["passed"]=check["sim_max_abs"]<=1e-3 and check["sim_p95"]<=1e-4 and not np.any(delta>1e-3) and check["contact_atoms_match"] and check["first_measured_events_match"]
                if not check["passed"]:raise ValueError("Same-chain repeat physics/contact/event mismatch")
                block=original["block"];dest=out/"auxiliary"/row["scenario"]/f'block_{block["block"]}'
                identity=dict(dataset=SOURCE,suite="libero90",task=row["task"],state=row["state"],
                    candidate_id=block["candidate_id"],observed_block_id=block["directory"])
                dump(dest/"temporal_teacher.json",dict(role="offline_supervision_only",identity=identity,
                    split=row["split"],kind=binding["kind"],frames=a,
                    selected_arm_only=True,physical_gt_is_not_deployment_feature=True))
                dump(dest/"temporal_measurements.json",physical)
                dump(dest/"alignment_audit.json",dict(passed=True,checks=original["checks"],repeat=check,
                    actual_observation_certificate_pending=True,raw_geometry_not_contact_certificate=True))
                np.savez_compressed(dest/"physical_teacher_arrays.npz",proprio=original["proprio"],
                    qpos=original["qpos"],qvel=original["qvel"],**original["segments"])
                chain_report.append(dict(block=block["block"],directory=str(dest.relative_to(out)),
                    physical_first_events=physical["first_measured_events"],observation_supervision_ready=False))
            reports.append(dict(scenario=row["scenario"],split=row["split"],blocks=chain_report,
                frozen_snapshot_sha256=sha(snap),passed=True))
            verify_chain_sources(root,row)
        verify_source_table(root,manifest["source_sha256"])
        report=dict(schema=SCHEMA,technical_replay_passed=True,chains=reports,role="historical_auxiliary_only",
            source_unchanged=True,new_policy_queries=0,new_candidate_actions=0,training_started=False,training_ready=False,
            actual_observability_certificates_pending=True,script_sha256=sha(Path(__file__)),
            base_script_sha256=sha(base_path),snapshot_loader_sha256=sha(fork_path))
        dump(out/"pilot_audit.json",report);return report
    finally:
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot",action="store_true",help="Explicit approved technical chain replay; default only freezes manifest")
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out.exists() or out==root or out in root.parents or root not in out.parents or (root/"outputs"/SOURCE) in out.parents or (root/"outputs"/ROOT_SOURCE) in out.parents:
        raise ValueError("New root-local output only; source overwrite/mutation forbidden")
    out.mkdir(parents=True,exist_ok=False)
    try:
        manifest=freeze(root);dump(out/"frozen_membership.json",manifest)
        if args.pilot:pilot(root,out,manifest)
        dump(out/"status.json",dict(phase="technical_pilot_complete" if args.pilot else "manifest_frozen",
            training_started=False,new_queries=0,training_ready=False))
    except Exception as error:
        dump(out/"failure.json",dict(error_type=type(error).__name__,message=str(error),source_retained=True,
            new_queries=0,training_started=False));raise
    finally:
        # This NEW output only; source trajectories remain immutable. Include
        # engineering failures as well as accepted technical records.
        files={str(p.relative_to(out)).replace("\\","/"):sha(p) for p in out.rglob("*")
            if p.is_file() and p.name!="SHA256SUMS.txt"}
        (out/"SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name,digest in sorted(files.items())),encoding="utf-8")


if __name__=="__main__":main()
