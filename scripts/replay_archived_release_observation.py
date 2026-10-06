"""Frozen visual-role metrics for two real historical terminal tails, no fit.

Only emits offline measurements and actual-RGB correspondence metrics. A new
chain-scoped certificate must separately evaluate them. Original proprio is
not fabricated; new repeat-verified measurements retain their own SHA lineage.
"""
from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path
import sys

import numpy as np

import replay_archived_release_tail as tail

SCHEMA="archived_terminal_tail_observation_metrics_v1"
VERSION="known_task_archived_release_observation_20261006_v1"
TAIL_SHA="bcd446ad73dc767ca7cf82dcfc532ceea7bf89d8217210f64328dc2e06fb08f4"
ADAPTER_SHA="957e14e0e60bb4e2cae3db99311e2f2b4ee59d31b4fd49951b33c3f047a9d99e"
GEOMETRY_SHA="f2eeca8e17d43f5939889a635a04252e4ee78bcd15632e9179a843e6843c7991"
ROLES=("target","eef","anchor","left","right")


def source_identity(lineage):
    return {k:lineage[k] for k in ("dataset","suite","task","state","candidate_id","observed_block_id")}|{"source_step":lineage["local_step"]}


def verified_endpoints(base,adapter_path):
    if tail.sha(adapter_path)!=ADAPTER_SHA:raise ValueError("Exact frozen adapter required before closure access")
    function=inspect.getclosurevars(base.replay).nonlocals.get("endpoints")
    if (not inspect.isfunction(function) or function.__qualname__!="attach.<locals>.endpoints" or
        Path(inspect.getsourcefile(function)).resolve()!=Path(adapter_path).resolve() or
        list(inspect.signature(function).parameters)!=["start","end","view","role"]):
        raise ValueError("Frozen adapter endpoint dependency API changed")
    return function


def aligned_history(row,prefix,short,caches):
    expected=sum(len(b["frames"]) for b in prefix)+len(short["frames"])-1
    if len(caches)!=expected:raise ValueError("Enhanced capture cache and actual source steps are misaligned")
    frames=[];lineage=[];selected=[];proprio=[];qpos=[];qvel=[];checks=[];index=0
    for b in prefix:
        for step,frame in enumerate(b["frames"]):
            cache=caches[index];index+=1
            if step==0 and frames:continue
            frames.append(copy.deepcopy(frame));selected.append(cache);lineage.append(tail.frame_identity(row,b["block"],step))
            proprio.append(b["proprio"][step]);qpos.append(b["qpos"][step]);qvel.append(b["qvel"][step]);checks.append(b["checks"][step])
    for step,frame in enumerate(short["frames"][1:],1):
        frames.append(copy.deepcopy(frame));selected.append(caches[index]);index+=1
        lineage.append(tail.frame_identity(row,row["tail"],step));proprio.append(short["proprio"][step])
        qpos.append(short["qpos"][step]);qvel.append(short["qvel"][step]);checks.append(dict(short["checks"][step-1],
            endpoint_proprio_max_abs=None,original_endpoint_proprio_available=False))
    if [x["chain_step"] for x in lineage]!=list(range(len(lineage))):raise ValueError("Actual chain indices are not contiguous")
    return dict(frames=frames,lineage=lineage,caches=selected,proprio=np.stack(proprio),
        qpos=np.stack(qpos),qvel=np.stack(qvel),checks=checks)


def causal_starts(index,source):
    if type(index) is not int or index<0:raise ValueError("Real nonnegative chain index required")
    origin=index-source["source_step"]
    return sorted(s for s in {max(1,index-2),max(1,index-16),origin+1,origin+2} if 0<s<index)


def enrich(first,second,qc,adapter,geometry,endpoints):
    if len(first["frames"])!=len(second["frames"]):raise ValueError("Same-arm observation repeat length changed")
    frames=first["frames"];caches=first["caches"];repeated=second["caches"]
    for index,frame in enumerate(frames):
        evidence=frame["observation_evidence"]
        evidence["chain_index"]=index;evidence["source_identity"]=source_identity(first["lineage"][index])
        evidence["same_candidate_repeat_qc"]=dict(physics_passed=qc["passed"],
            qpos_qvel_max_abs=qc["sim_max_abs"],qpos_qvel_p95=qc["sim_p95"],
            all_contact_atoms_match=qc["contact_atoms_match"],same_source_arm_not_candidate_zero_surrogate=True)
        for view in ("primary","wrist"):
            a,b=caches[index][view],repeated[index][view];noise={}
            for role in ROLES:
                da=adapter.descriptor(a["roles"][role],a["actual"],a["rendered"])
                db=adapter.descriptor(b["roles"][role],b["actual"],b["rendered"])
                ca,cb=da["centroid_xy"],db["centroid_xy"]
                noise[role]=dict(centroid_noise_px=float(np.linalg.norm(np.asarray(ca)-cb)) if ca is not None and cb is not None else None,
                    rgb_repeat=geometry.local_rgb_registration(a["rendered"],b["rendered"],a["roles"][role]|b["roles"][role]))
            evidence["views"][view]["repeat_noise"]=noise
        intervals=[]
        for start in causal_starts(index,evidence["source_identity"]):
            for view in ("primary","wrist"):
                a,b=caches[start][view],caches[index][view]
                tracks={r:geometry.track_rgb_role(a["actual"],b["actual"],a["roles"][r],b["roles"][r],
                    projected_endpoints=endpoints(caches[start],caches[index],view,r)) for r in ROLES}
                noises={}
                for role in ROLES:
                    values=[frames[j]["observation_evidence"]["views"][view]["repeat_noise"][role]["centroid_noise_px"] for j in (start,index)]
                    noises[role]=max(values) if all(x is not None for x in values) else None
                intervals.append(dict(start_step=start,end_step=index,view=view,tracks=tracks,same_candidate_repeat_noise=noises,
                    source_start_identity=source_identity(first["lineage"][start]),source_end_identity=source_identity(first["lineage"][index]),
                    cross_block_context_preserves_distinct_candidate_ids=True,no_future_identity_backfill=True,
                    geometry_correspondence_uses_private_depth_only_for_annotation=True,
                    finger_identity_requires_independent_same_side_actual_RGB_tracks=True,
                    camera_motion_compensated_by_per_frame_projection=False))
        evidence["actual_rgb_temporal_intervals"]=adapter.json_ready(intervals)
    return frames


def original_images(row,lineage,source_hashes):
    result={};folder=Path(lineage["observed_block_id"]);step=lineage["local_step"]
    for view in ("primary","wrist"):
        file=folder/"trajectory.npz" if step else folder/"evidence/block"/f"O_t_{view}.npy"
        relative=str(file).replace("\\","/");digest=source_hashes.get(relative)
        if not digest:raise ValueError("Original actual image hash lineage missing")
        result[view]=dict(path=relative,sha256=digest,array_name=view if step else None,
            array_index=step-1 if step else None,cached_reference_only=lineage["cached_reference_only"])
    return result


def execute(root,out,manifest,archive,adapter_path):
    code=Path(__file__).resolve().parent
    geometry=tail.import_verified("observation_geometry",code/"observation_geometry.py",GEOMETRY_SHA)
    adapter=tail.import_verified("_tail_visual_adapter",adapter_path,ADAPTER_SHA)
    teacher=tail.import_verified("temporal_recovery_teacher",code/"temporal_recovery_teacher.py",tail.TEACHER_SHA)
    base=archive.import_file("_tail_visual_capture",archive.verified_base_path(code));runtime=adapter.attach(base,root)
    endpoints=verified_endpoints(base,adapter_path)
    prior_audit=tail.read(root/"outputs"/tail.PREVIOUS/"pilot_audit.json")
    fork=tail.import_verified("_tail_visual_snapshot",root/"research_runs"/archive.REFERENCE/"collect_snapshot_fork.py",prior_audit["snapshot_loader_sha256"])
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    import pickle
    env=None;reports=[]
    try:
        for frozen in manifest["chains"]:
            archive.verify_source_table(root,manifest["source_sha256"]);row=copy.deepcopy(frozen)
            for b in row["chain"]:b["absolute_directory"]=str(archive.within(root,b["directory"]))
            row["tail"]["absolute_directory"]=str(archive.within(root,row["tail"]["directory"]))
            if env is not None:env.close()
            task=benchmark.get_benchmark_dict()["libero_90"]().get_task(row["task"])
            env,_=get_libero_env(task,"cosmos",resolution=256);env.reset();binding=base.bind(env,row["task"])
            archive.verify_chain_sources(root,row)
            snapshot=pickle.loads((archive.within(root,row["root_provenance"]["root_directory"])/"root_runtime_snapshot.pkl").read_bytes())
            histories=[];measurements=[];runs=[]
            for repeat in range(2):
                runtime["frames"]=[]
                prefix=archive.run_chain(env,fork,base,binding,snapshot,row);short=tail.replay_tail(env,base,binding,row,prefix)
                h=aligned_history(row,prefix,short,runtime["frames"]);histories.append(h);runs.append((prefix,short))
                measurements.append(tail.derive_tail_windows(teacher,h["frames"],h["lineage"],row["tail"]["valid_length"]))
            prefix_qc=[]
            for a,b in zip(runs[0][0],runs[1][0]):
                a["terminal_steps"]=[];b["terminal_steps"]=[];prefix_qc.append(tail.repeat_check(a,b))
            short_qc=tail.repeat_check(runs[0][1],runs[1][1])
            if tail.event_signature(measurements[0])!=tail.event_signature(measurements[1]):raise ValueError("Repeated original event sources changed")
            combined_qc=tail.repeat_check(dict(frames=histories[0]["frames"],qpos=histories[0]["qpos"],qvel=histories[0]["qvel"],terminal_steps=runs[0][1]["terminal_steps"]),
                dict(frames=histories[1]["frames"],qpos=histories[1]["qpos"],qvel=histories[1]["qvel"],terminal_steps=runs[1][1]["terminal_steps"]))
            h=histories[0];frames=enrich(h,histories[1],combined_qc,adapter,geometry,endpoints);dest=out/"chains"/row["scenario"]
            array_path=dest/"replay_arrays.npz";dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(array_path,proprio=h["proprio"],qpos=h["qpos"],qvel=h["qvel"],
                **{f'{v}_{name}':np.stack([c[v][name] for c in h["caches"]]) for v in ('primary','wrist') for name in ('actual','rendered','geom')},
                **{f'{v}_role_{r}':np.stack([c[v]['roles'][r] for c in h["caches"]]) for v in ('primary','wrist') for r in ROLES})
            digest=tail.sha(array_path);canonical=[]
            for index,frame in enumerate(frames):
                canonical.append(dict(chain_index=index,source_identity=source_identity(h["lineage"][index]),physics=frame["physics"],
                    observation_evidence=frame["observation_evidence"],
                    measured_proprio=dict(gripper_qpos_m=h['proprio'][index,:2].tolist(),eef_pos_m=h['proprio'][index,2:5].tolist(),
                        replay_lineage_sha256=digest,array_index=index,source='newly_measured_verified_historical_replay'),
                    repeat_qc=combined_qc,original_replay_checks=h['checks'][index],
                    original_image_sources=original_images(row,h['lineage'][index],manifest['source_sha256'])))
            tail.dump(dest/'temporal_teacher.json',dict(schema=SCHEMA,role='offline_supervision_only',frames=canonical,
                split=row['split'],real_short_length=row['tail']['valid_length'],original_complete_blocks_count_unchanged=797,
                replay_arrays_sha256=digest,original_short_endpoint_proprio_available=False,training_ready=False,
                original_per_step_proprio_not_fabricated=True,pre_execution_deployment_inputs_created=False))
            tail.dump(dest/'temporal_measurements.json',measurements[0])
            tail.dump(dest/'replay_audit.json',dict(passed=True,prefix_repeat_qc=prefix_qc,short_repeat_qc=short_qc,
                combined_repeat_qc=combined_qc,original_short_proprio_check=None,all_original_RGB_and_terminal_lengths_match=True,
                observation_certification_is_separate=True,new_mask_not_created=True))
            reports.append(dict(scenario=row['scenario'],split=row['split'],actual_short_length=row['tail']['valid_length'],
                canonical_frames=len(canonical),physical_release_exists=measurements[0]['physical_release_exists'],
                original_short_proprio_available=False,technical_passed=True,training_ready=False))
            archive.verify_source_table(root,manifest['source_sha256'])
            tail.dump(out/'status.json',dict(phase='finite_replay_running',completed_chains=len(reports),planned_chains=2,training_started=False,training_ready=False))
        report=dict(schema=SCHEMA,technical_replay_passed=True,chains=reports,source_unchanged=True,new_queries=0,new_candidate_actions=0,
            code_sha256={p.name:tail.sha(p) for p in code.glob('*.py')},
            endpoints_dependency_API='exact SHA verified adapter.attach.<locals>.endpoints via inspected closure',
            old_physical_atoms_and_masks_not_changed=True,certificates_not_created=True,training_ready=False,training_started=False)
        tail.dump(out/'pilot_audit.json',report)
    finally:
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    root=args.root.resolve();out=args.output.resolve();code=Path(__file__).resolve().parent
    if tail.sha(code/'replay_archived_release_tail.py')!=TAIL_SHA:raise ValueError('Exact source-frozen short replay helper required')
    if out.exists() or out.parent!=root/'outputs':
        raise ValueError('New independent output required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        sys.path.insert(0,str(root))
        archive=tail.import_verified('_observable_tail_archive',code/'replay_archived_release_evidence.py',tail.ARCHIVE_SHA)
        manifest=tail.freeze(root,args.contract.resolve(),archive);manifest['observation_schema']=SCHEMA
        manifest['same_arm_repeat_not_surrogate']=True;manifest['no_certification_or_training']=True
        tail.dump(out/'frozen_membership.json',manifest)
        if args.replay:execute(root,out,manifest,archive,code/'collect_recovery_observation_evidence.py')
        tail.dump(out/'status.json',dict(phase='finite_replay_complete' if args.replay else 'manifest_frozen',training_started=False,training_ready=False))
    except Exception as error:
        tail.dump(out/'failure.json',dict(error_type=type(error).__name__,message=str(error),source_retained=True,training_started=False));raise
    finally:
        files={str(p.relative_to(out)):tail.sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'}
        (out/'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name,digest in sorted(files.items())),encoding='utf-8')


if __name__=='__main__':main()
