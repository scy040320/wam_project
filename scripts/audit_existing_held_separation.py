"""Read-only 40-pool audit of reusable separation evidence, never held labels.

Physical carried=False means insufficient carrying evidence, NOT held=False.
Only reports Y-side diagnostics and joins frozen carrying positives at the
same source-scoped pool, split and step. No file write, replay, fit or query.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

SOURCE="known_task_recovery_observation_audit_20261006_v5_motion_typed_support_full"
LABELS=SOURCE+"_labels_v3_perspective"
SEPARATION_SHA="f399824634fbbe3c4e60bb494a5f21b9135b139cff71c1abf8520a6c16a2110e"
DEADLINES=(4,8,12,16)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):h.update(part)
    return h.hexdigest()


def table(root):
    rows={}
    for line in (root/"SHA256SUMS.txt").read_text().splitlines():
        digest,name=line.split("  ",1)
        if name in rows:raise ValueError("Duplicate source manifest reference")
        rows[name]=digest
    return rows


def read_verified(root,path,manifest):
    relative=str(path.relative_to(root))
    if relative not in manifest or sha(path)!=manifest[relative]:raise ValueError("Evidence SHA mismatch: "+relative)
    return json.loads(path.read_text())


def physical_false(atom):
    return atom.get("measurement_valid") is True and atom.get("value") is False


def certified_positive(atom):
    return (atom.get("measurement_valid") is True and atom.get("physical_value") is True and
        atom.get("new_mask") is True and atom.get("deployment_allowed") is False)


def continuous_identity_window(cert,step,view):
    """Frozen same-view target/EEF identities + valid actual-RGB motion span."""
    if step<4:return False
    window=list(range(step-2,step+1))
    for s in window:
        frame=cert["frame_certificates"][s]
        v=frame.get("views",{}).get(view,{})
        if (frame.get("step")!=s or v.get("registered_actual") is not True or
            any(v.get("roles",{}).get(role,{}).get("identity_certified") is not True for role in ("target","eef"))):return False
    for row in cert.get("motion_certificates",[]):
        m=row.get("certificate",{});start=m.get("start_step")
        if (m.get("value") is True and m.get("reference_role")=="eef" and m.get("view")==view and
            m.get("end_step")==step and type(start) is int and 2<=start<=step-2 and
            m.get("used_steps")==list(range(start,step+1)) and m.get("deployment_allowed") is False):return True
    return False


def separation_measurement(rgb,tm,em,view,sep):
    """Generic target/EEF gap measurement; NEVER call finger-negative API."""
    roles=view.get("roles",{});repeat=view.get("repeat_noise",{})
    result=dict(eligible=False,physical_label_created=False,certifies_held_false=False)
    if view.get("fresh_geometry_registered_to_actual_input") is not True:
        return result|dict(reason="fresh_geometry_unregistered")
    if any(not sep._registration(roles.get(role,{}).get("local_rgb")) for role in ("target","eef")):
        return result|dict(reason="own_role_local_RGB_registration_failed")
    noises=[];rgb_noises=[]
    for role in ("target","eef"):
        r=repeat.get(role,{})
        if (not sep._finite(r.get("centroid_noise_px")) or r["centroid_noise_px"]<0 or
            not sep._registration(r.get("rgb_repeat")) or r["rgb_repeat"].get("passed") is not True):
            return result|dict(reason="same_candidate_own_role_repeat_noise_missing")
        noises.append(r["centroid_noise_px"]);rgb_noises.append(r["rgb_repeat"]["mean_abs"])
    if tm.sum()<16 or em.sum()<16 or np.any(tm&em):return result|dict(reason="complete_role_masks_missing_or_overlapping")
    if any(np.ptp(rgb[mask].astype(float),axis=0).max()<=0 for mask in (tm,em)):
        return result|dict(reason="own_role_actual_RGB_structure_missing")
    floor=max(2.,3.*max(rgb_noises));boundary=[sep._boundary_witness(rgb.astype(float),mask,floor) for mask in (tm,em)]
    if min(map(len,boundary))<2:return result|dict(reason="both_actual_RGB_silhouettes_not_verified")
    gap=sep._minimum_gap(sep._mask_boundary(tm),sep._mask_boundary(em));noise=max(noises)
    enough=gap>=2. and gap>3.*noise
    return result|dict(eligible=bool(enough),reason="eligible_separation_measurement_only" if enough else "gap_below_frozen_floor_or_noise",
        complete_target_EEF_gap_px=gap,same_candidate_noise_px=noise,actual_target_boundary_pixels=len(boundary[0]),
        actual_EEF_boundary_pixels=len(boundary[1]),contrast_floor=floor,EEF_is_rendered_finger_union_not_complete_hand=True)


def deadline_counts(events):
    first={}
    for e in events:
        key=(e["pool_id"],e["candidate_id"],e["split"])
        first[key]=min(first.get(key,17),e["step"])
    return {str(d):sum(s<=d for s in first.values()) for d in DEADLINES}


def run(root,*,include_unresolved_contact_audit=False):
    root=Path(root).resolve();source=root/"outputs"/SOURCE;labels=root/"outputs"/LABELS
    sm,lm=table(source),table(labels);records=read_verified(source,source/"records.json",sm)
    old_audit=read_verified(labels,labels/"observability_label_audit.json",lm)
    originals=read_verified(source,source/"source_sha256.json",sm)
    helper=root/"research_runs/known_task_recovery_observation_audit_20261006_v5_motion_typed_support/observable_separation.py"
    if not helper.is_file():helper=root/"wam_reranking/observable_separation.py"
    if sha(helper)!=SEPARATION_SHA or old_audit["negative_interface_extractor_sha256"]!=SEPARATION_SHA:
        raise ValueError("Only frozen separation primitive source may be measured")
    spec=importlib.util.spec_from_file_location("_frozen_separation_measurement",helper);sep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sep)
    if len(records)!=160:raise ValueError("Only existing full160 scope allowed")
    data={};pool_splits={};source_files={};positive=defaultdict(list);physical_positive_cids=set();positive_cids=set()
    for r in records:
        pool,cid,split=r["pool_id"],r["candidate_id"],r["split"]
        if (pool,cid) in data or cid not in range(4) or pool_splits.setdefault(pool,split)!=split:
            raise ValueError("Duplicate candidate or pool split mismatch")
        directory=source/r["directory"]
        teacher=read_verified(source,directory/"temporal_teacher.json",sm)
        measurement=read_verified(source,directory/"temporal_measurements.json",sm)
        certpath=labels/"certificates"/pool/f"candidate_{cid}"/"observability_certificates.json"
        cert=read_verified(labels,certpath,lm)
        parts=pool.split("_",2)
        expected_identity=dict(dataset="known_task_recovery_v9_20261006_v5_paired_recovery",suite="libero90",
            task=int(parts[0].removeprefix("task")),state=int(parts[1].removeprefix("state")),
            candidate_id=cid,observed_block_id=pool+"/first_block")
        if teacher["identity"]!=cert["identity"] or teacher["split"]!=split or cert["split"]!=split:
            raise ValueError("Source-scoped candidate certificate identity mismatch")
        if teacher["identity"]!=expected_identity:raise ValueError("Dataset/suite/task/state/block membership mismatch")
        if teacher["identity"]["candidate_id"]!=cid or len(teacher["frames"])!=17 or len(cert["labels"])!=17:
            raise ValueError("Finite temporal identity/shape mismatch")
        data[(pool,cid)]=(r,directory,teacher,measurement,cert)
        for step in range(1,17):
            atom=measurement["frames"][step]["physics"].get("carried_sufficient_evidence",{})
            if atom.get("measurement_valid") and atom.get("value") is True:physical_positive_cids.add((pool,cid))
            if certified_positive(cert["labels"][step]["atoms"]["carried_sufficient_evidence"]):
                positive[(pool,split,step)].append(cid);positive_cids.add((pool,cid))
    if len(pool_splits)!=40 or any({cid for p,cid in data if p==pool}!=set(range(4)) for pool in pool_splits):
        raise ValueError("Original complete forty 4-candidate pools required")
    if len(physical_positive_cids)!=76 or len(positive_cids)!=71:raise ValueError("Frozen carrying counts changed")
    stage=Counter();reasons=Counter();events=[];base_false=[];both_no_contacts=[];source_refs=[];unresolved=[];inventory=Counter()
    for (pool,cid),(r,directory,teacher,measurement,cert) in data.items():
        split=r["split"];false_steps=[s for s in range(1,17) if physical_false(measurement["frames"][s]["physics"].get("carried_sufficient_evidence",{}))]
        for s in range(1,17):
            atom=measurement["frames"][s]["physics"].get("carried_sufficient_evidence",{})
            inventory[str(atom.get("value"))+":"+str(atom.get("measurement_valid"))+":"+teacher["kind"]]+=1
        unresolved_steps=[]
        if include_unresolved_contact_audit and teacher["kind"]=="rigid":
            for s in range(1,17):
                atom=measurement["frames"][s]["physics"].get("carried_sufficient_evidence",{})
                phys=teacher["frames"][s]["physics"]
                if (atom.get("value") is None and phys.get("left_finger_target_contact") is False and
                    phys.get("right_finger_target_contact") is False):unresolved_steps.append(s)
        stage["physical_carry_FALSE_candidate_steps"]+=len(false_steps)
        if false_steps:stage["physical_carry_FALSE_candidates"]+=1
        original=root/teacher["source_folder"]
        segpath=directory/"physical_teacher_arrays.npz"
        seg=None;rgb={};cache={}
        stage["unresolved_rigid_with_current_two_measured_no_contacts_candidate_steps"]+=len(unresolved_steps)
        for step in sorted(set(false_steps+unresolved_steps)):
            descriptor=dict(pool_id=pool,candidate_id=cid,split=split,step=step)
            if step in false_steps:base_false.append(descriptor)
            if step in unresolved_steps:unresolved.append(descriptor)
            phys=teacher["frames"][step]["physics"]
            if phys.get("left_finger_target_contact") is False and phys.get("right_finger_target_contact") is False:
                both_no_contacts.append(dict(pool_id=pool,candidate_id=cid,split=split,step=step))
            reliable_views=[v for v in ("primary","wrist") if continuous_identity_window(cert,step,v)]
            if not reliable_views:continue
            stage["continuous_same_view_identity_candidate_steps"]+=1
            if seg is None:
                rel=str(segpath.relative_to(source))
                if sha(segpath)!=sm[rel]:raise ValueError("Segmentation measurement changed")
                with np.load(segpath,allow_pickle=False) as a:seg={v:a[v] for v in ("primary","wrist")}
                for v in ("primary","wrist"):
                    rawpath=original/(v+".npy");ref=str(rawpath.relative_to(root));digest=sha(rawpath)
                    if originals.get(ref)!=digest or cert["evidence_sha256"]["actual_"+v]!=digest:
                        raise ValueError("Original actual RGB changed")
                    rgb[v]=np.load(rawpath,allow_pickle=False);source_files[ref]=digest
                    if seg[v].shape!=(17,256,256) or rgb[v].shape!=(17,256,256,3):raise ValueError("Temporal RGB/role segmentation mismatch")
                source_files[str(segpath.relative_to(root))]=sm[rel]
            endpoint=[];persistent=[]
            for view in reliable_views:
                for s in range(step-2,step+1):
                    key=(s,view)
                    if key not in cache:
                        ev=teacher["frames"][s]["observation_evidence"];mapping=ev["private_collision_vs_visual_mapping"]
                        ids={name:{x["geom_id"] for x in mapping[name]} for name in ("target","left","right")}
                        tm=np.isin(seg[view][s],list(ids["target"]));em=np.isin(seg[view][s],list(ids["left"]|ids["right"]))
                        cache[key]=separation_measurement(rgb[view][s],tm,em,ev["views"][view],sep)
                    reasons[cache[key]["reason"]]+=1
                if cache[(step,view)]["eligible"]:endpoint.append(view)
                if all(cache[(s,view)]["eligible"] for s in range(step-2,step+1)):persistent.append(view)
            if endpoint:stage["endpoint_separation_candidate_steps"]+=1
            if not persistent:continue
            stage["persistent_three_frame_separation_candidate_steps"]+=1
            no_contacts=all(teacher["frames"][s]["physics"].get("left_finger_target_contact") is False and
                teacher["frames"][s]["physics"].get("right_finger_target_contact") is False for s in range(step-2,step+1))
            peers=positive.get((pool,split,step),[])
            events.append(dict(pool_id=pool,candidate_id=cid,split=split,step=step,views=persistent,
                physical_carry_value=measurement["frames"][step]["physics"]["carried_sufficient_evidence"].get("value"),
                carry_state_eligibility="measured_FALSE" if step in false_steps else "unresolved_None_with_raw_two_no_contacts",
                physical_carry_FALSE_or_None_is_not_held_false=True,
                both_contacts_absent_all_three_frames=no_contacts,
                same_pool_step_certified_carry_positive_peers=peers,
                endpoint_measurements={v:cache[(step,v)] for v in persistent},
                admission="not_admitted",held_false_proved=False,old_masks_unchanged=True))
        source_refs.append(dict(pool_id=pool,candidate_id=cid,source_identity=teacher["identity"],split=split,
            teacher_sha256=sm[str((directory/"temporal_teacher.json").relative_to(source))],
            physical_measurements_sha256=sm[str((directory/"temporal_measurements.json").relative_to(source))],
            frozen_certificate_sha256=lm[str((labels/"certificates"/pool/f"candidate_{cid}"/"observability_certificates.json").relative_to(labels))]))
    paired=[e for e in events if e["same_pool_step_certified_carry_positive_peers"]]
    strict=[e for e in events if e["both_contacts_absent_all_three_frames"]]
    paired_strict=[e for e in strict if e["same_pool_step_certified_carry_positive_peers"]]
    def count_es(es):return dict(candidate_steps=len(es),candidates=len({(e["pool_id"],e["candidate_id"]) for e in es}),
        pools=len({e["pool_id"] for e in es}),deadline_candidate_counts=deadline_counts(es))
    per_split={s:{"reusable_separation":count_es([e for e in events if e["split"]==s]),
        "paired_strict_no_contacts":count_es([e for e in paired_strict if e["split"]==s])} for s in ("train","val")}
    contexts=defaultdict(Counter)
    for e in events:contexts[e["pool_id"].split("_",2)[2]]["separation_candidate_steps"]+=1
    for e in paired_strict:contexts[e["pool_id"].split("_",2)[2]]["paired_strict_candidate_steps"]+=1
    return dict(schema="existing_held_separation_evidence_diagnostic_v1",source=SOURCE,labels_source=LABELS,
        completed_candidates=160,pools=40,frozen_carrying_positive_candidates=dict(physical=76,certified=71),
        stages=dict(stage),physical_carry_value_validity_kind_inventory=dict(inventory),physical_carry_false=count_es(base_false),
        unresolved_rigid_with_current_two_measured_no_contacts=count_es(unresolved),both_contacts_absent_current=count_es(both_no_contacts),
        reusable_separation_evidence=count_es(events),same_pool_step_positive_vs_separation=count_es(paired),
        separation_plus_three_frame_both_contacts_absent=count_es(strict),same_pool_step_strict_pairs=count_es(paired_strict),
        by_split=per_split,by_context={k:dict(v) for k,v in contexts.items()},measurement_failure_counts=dict(reasons),
        detailed_eligible_separation_events=events,source_candidate_references=source_refs,actually_read_array_sha256=source_files,
        source_manifest_sha256=sha(source/"SHA256SUMS.txt"),labels_manifest_sha256=sha(labels/"SHA256SUMS.txt"),
        separation_primitive_sha256=SEPARATION_SHA,frozen_numeric_thresholds=dict(min_gap_px=2.,min_role_pixels=16,
            actual_boundary_pixels_per_role=2,repeat_noise_multiplier=3.,RGB_mean_abs=2.,RGB_p95=8.,RGB_fraction_gt5=.06,PSNR=30.),
        held_FALSE_proved_candidates=0,new_supervision_admitted=False,old_labels_and_masks_unchanged=True,
        unresolved_contact_diagnostic_enabled=include_unresolved_contact_audit,unresolved_None_not_converted_to_FALSE=True,
        training_started=False,new_replay_or_queries=0,deployment_features_created=False,
        limitations=["carried_sufficient_evidence_FALSE_is_not_held_FALSE","gap_between_target_and_rendered_finger_union_is_not_complete_hand_geometry",
            "frozen_finger_negative_contact_API_not_renamed_or_called_with_EEF_side","current_or_persistent_gap_measurement_is_Y_only_and_not_force_closure",
            "new_narrow_held_at_use_predicate_contract_requires_separate_approval_and_exclusion_conditions"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--compact",action="store_true")
    p.add_argument("--include-unresolved-contact-audit",action="store_true");args=p.parse_args()
    result=run(args.root,include_unresolved_contact_audit=args.include_unresolved_contact_audit)
    if args.compact:
        events=result.pop("detailed_eligible_separation_events");first={}
        for e in events:
            key=(e["pool_id"],e["candidate_id"],e["split"])
            if key not in first:first[key]=e
        result["first_eligible_separation_per_candidate"]=list(first.values())
        refs=result.pop("source_candidate_references");arrays=result.pop("actually_read_array_sha256")
        result["verified_source_candidate_references_count"]=len(refs)
        result["verified_candidate_reference_table_sha256"]=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest()
        result["actually_read_array_count"]=len(arrays)
        result["actually_read_array_sha_table_sha256"]=hashlib.sha256(json.dumps(arrays,sort_keys=True).encode()).hexdigest()
        result["compact_report_full_per_step_measurements_reproducible_from_frozen_source"]=True
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
