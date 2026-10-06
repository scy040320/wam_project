"""Finite, source-frozen historical release geometry diagnostics; no labels.

The endpoint trace reproduces the exact frozen adapter and compares outputs.
It only explains rejection, retaining 1 mm depth / 1 px LK/material thresholds.
No projected simulator point certifies real RGB identity or a recovery event.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCHEMA = "archived_release_geometry_rejection_diagnostic_v1"
VERSION = "known_task_archived_release_geometry_diagnostic_20261006_v1"
PRODUCER_SHA = "ab2f285046612fddb04e9ce0faf29a2871d78daa4ab5d2f505e9aec6a3f8895b"
TAIL_SHA = "bcd446ad73dc767ca7cf82dcfc532ceea7bf89d8217210f64328dc2e06fb08f4"
WINDOWS = ((65, 66), (65, 67), (66, 67))
CAPTURE_INDICES = (57, 65, 66, 67)
ROLES = ("target", "eef", "left", "right")
DEPTH_TOLERANCE_M = .001
LK_TOLERANCE_PX = 1.


def json_ready(value):
    if isinstance(value, dict): return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(x) for x in value]
    if isinstance(value, np.ndarray): return json_ready(value.tolist())
    if isinstance(value, np.generic): return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def role_membership(points, mask):
    p = np.asarray(points, float).reshape(-1, 2); mask = np.asarray(mask, bool)
    h, w = mask.shape; finite = np.isfinite(p).all(axis=1)
    ok = finite & (p[:, 0] >= 0.) & (p[:, 0] < w) & (p[:, 1] >= 0.) & (p[:, 1] < h)
    idx = np.full(p.shape, -1, np.int64); idx[ok] = np.floor(p[ok] + .5).astype(int)
    ok &= (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
    selected = np.zeros(len(p), bool); selected[ok] = mask[idx[ok, 1], idx[ok, 0]]
    return selected, idx


def trace_material_points(start, end, view, role, corners, project):
    """Explain exact frozen nearest-depth rigid-body material-point transform."""
    a, b = start[view], end[view]; params = a["camera"]
    p = np.asarray(corners, float).reshape(-1, 2); h, w = params["image_shape"]
    c = params["image_convention"]; f = h / (2 * np.tan(np.deg2rad(params["fovy_degrees"]) / 2))
    start_role, idx = role_membership(p, a["roles"][role])
    bounds = (idx[:, 0] >= 0) & (idx[:, 0] < w) & (idx[:, 1] >= 0) & (idx[:, 1] < h)
    world = np.full((len(p), 3), np.nan); moved = world.copy()
    gid = np.full(len(p), -1, int); bid = gid.copy(); depth = np.full(len(p), np.nan)
    start_valid = bounds.copy()
    expected_body_ids = sorted(set(int(a["geom_body_ids"][g]) for g in
        np.unique(a["geom"][a["roles"][role]]) if 0 <= int(g) < len(a["geom_body_ids"])))
    for i, (x, y) in enumerate(idx):
        if not bounds[i]: continue
        gid[i] = int(a["geom"][y, x]); depth[i] = float(a["depth"][y, x])
        if gid[i] < 0 or gid[i] >= len(a["geom_body_ids"]) or not np.isfinite(depth[i]) or depth[i] <= 0.:
            start_valid[i] = False; continue
        bid[i] = int(a["geom_body_ids"][gid[i]])
        if bid[i] < 0 or bid[i] >= len(a["body_pos"]): start_valid[i] = False; continue
        ycv = h - 1 - p[i, 1] if c == -1 else p[i, 1]
        cv = np.array([(p[i, 0] - w / 2) * depth[i] / f, (ycv - h / 2) * depth[i] / f, depth[i]])
        world[i] = params["camera_rotation_world_from_camera"] @ (cv * np.array([1, -1, -1])) + params["camera_position_m"]
        local = a["body_rot"][bid[i]].T @ (world[i] - a["body_pos"][bid[i]])
        moved[i] = b["body_rot"][bid[i]] @ local + b["body_pos"][bid[i]]
    projected = project(moved, **b["camera"]); stationary = project(world, **b["camera"])
    roundtrip = project(world, **a["camera"])
    valid = start_valid & projected["valid"] & stationary["valid"]
    end_role, end_idx = role_membership(projected["pixel_xy"], b["roles"][role])
    residual = np.full(len(p), np.nan); end_gid = np.full(len(p), -1, int)
    end_bid = end_gid.copy(); end_surface = np.full(len(p), np.nan)
    for i in np.flatnonzero(valid):
        x, y = end_idx[i]
        if x < 0 or x >= w or y < 0 or y >= h: valid[i] = False; continue
        end_surface[i] = float(b["depth"][y, x]); end_gid[i] = int(b["geom"][y, x])
        if 0 <= end_gid[i] < len(b["geom_body_ids"]): end_bid[i] = int(b["geom_body_ids"][end_gid[i]])
        residual[i] = float(end_surface[i] - projected["camera_depth_m"][i])
        # Exact frozen code used this comparison; invalid depth is separately
        # exposed, not silently converted into valid supervision.
        if abs(residual[i]) > DEPTH_TOLERANCE_M: valid[i] = False
    reasons = []
    for i in range(len(p)):
        r = []
        if not start_valid[i]: r.append("start_depth_or_geom_invalid")
        if not start_role[i]: r.append("start_corner_not_in_own_role")
        if bid[i] not in expected_body_ids: r.append("start_role_body_binding_mismatch")
        if not projected["valid"][i]: r.append("material_projection_outside_or_behind_camera")
        if not stationary["valid"][i]: r.append("camera_only_projection_outside_or_behind_camera")
        if np.isfinite(residual[i]) and abs(residual[i]) > DEPTH_TOLERANCE_M: r.append("material_endpoint_surface_depth_error_gt_1mm")
        if projected["valid"][i] and not end_role[i]: r.append("material_endpoint_not_in_own_rendered_role")
        if projected["valid"][i] and not np.isfinite(end_surface[i]): r.append("end_depth_unavailable_or_nonfinite")
        reasons.append(r)
    return dict(start_pixel_xy=p, end_pixel_xy=projected["pixel_xy"],
        camera_only_end_pixel_xy=stationary["pixel_xy"], valid=valid), dict(
        start_geom_id=gid, start_body_id=bid, expected_role_body_ids=expected_body_ids,
        role_binding_check_kind="derived_role_consistency_only_not_independent_entity_certification",
        start_metric_depth_m=depth, start_depth_and_geom_valid=start_valid,
        start_role_bound=start_role, start_world_m=world, transformed_material_world_m=moved,
        start_roundtrip_error_px=np.linalg.norm(roundtrip["pixel_xy"]-p, axis=1),
        projected_depth_m=projected["camera_depth_m"], sampled_end_depth_m=end_surface,
        end_surface_minus_material_depth_m=residual, end_geom_id=end_gid, end_body_id=end_bid,
        material_projection_valid=projected["valid"], camera_only_projection_valid=stationary["valid"],
        projected_endpoint_role_bound=end_role, frozen_surface_valid=valid, rejection_reasons=reasons)


def actual_lk(start, end, sm, em):
    import cv2
    a = cv2.cvtColor(np.rint(start).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.rint(end).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(a, maxCorners=100, qualityLevel=.01,
        minDistance=3., mask=np.asarray(sm, np.uint8)*255, blockSize=3)
    if corners is None: return None
    p0 = np.asarray(corners, np.float32).reshape(-1, 2)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01)
    p1,sf,_=cv2.calcOpticalFlowPyrLK(a,b,p0.reshape(-1,1,2),None,winSize=(21,21),maxLevel=3,criteria=criteria)
    if p1 is None or sf is None: return dict(start_pixel_xy=p0, reason="forward_tracking_failed")
    p1=np.asarray(p1,np.float32).reshape(-1,2)
    back,sb,_=cv2.calcOpticalFlowPyrLK(b,a,p1.reshape(-1,1,2),None,winSize=(21,21),maxLevel=3,criteria=criteria)
    if back is None or sb is None: return dict(start_pixel_xy=p0,end_pixel_xy=p1,reason="backward_tracking_failed")
    back=np.asarray(back,np.float32).reshape(-1,2);fb=np.linalg.norm(back-p0,axis=1)
    own_start,_=role_membership(p0,sm);own_end,_=role_membership(p1,em)
    keep=np.asarray(sf).reshape(-1).astype(bool)&np.asarray(sb).reshape(-1).astype(bool)
    keep &= np.isfinite(p1).all(axis=1)&np.isfinite(back).all(axis=1)&np.isfinite(fb)&(fb<=LK_TOLERANCE_PX)&own_start&own_end
    return dict(start_pixel_xy=p0,end_pixel_xy=p1,forward_backward_error_px=fb,
        forward_status=np.asarray(sf).reshape(-1).astype(bool),backward_status=np.asarray(sb).reshape(-1).astype(bool),
        start_own_role=own_start,end_own_role=own_end,actual_rgb_proxy_keep=keep,
        actual_rgb_proxy_count=int(keep.sum()),reason="measured_only")


def diagnostic(start, end, view, role, geometry, frozen_endpoints):
    traces=[];raw=actual_lk(start[view]["actual"],end[view]["actual"],start[view]["roles"][role],end[view]["roles"][role])
    def callback(corners):
        exact=frozen_endpoints(start,end,view,role)(corners)
        recreated,trace=trace_material_points(start,end,view,role,corners,geometry.project_world_points)
        for key in ("start_pixel_xy","end_pixel_xy","camera_only_end_pixel_xy"):
            if not np.allclose(exact[key],recreated[key],rtol=0.,atol=1e-10,equal_nan=True):
                raise ValueError("Diagnostic endpoint reproduction differs from frozen adapter")
        if not np.array_equal(exact["valid"],recreated["valid"]):raise ValueError("Diagnostic valid mask differs from frozen adapter")
        if raw and "end_pixel_xy" in raw and np.array_equal(corners,raw["start_pixel_xy"]):
            error=np.linalg.norm(raw["end_pixel_xy"]-exact["end_pixel_xy"],axis=1)
            trace["actual_rgb_lk"]=raw;trace["material_vs_actual_lk_endpoint_error_px"]=error
            for i,r in enumerate(trace["rejection_reasons"]):
                if not raw["actual_rgb_proxy_keep"][i]:r.append("actual_LK_FB_or_own_role_failed")
                if np.isfinite(error[i]) and error[i]>LK_TOLERANCE_PX:r.append("material_point_vs_actual_LK_error_gt_1px")
        traces.append(trace);return exact
    report=geometry.track_rgb_role(start[view]["actual"],end[view]["actual"],start[view]["roles"][role],end[view]["roles"][role],projected_endpoints=callback)
    counts={}
    for t in traces:
        for reasons in t["rejection_reasons"]:
            for reason in reasons:counts[reason]=counts.get(reason,0)+1
    return json_ready(dict(frozen_track_report=report,raw_actual_LK=raw,point_traces=traces,
        rejection_reason_counts=counts,depth_tolerance_m=DEPTH_TOLERANCE_M,material_LK_tolerance_px=LK_TOLERANCE_PX,
        threshold_changed=False,label_created=False,actual_RGB_proxy_is_not_geometry_certificate=True))


def run(root, out, producer, archive, manifest):
    code=Path(__file__).resolve().parent;tail=producer.tail
    geometry=tail.import_verified("observation_geometry",code/"observation_geometry.py",producer.GEOMETRY_SHA)
    adapter=tail.import_verified("_release_diagnostic_adapter",code/"collect_recovery_observation_evidence.py",producer.ADAPTER_SHA)
    base=archive.import_file("_release_diagnostic_base",archive.verified_base_path(code));runtime=adapter.attach(base,root)
    endpoints=producer.verified_endpoints(base,code/"collect_recovery_observation_evidence.py")
    audit=tail.read(root/"outputs"/tail.PREVIOUS/"pilot_audit.json")
    fork=tail.import_verified("_release_diagnostic_snapshot",root/"research_runs"/archive.REFERENCE/"collect_snapshot_fork.py",audit["snapshot_loader_sha256"])
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    import pickle
    env=None;summaries=[]
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
            histories=[];runs=[]
            for repeat in range(2):
                runtime["frames"]=[];prefix=archive.run_chain(env,fork,base,binding,snapshot,row)
                short=tail.replay_tail(env,base,binding,row,prefix);runs.append((prefix,short))
                histories.append(producer.aligned_history(row,prefix,short,runtime["frames"]))
            h=histories[0];qc=tail.repeat_check(dict(frames=h["frames"],qpos=h["qpos"],qvel=h["qvel"],terminal_steps=runs[0][1]["terminal_steps"]),
                dict(frames=histories[1]["frames"],qpos=histories[1]["qpos"],qvel=histories[1]["qvel"],terminal_steps=runs[1][1]["terminal_steps"]))
            dest=out/"chains"/row["scenario"];dest.mkdir(parents=True,exist_ok=False)
            arrays={"chain_indices":np.asarray(CAPTURE_INDICES),"proprio":h["proprio"][list(CAPTURE_INDICES)],
                "qpos":h["qpos"][list(CAPTURE_INDICES)],"qvel":h["qvel"][list(CAPTURE_INDICES)]}
            for v in ("primary","wrist"):
                for name in ("actual","rendered","depth","geom","body_pos","body_rot","geom_body_ids"):
                    arrays[v+"_"+name]=np.stack([h["caches"][i][v][name] for i in CAPTURE_INDICES])
                for role in ROLES:arrays[v+"_role_"+role]=np.stack([h["caches"][i][v]["roles"][role] for i in CAPTURE_INDICES])
            np.savez_compressed(dest/"relevant_geometry.npz",**arrays)
            camera_and_source=[]
            for i in CAPTURE_INDICES:
                camera_and_source.append(dict(chain_index=i,source_identity=producer.source_identity(h["lineage"][i]),
                    original_image_sources=producer.original_images(row,h["lineage"][i],manifest["source_sha256"]),
                    original_replay_checks=h["checks"][i],cameras={v:h["caches"][i][v]["camera"] for v in ("primary","wrist")},
                    new_replay_measurement_not_original_proprio=True))
            tail.dump(dest/"camera_and_source.json",json_ready(camera_and_source));results=[]
            for start,end in WINDOWS:
                for view in ("primary","wrist"):
                    for role in ROLES:
                        d=diagnostic(h["caches"][start],h["caches"][end],view,role,geometry,endpoints)
                        results.append(dict(start_chain_index=start,end_chain_index=end,view=view,role=role,
                            source_start_identity=producer.source_identity(h["lineage"][start]),source_end_identity=producer.source_identity(h["lineage"][end]),**d))
            tail.dump(dest/"geometry_rejections.json",results)
            summary=dict(scenario=row["scenario"],split=row["split"],same_arm_repeat_qc=qc,windows=list(WINDOWS),
                frozen_geometry_matched=sum(x["frozen_track_report"]["geometry_matched_count"] for x in results),
                actual_RGB_proxy_tracks=sum((x["raw_actual_LK"] or {}).get("actual_rgb_proxy_count",0) for x in results),
                per_window_role_counts=[dict(start=x["start_chain_index"],end=x["end_chain_index"],view=x["view"],role=x["role"],
                    actual_rgb_proxy_count=(x["raw_actual_LK"] or {}).get("actual_rgb_proxy_count",0),
                    geometry_matched_count=x["frozen_track_report"]["geometry_matched_count"],
                    rejection_reason_counts=x["rejection_reason_counts"]) for x in results],
                source_unchanged=True,threshold_changed=False,label_created=False)
            summaries.append(summary);tail.dump(dest/"diagnostic_audit.json",summary)
            archive.verify_source_table(root,manifest["source_sha256"])
            tail.dump(out/"status.json",dict(phase="bounded_diagnostic_running",chains_completed=len(summaries),chains_planned=2))
        tail.dump(out/"diagnostic_audit.json",dict(schema=SCHEMA,technical_replay_passed=True,chains=summaries,
            code_sha256={p.name:tail.sha(p) for p in code.glob("*.py")},training_started=False,labels_changed=False,
            new_policy_queries=0,new_candidate_actions=0,measurement_only=True))
    finally:
        if env is not None:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--contract",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--replay",action="store_true");args=p.parse_args()
    root=args.root.resolve();out=args.output.resolve();code=Path(__file__).resolve().parent
    if out.exists() or out.parent!=root/"outputs":raise ValueError("New independent root/outputs directory required")
    if hashlib.sha256((code/"replay_archived_release_tail.py").read_bytes()).hexdigest()!=TAIL_SHA:
        raise ValueError("Exact source-frozen tail helper required before any replay")
    import replay_archived_release_tail as tail
    producer=tail.import_verified("_frozen_release_observation_producer",code/"replay_archived_release_observation.py",PRODUCER_SHA)
    archive=tail.import_verified("_frozen_release_archive",code/"replay_archived_release_evidence.py",tail.ARCHIVE_SHA)
    out.mkdir(exist_ok=False)
    try:
        sys.path.insert(0,str(root));manifest=tail.freeze(root,args.contract.resolve(),archive)
        manifest.update(diagnostic_schema=SCHEMA,fixed_windows=list(WINDOWS),capture_indices=CAPTURE_INDICES,
            no_labels_created=True,no_threshold_change=True)
        tail.dump(out/"frozen_membership.json",manifest)
        if args.replay:run(root,out,producer,archive,manifest)
        tail.dump(out/"status.json",dict(phase="bounded_diagnostic_complete" if args.replay else "manifest_frozen",training_started=False))
    except Exception as error:
        tail.dump(out/"failure.json",dict(error_type=type(error).__name__,message=str(error),source_retained=True));raise
    finally:
        files={str(f.relative_to(out)):tail.sha(f) for f in out.rglob("*") if f.is_file() and f.name!="SHA256SUMS.txt"}
        (out/"SHA256SUMS.txt").write_text("".join(f"{digest}  {name}\n" for name,digest in sorted(files.items())),encoding="utf-8")


if __name__=="__main__":main()
