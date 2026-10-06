"""Verified replay teacher sidecars; no Cosmos query, fitting or deployment GT.

Reuses immutable first-block actions and runtime snapshots. Every aligned
post-action RGB/proprio frame must match the original frozen tolerances.
Physical state/instance IDs/segmentation remain supervision-only. No episode
success, predicted image or attribution is used to derive teacher evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
import types
import shutil

import numpy as np

try:
    from temporal_recovery_teacher import derive_temporal_recovery
except ModuleNotFoundError:
    from wam_reranking.temporal_recovery_teacher import derive_temporal_recovery

SOURCE = 'known_task_recovery_v9_20261006_v5_paired_recovery'
REFERENCE = 'candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat'
BINDINGS = {
    0: ('wooden_cabinet_1_top_level', 'wooden_cabinet_1', 'articulated', 'm'),
    9: ('akita_black_bowl_1', 'plate_1', 'rigid', None),
    46: ('alphabet_soup_1', 'basket_1', 'rigid', None),
    57: ('cream_cheese_1', 'wooden_tray_1', 'rigid', None),
}
PILOT = {(0,31,'normal'), (0,31,'execution_contact_deviation'),
         (9,10,'object_shift'), (9,10,'execution_contact_deviation'),
         (46,10,'normal'), (46,10,'execution_contact_deviation'),
         (57,10,'visual_occlusion'), (57,10,'unknown')}
MIN_VISIBLE_PIXELS = 16


def decode_segmentation_rgb(rgb, scene):
    """Original robosuite ID decode, promoted before arithmetic for NumPy 2."""
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
        raise ValueError('Segmentation encoding must be HxWx3 uint8')
    values=image.astype(np.uint32)
    indices=values[:,:,0]+values[:,:,1]*256+values[:,:,2]*65536
    indices[indices>=scene.ngeom+1]=0
    ids=np.full((scene.ngeom+1,2),-1,dtype=np.int32)
    for i in range(scene.ngeom):
        geom=scene.geoms[i]
        if geom.segid!=-1:ids[geom.segid+1]=[geom.objtype,geom.objid]
    return ids[indices]


def install_scoped_segmentation_decoder(env):
    """Only this environment context changes; installed robosuite stays intact."""
    context=env.sim._render_context_offscreen
    if hasattr(context,'_temporal_original_read_pixels'):return
    original=context.read_pixels;context._temporal_original_read_pixels=original
    def reader(self,width,height,depth=False,segmentation=False):
        result=original(width,height,depth=depth,segmentation=False)
        if not segmentation:return result
        if depth:
            rgb,dep=result;return decode_segmentation_rgb(rgb,self.scn),dep
        return decode_segmentation_rgb(result,self.scn)
    context.read_pixels=types.MethodType(reader,context)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def image_check(actual, expected):
    if actual.shape != expected.shape: raise ValueError('Image shape mismatch')
    d=np.asarray(actual,float)-np.asarray(expected,float);v=np.abs(d);mse=float((d*d).mean())
    record=dict(mean_abs=float(v.mean()),p95=float(np.quantile(v,.95)),
                fraction_gt5=float(np.mean(v>5)),psnr=99. if mse==0 else float(20*np.log10(255/np.sqrt(mse))))
    record['passed']=record['mean_abs']<=2 and record['p95']<=8 and record['fraction_gt5']<=.06 and record['psnr']>=30
    return record


def body_descendants(model, root):
    result=set()
    for bid in range(model.nbody):
        node=bid
        while node>0 and node!=root: node=int(model.body_parentid[node])
        if node==root: result.add(bid)
    return result


def bind(env, task):
    inner=env.env;model=env.sim.model;target,anchor,kind,unit=BINDINGS[task]
    if kind=='articulated':
        jid=int(model.joint_name2id(target));target_body=int(model.jnt_bodyid[jid])
        qadr=int(model.jnt_qposadr[jid]);joint_range=np.asarray(model.jnt_range[jid]).tolist()
    else:
        target_body=int(inner.obj_body_id[target]);jid=qadr=None;joint_range=None
    anchor_body=int(inner.obj_body_id[anchor])
    bodies=body_descendants(model,target_body)
    target_geoms={g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}
    anchor_bodies=body_descendants(model,anchor_body)
    anchor_geoms={g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in anchor_bodies}-target_geoms
    important=inner.robots[0].gripper.important_geoms
    left={int(model.geom_name2id(g)) for g in important['left_finger']}
    right={int(model.geom_name2id(g)) for g in important['right_finger']}
    pads={int(model.geom_name2id(g)) for key in ('left_fingerpad','right_fingerpad') for g in important[key]}
    if not all((target_geoms,left,right,pads)): raise RuntimeError('Incomplete source-scoped physical binding')
    world_geoms={g for g in range(model.ngeom) if not str(model.geom_id2name(g) or '').startswith(('robot','gripper'))}
    return dict(target=target,anchor=anchor,kind=kind,joint_unit=unit,
                target_body=target_body,anchor_body=anchor_body,target_geoms=target_geoms,
                anchor_geoms=anchor_geoms,left=left,right=right,pads=pads,qadr=qadr,
                joint_range=joint_range,world_geoms=world_geoms)


def capture(env, raw, actual, binding, step, convention):
    sim=env.sim;model=sim.model;b=binding
    pairs=[(int(sim.data.contact[i].geom1),int(sim.data.contact[i].geom2)) for i in range(sim.data.ncon)]
    def touches(a,z): return any((x in a and y in z) or (y in a and x in z) for x,y in pairs)
    target=b['target_geoms'];finger=b['left']|b['right']
    physics=dict(eef_pos_m=np.asarray(raw['robot0_eef_pos'],float).tolist(),
        target_pos_m=np.asarray(sim.data.body_xpos[b['target_body']],float).tolist(),
        anchor_pos_m=np.asarray(sim.data.body_xpos[b['anchor_body']],float).tolist(),
        target_quat_wxyz=np.asarray(sim.data.body_xquat[b['target_body']],float).tolist(),
        gripper_aperture_m=float(np.abs(np.asarray(raw['robot0_gripper_qpos'])).sum()),
        left_finger_target_contact=touches(target,b['left']),right_finger_target_contact=touches(target,b['right']),
        target_support_contact=touches(target,b['world_geoms']-target),
        target_anchor_contact=touches(target,b['anchor_geoms']))
    if b['qadr'] is not None: physics['target_joint_qpos']=float(sim.data.qpos[b['qadr']])
    views={};segments={};observability={}
    for view, camera, key in [('primary','agentview','agentview_image'),('wrist','robot0_eye_in_hand','robot0_eye_in_hand_image')]:
        seg=np.asarray(sim.render(width=256,height=256,camera_name=camera,segmentation=True))
        # Match robosuite IMAGE_CONVENTION followed by the frozen vertical flip.
        seg=np.flipud(seg[::convention])
        if seg.shape!=(256,256,2): raise RuntimeError('Segmentation shape contract')
        geom=np.where(seg[:,:,0]==5,seg[:,:,1],-1)
        physical=np.flipud(np.asarray(raw[key]))
        check=image_check(actual[view],physical)
        target_pixels=np.isin(geom,list(target));left_pixels=np.isin(geom,list(b['left']));right_pixels=np.isin(geom,list(b['right']))
        counts={name:int(mask.sum()) for name,mask in [('target',target_pixels),('left_finger',left_pixels),('right_finger',right_pixels)]}
        centers={}
        for name,mask in [('target',target_pixels),('left_finger',left_pixels),('right_finger',right_pixels)]:
            ys,xs=np.nonzero(mask);centers[name]=[float(xs.mean()),float(ys.mean())] if len(xs) else None
        views[view]=dict(physical_render_comparison=check,visible_geom_pixels=counts,
            visible_geom_centroids_px=centers,
            target_visible=counts['target']>=MIN_VISIBLE_PIXELS,
            target_and_both_fingers_visible=all(n>=MIN_VISIBLE_PIXELS for n in counts.values()),
            registration_and_input_reliable=check['passed'])
        # Segmentation is private teacher evidence, never a deployment feature.
        segments[view]=geom.astype(np.int32)
        observability[view+'_reliable']=check['passed']
    reliable=[v for v in views.values() if v['registration_and_input_reliable']]
    observability.update(target_visible=any(v['target_visible'] for v in reliable),
        eef_visible=any(v['target_and_both_fingers_visible'] for v in reliable),
        identity_consistent=None,contact_observable=None,cross_view_consistent=None)
    frame=dict(step=step,physics=physics,observability=observability,views=views,
        private_binding_role='exact simulator instance measurement, not visual tracking or contact observability certification',
        contact_geom_pairs=[[model.geom_id2name(x),model.geom_id2name(y)] for x,y in pairs])
    return frame,segments


def replay(env,fork,base,binding,folder,snapshot,cause,*,save_segments=True):
    if not np.array_equal(np.asarray(snapshot['sim_state']),np.load(folder/'exact_start_state.npy',allow_pickle=False)):
        raise RuntimeError('Source candidate and runtime snapshot disagree')
    env.reset();install_scoped_segmentation_decoder(env)
    raw=fork.restore_runtime_snapshot(env,copy.deepcopy(snapshot))
    if not np.array_equal(np.asarray(env.get_sim_state()),snapshot['sim_state']): raise RuntimeError('Exact start snapshot failure')
    from robosuite import macros
    from robosuite.environments.robot_env import IMAGE_CONVENTION_MAPPING
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    primary=np.load(folder/'primary.npy',allow_pickle=False);wrist=np.load(folder/'wrist.npy',allow_pickle=False)
    proprio=np.load(folder/'proprio.npy',allow_pickle=False);actions=np.load(folder/'applied.npy',allow_pickle=False)
    if primary.shape!=(17,256,256,3) or wrist.shape!=primary.shape or proprio.shape!=(17,9) or actions.shape!=(16,7): raise RuntimeError('Original complete temporal arrays required')
    frames=[];checks=[];segments={'primary':[],'wrist':[]};qpos=[];qvel=[];done_seen=False
    for step in range(17):
        if step:
            raw,_,done,_=env.step(np.asarray(actions[step-1],np.float32).tolist());done_seen=done_seen or bool(done)
        physical_raw=copy.deepcopy(raw);observed=raw
        if cause=='visual_occlusion':observed,_=base.InterventionEnv._mask_primary(observed)
        if cause=='unknown':observed,_=base.InterventionEnv._shift_primary_view(observed)
        actual={'primary':primary[step],'wrist':wrist[step]}
        replayed={'primary':np.flipud(observed['agentview_image']), 'wrist':np.flipud(observed['robot0_eye_in_hand_image'])}
        pr=np.concatenate((observed['robot0_gripper_qpos'],observed['robot0_eef_pos'],observed['robot0_eef_quat']))
        check=dict(step=step,primary=image_check(replayed['primary'],primary[step]),wrist=image_check(replayed['wrist'],wrist[step]),
                   proprio_max_abs=float(np.abs(pr-proprio[step]).max()),cached_entry_diagnostic_only=step==0)
        check['passed']=step==0 or (check['primary']['passed'] and check['wrist']['passed'] and check['proprio_max_abs']<=1e-3)
        checks.append(check)
        if not check['passed']: raise RuntimeError('Original trajectory replay mismatch:'+json.dumps(check))
        frame,seg=capture(env,physical_raw,actual,binding,step,convention)
        frames.append(frame);qpos.append(sim_copy(env,'qpos'));qvel.append(sim_copy(env,'qvel'))
        if save_segments:
            for view in segments:segments[view].append(seg[view])
    if done_seen: raise RuntimeError('Original full candidate became terminal during verified replay')
    return frames,checks,segments,np.stack(qpos),np.stack(qvel)


def sim_copy(env,name):return np.asarray(getattr(env.sim.data,name),float).copy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--pilot-only',action='store_true');ap.add_argument('--reuse-pilot',type=Path);a=ap.parse_args()
    root=a.root.resolve();out=a.output.resolve();source=root/'outputs'/SOURCE
    if out.exists() or source==out or source in out.parents: raise RuntimeError('Refuse overwrite or mutate frozen source')
    out.mkdir(parents=True);start=time.time();status=dict(phase='starting',completed_candidates=0,training_started=False)
    dump(out/'status.json',status)
    fork=load('temporal_replay_snapshot',root/'research_runs'/REFERENCE/'collect_snapshot_fork.py')
    base=load('temporal_replay_intervention',root/'research_runs'/REFERENCE/'base_collector_shift_calibrated.py')
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    pools=json.loads((source/'pools.json').read_text());source_audit=json.loads((source/'completion_audit.json').read_text())
    if not source_audit['passed'] or len(pools)!=40:raise RuntimeError('Frozen source audit failed')
    original={};completed=[];qc=[];env=None;task_id=None;reused={};pilot_source=None
    if a.reuse_pilot:
        pilot_source=a.reuse_pilot.resolve()
        if a.pilot_only or pilot_source==out or out in pilot_source.parents:raise RuntimeError('Invalid pilot reuse source')
        previous=json.loads((pilot_source/'completion_audit.json').read_text())
        if not previous['passed'] or previous['completed_candidates']!=32:raise RuntimeError('Pilot reuse audit failed')
        for line in (pilot_source/'SHA256SUMS.txt').read_text().splitlines():
            expected,relative=line.split('  ',1);path=(pilot_source/relative).resolve()
            if pilot_source not in path.parents or sha(path)!=expected:raise RuntimeError('Pilot hash mismatch')
        for row in json.loads((pilot_source/'records.json').read_text()):
            reused[(row['pool_id'],row['candidate_id'])]=row
        prior_original=json.loads((pilot_source/'source_sha256.json').read_text())
        for relative,h in prior_original.items():
            path=(root/relative).resolve()
            if root not in path.parents or sha(path)!=h:raise RuntimeError('Frozen source changed since pilot:'+relative)
        original.update(prior_original)
    protocol=dict(schema='verified_temporal_recovery_teacher_v1',inherits='candidate_recovery_labels_v2',
        source=SOURCE,role='offline_supervision_only',new_cosmos_queries=0,new_candidate_actions=0,
        all_40_pools_retained=True,pilot_pools=sorted([list(p) for p in PILOT]),
        exact_snapshot_start=True,post_action_alignment_steps=list(range(1,17)),
        cached_entry_proprio_diagnostic_only=True,all_frozen_replay_thresholds_unchanged=True,
        observability_min_geom_pixels=MIN_VISIBLE_PIXELS,visibility_floor_not_a_model_threshold=True,
        target_finger_same_reliable_view_required=True,physical_teacher_not_deployable=True,
        numpy2_segmentation_rgb_integer_promotion='scoped renderer decoder; no installed package edit',
        no_terminal_success_or_attribution_used_as_teacher=True,no_auto_training=True,
        bindings={str(k):list(v) for k,v in BINDINGS.items()},script_sha256=sha(Path(__file__)),
        reused_pilot=str(pilot_source) if pilot_source else None)
    dump(out/'protocol.json',protocol)
    try:
        for pool in pools:
            identity=(pool['task'],pool['state'],pool['cause'])
            if a.pilot_only and identity not in PILOT: continue
            if task_id!=pool['task']:
                if env is not None: env.close()
                task_id=pool['task'];env,_=get_libero_env(benchmark.get_benchmark_dict()['libero_90']().get_task(task_id),'cosmos',resolution=256)
                env.reset();binding=bind(env,task_id)
            pd=source/'pools'/pool['pool_id'];snapshot=pickle.loads((pd/'snapshot.pkl').read_bytes())
            original[str((pd/'snapshot.pkl').relative_to(root))]=sha(pd/'snapshot.pkl')
            for candidate in pool['candidates']:
                folder=source/candidate['directory'];dest=out/'supervision_only'/pool['pool_id']/f'candidate_{candidate["candidate_id"]}'
                for name in ['primary.npy','wrist.npy','proprio.npy','applied.npy','requested.npy','planned_actions.npy','exact_start_state.npy']:
                    path=folder/name;original[str(path.relative_to(root))]=sha(path)
                key=(pool['pool_id'],candidate['candidate_id'])
                if key in reused:
                    prior=pilot_source/'supervision_only'/pool['pool_id']/f'candidate_{candidate["candidate_id"]}'
                    shutil.copytree(prior,dest)
                    completed.append(dict(pool_id=pool['pool_id'],candidate_id=candidate['candidate_id'],split=pool['split'],directory=str(dest.relative_to(out)),pilot_reused=True))
                    if candidate['candidate_id']==0:qc.append(json.loads((dest/'repeat_qc.json').read_text()))
                    continue
                frames,checks,segments,qpos,qvel=replay(env,fork,base,binding,folder,snapshot,pool['cause'])
                dest.mkdir(parents=True,exist_ok=False)
                arrays={k:np.stack(v) for k,v in segments.items()};arrays.update(qpos=qpos,qvel=qvel)
                np.savez_compressed(dest/'physical_teacher_arrays.npz',**arrays)
                dump(dest/'temporal_teacher.json',dict(role='offline_supervision_only',source_folder=str(folder.relative_to(root)),
                    identity=dict(dataset=SOURCE,suite='libero90',task=pool['task'],state=pool['state'],candidate_id=candidate['candidate_id'],observed_block_id=pool['pool_id']+'/first_block'),
                    split=pool['split'],kind=binding['kind'],joint_unit=binding['joint_unit'],joint_range=binding['joint_range'],frames=frames,
                    segmentation_not_model_input=True,physical_teacher_arrays='physical_teacher_arrays.npz'))
                derived=derive_temporal_recovery(frames,entity_kind=binding['kind'],joint_unit=binding['joint_unit'])
                dump(dest/'temporal_measurements.json',derived)
                dump(dest/'replay_audit.json',dict(passed=True,checks=checks,exact_start=True,source_unchanged=True))
                completed.append(dict(pool_id=pool['pool_id'],candidate_id=candidate['candidate_id'],split=pool['split'],directory=str(dest.relative_to(out))))
                if candidate['candidate_id']==0:
                    repeated,repeated_checks,repeated_segments,rq,rv=replay(env,fork,base,binding,folder,snapshot,pool['cause'])
                    delta=np.abs(np.c_[qpos,qvel]-np.c_[rq,rv])
                    passed=delta.max()<=1e-3 and np.quantile(delta,.95)<=1e-4 and not np.any(delta>1e-3)
                    report=dict(pool_id=pool['pool_id'],passed=bool(passed),max_abs=float(delta.max()),p95=float(np.quantile(delta,.95)),
                        teacher_contact_atoms_match=all(all(x['physics'][name]==y['physics'][name] for name in ('left_finger_target_contact','right_finger_target_contact','target_support_contact','target_anchor_contact')) for x,y in zip(frames,repeated)),
                        first_measured_events_match=derived['first_measured_events']==derive_temporal_recovery(repeated,entity_kind=binding['kind'],joint_unit=binding['joint_unit'])['first_measured_events'],
                        checked_steps=list(range(17)))
                    if not all((report['passed'],report['teacher_contact_atoms_match'],report['first_measured_events_match'])):raise RuntimeError('Repeated physical teacher mismatch:'+json.dumps(report))
                    dump(dest/'repeat_teacher.json',dict(role='offline_repeat_qc_only',frames=repeated,checks=repeated_checks))
                    repeat_arrays={k:np.stack(v) for k,v in repeated_segments.items()};repeat_arrays.update(qpos=rq,qvel=rv)
                    np.savez_compressed(dest/'repeat_teacher_arrays.npz',**repeat_arrays)
                    qc.append(report);dump(dest/'repeat_qc.json',report)
                dump(out/'status.json',dict(phase='pilot' if a.pilot_only else 'replaying',completed_candidates=len(completed),
                    total_candidates=32 if a.pilot_only else 160,current_pool=pool['pool_id'],candidate_id=candidate['candidate_id'],elapsed_seconds=time.time()-start,training_started=False))
                print(json.dumps(dict(completed_candidates=len(completed),pool_id=pool['pool_id'],candidate_id=candidate['candidate_id'])),flush=True)
        expected=32 if a.pilot_only else 160
        if len(completed)!=expected:raise RuntimeError('Frozen cardinality mismatch')
        if len({(r['pool_id'],r['candidate_id']) for r in completed})!=expected:raise RuntimeError('Duplicate source candidate identity')
        for relative,h in original.items():
            if sha(root/relative)!=h:raise RuntimeError('Original evidence mutated:'+relative)
        dump(out/'source_sha256.json',original);dump(out/'records.json',completed)
        report=dict(passed=True,role='verified_replay_teacher_not_original_recorded_sim_trace',completed_candidates=len(completed),
            repeat_qc_pools=len(qc),all_steps_aligned=True,source_unchanged=True,training_started=False,
            label_contract_ready=False,label_contract_pending='temporal teacher evidence/observability/within-pool variation audit',
            elapsed_seconds=time.time()-start)
        dump(out/'completion_audit.json',report)
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['status.json','SHA256SUMS.txt']}
        (out/'SHA256SUMS.txt').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(hashes.items())))
        dump(out/'status.json',dict(phase='complete',**report))
    except Exception as error:
        dump(out/'failure.json',dict(error_type=type(error).__name__,message=str(error),completed_candidates=len(completed),traceback=traceback.format_exc(),original_results_retained=True))
        dump(out/'status.json',dict(phase='failed',completed_candidates=len(completed),training_started=False));raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()

