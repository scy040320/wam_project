"""Enrich frozen replay with local RGB/depth/interface evidence, without fitting.

This adapter never alters the immutable replay module, source actions, models or
tolerances. Exact teacher body/geom roles, depth and camera state are private
annotation aids. Actual RGB tracking and repeat noise are saved separately from
physical truth. They must pass a separately versioned certificate before use.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from observation_geometry import (saved_image_from_raw, project_world_points,
        local_rgb_registration, contact_interface_witness, track_rgb_role)
except ModuleNotFoundError:
    from wam_reranking.observation_geometry import (saved_image_from_raw, project_world_points,
        local_rgb_registration, contact_interface_witness, track_rgb_role)

BASE_SHA = '3684243f669a27430b27f45673c138859548e160dff757de7269eb25128e8069'
BASE_VERSION = 'known_task_temporal_recovery_supervision_20261006_v2_scoped_segmentation'
TRACK_ROLES = ('target','eef','anchor','left','right')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def json_ready(value):
    if isinstance(value, dict): return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray): return json_ready(value.tolist())
    if isinstance(value, np.generic): return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def descriptor(mask, actual, rendered):
    ys,xs = np.nonzero(mask)
    return dict(rendered_pixels=int(mask.sum()),
        centroid_xy=[float(xs.mean()),float(ys.mean())] if len(xs) else None,
        bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
        local_rgb=local_rgb_registration(actual,rendered,mask))


def causal_motion_start_steps(step):
    """Add cumulative real-frame2 windows without backfilling frame1 identity."""
    if type(step) is not int or step not in range(17):
        raise ValueError('Actual frame indices must lie in 0..16')
    return sorted(start for start in {max(1,step-2),1,2} if start<step)


def fixed_branch_bodies(model, root, descendants):
    """Exact root plus fixed child branches, excluding independently jointed objects."""
    # The world body has many unrelated surfaces and is never expanded to the
    # whole scene. Its own geoms remain ambiguous instance annotation only.
    if root==0:return {0}
    selected=set()
    for bid in descendants(model,root):
        node=bid;fixed=True
        while node!=root:
            if int(model.body_jntnum[node])>0:fixed=False;break
            node=int(model.body_parentid[node])
        if fixed:selected.add(bid)
    return selected


def support_instance_mask(cache, instance):
    """Private typed body mask; never an all-world support visibility union."""
    return np.isin(cache['geom'],[g['geom_id'] for g in instance['visual_geoms']])


def attach(base, root):
    old_bind,old_capture,old_replay = base.bind,base.capture,base.replay
    runtime = dict(frames=[], seen=set())

    def bind(env,task):
        b=old_bind(env,task);model=env.sim.model
        def render_role(ids):
            # Collision bodies for finger roots, not just pad tip bodies.
            roots={int(model.geom_bodyid[g]) for g in ids}
            bodies=set().union(*(base.body_descendants(model,r) for r in roots))
            return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}
        b['left_render']=render_role(b['left']);b['right_render']=render_role(b['right'])
        if b['left_render'] & b['right_render']:raise RuntimeError('Left/right visual roles overlap')
        b['target_render']=set(b['target_geoms'])
        # Static children may contain all visible anchor geometry (cabinet
        # root has none). Prune branches with any additional joint, including
        # other drawers; the root's own freejoint is allowed for rigid anchors.
        anchor_bodies=fixed_branch_bodies(model,b['anchor_body'],base.body_descendants)
        b['anchor_render']={g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in anchor_bodies}-b['target_render']
        # Literal support counterpart bodies have separate identities. Scene
        # co-presence is not a supporting-surface instance observation.
        support_geoms=set(b['world_geoms'])-b['target_render']
        b['typed_support_instances']={}
        for bid in sorted({int(model.geom_bodyid[g]) for g in support_geoms}):
            bodies=fixed_branch_bodies(model,bid,base.body_descendants)
            ids={g for g in support_geoms if int(model.geom_bodyid[g]) in bodies}
            key='body_'+str(bid)
            b['typed_support_instances'][key]=dict(support_instance_key=key,
                body_id=bid,body_name=model.body_id2name(bid),identity_ambiguous=bid==0,
                instance_annotation_only=True,
                identity_basis='private_counterpart_body_binding_not_actual_RGB_identity',
                world_body_not_a_unique_surface_instance=bid==0,
                visual_geoms=[dict(geom_id=g,geom_name=model.geom_id2name(g),
                    body_id=int(model.geom_bodyid[g]),body_name=model.body_id2name(int(model.geom_bodyid[g])),
                    render_group=int(model.geom_group[g])) for g in sorted(ids)])
        b['visual_role_mapping']={name:[dict(geom_id=g,geom_name=model.geom_id2name(g),
            body_id=int(model.geom_bodyid[g]),body_name=model.body_id2name(int(model.geom_bodyid[g])),
            render_group=int(model.geom_group[g])) for g in sorted(ids)] for name,ids in
            [('target',b['target_render']),('anchor',b['anchor_render']),('left',b['left_render']),('right',b['right_render'])]}
        b['geom_body_ids']=np.asarray(model.geom_bodyid,dtype=int).copy()
        return b

    def capture(env,raw,actual,binding,step,convention):
        frame,segments=old_capture(env,raw,actual,binding,step,convention)
        sim=env.sim;model=sim.model;b=binding
        from robosuite.utils.camera_utils import get_real_depth_map
        contacts=[]
        for i in range(sim.data.ncon):
            c=sim.data.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in b['target_geoms'] and g2 not in b['target_geoms']:continue
            other=g2 if g1 in b['target_geoms'] else g1
            role=('left' if other in b['left'] else 'right' if other in b['right'] else
                  'support' if other in b['world_geoms'] else 'other')
            other_body=int(model.geom_bodyid[other])
            contacts.append(dict(role=role,geom_ids=[g1,g2],geom_names=[model.geom_id2name(g1),model.geom_id2name(g2)],
                counterpart_body_id=other_body,counterpart_body_name=model.body_id2name(other_body),
                support_instance_key='body_'+str(other_body) if role=='support' else None,
                position_world_m=np.asarray(c.pos).tolist(),signed_distance_m=float(c.dist)))
        views={};cache={}
        for view,camera,key in [('primary','agentview','agentview_image'),('wrist','robot0_eye_in_hand','robot0_eye_in_hand_image')]:
            cid=int(model.camera_name2id(camera))
            params=dict(camera_position_m=np.asarray(sim.data.cam_xpos[cid]).copy(),
                camera_rotation_world_from_camera=np.asarray(sim.data.cam_xmat[cid]).reshape(3,3).copy(),
                fovy_degrees=float(model.cam_fovy[cid]),image_shape=(256,256),image_convention=convention)
            raw_rgb,raw_depth=sim.render(width=256,height=256,camera_name=camera,depth=True)
            rendered=saved_image_from_raw(raw_rgb,image_convention=convention)
            render_registration=base.image_check(rendered,np.flipud(np.asarray(raw[key])))
            # The frozen replay separately enforces original source RGB and
            # proprio alignment. A fresh geometry render is an annotation aid:
            # disagreement masks this view, never substitutes its current GT
            # image for the actual cached observation or invalidates pairing.
            depth=saved_image_from_raw(get_real_depth_map(sim,raw_depth),image_convention=convention)
            geom=segments[view]
            roles={name:np.isin(geom,list(ids)) for name,ids in [('target',b['target_render']),
                ('left',b['left_render']),('right',b['right_render']),('anchor',b['anchor_render'])]}
            roles['eef']=roles['left']|roles['right']
            desc={name:descriptor(mask,actual[view],rendered) for name,mask in roles.items()}
            witness=[]
            for contact in contacts:
                if contact['role'] not in ('left','right','support'):continue
                projection=project_world_points(contact['position_world_m'],**params)
                if contact['role']=='support':
                    instance=b['typed_support_instances'].get(contact['support_instance_key'])
                    ids={g['geom_id'] for g in instance['visual_geoms']} if instance else set()
                    other_mask=np.isin(geom,list(ids))
                else:other_mask=roles[contact['role']]
                report=contact_interface_witness(projection,depth_m=depth,target_mask=roles['target'],
                    finger_mask=other_mask,actual_rgb=actual[view],rendered_rgb=rendered)
                witness.append(dict(physics_contact=contact,interface_evidence=report))
            views[view]=dict(camera= json_ready(params), global_rgb=base.image_check(actual[view],rendered),
                fresh_render_vs_cached_observation=render_registration,
                fresh_geometry_registered_to_actual_input=step>0 and render_registration['passed'] and base.image_check(actual[view],rendered)['passed'],
                roles=desc,contact_interfaces=witness,
                support_instances={},support_identity_requires_typed_actual_RGB_track=True,
                literal_contact_observability_is_separate_from_carrying=True)
            cache[view]=dict(actual=actual[view].copy(),rendered=rendered.copy(),depth=depth,geom=geom,
                roles=roles,camera=params,body_pos=np.asarray(sim.data.body_xpos).copy(),
                body_rot=np.asarray(sim.data.body_xmat).reshape(-1,3,3).copy(),geom_body_ids=b['geom_body_ids'])
        evidence=dict(step=step,role='offline_observation_supervision_only',views=views,
            private_collision_vs_visual_mapping=b['visual_role_mapping'],
            private_typed_support_mapping=b['typed_support_instances'],
            support_instance_binding_is_not_observation_certification=True,
            cached_entry_is_diagnostic_only=step==0,physical_events_do_not_certify_input_observability=True)
        frame['observation_evidence']=json_ready(evidence);runtime['frames'].append(cache)
        return frame,segments

    def endpoints(start,end,view,role):
        a,b=start[view],end[view];params=a['camera'];h,w=params['image_shape'];c=params['image_convention']
        f=h/(2*np.tan(np.deg2rad(params['fovy_degrees'])/2))
        def convert(corners):
            p=np.asarray(corners,float).reshape(-1,2);indices=np.floor(p+.5).astype(int)
            valid=(indices[:,0]>=0)&(indices[:,0]<w)&(indices[:,1]>=0)&(indices[:,1]<h)
            world=np.full((len(p),3),np.nan);transformed=world.copy()
            for i,(x,y) in enumerate(indices):
                if not valid[i]:continue
                gid=int(a['geom'][y,x]);z=float(a['depth'][y,x]);bid=int(a['geom_body_ids'][gid]) if gid>=0 else -1
                if gid<0 or not np.isfinite(z) or z<=0:valid[i]=False;continue
                ycv=h-1-p[i,1] if c==-1 else p[i,1]
                cv=np.array([(p[i,0]-w/2)*z/f,(ycv-h/2)*z/f,z])
                world[i]=params['camera_rotation_world_from_camera'] @ (cv*np.array([1,-1,-1]))+params['camera_position_m']
                local=a['body_rot'][bid].T @ (world[i]-a['body_pos'][bid])
                transformed[i]=b['body_rot'][bid] @ local+b['body_pos'][bid]
            target=project_world_points(transformed,**b['camera'])
            stationary=project_world_points(world,**b['camera'])
            ok=valid&target['valid']&stationary['valid']
            # A projected point on the correct body is not necessarily the
            # visible surface of that body. Reject self-occluded endpoints.
            for i in np.flatnonzero(ok):
                x,y=np.floor(target['pixel_xy'][i]+.5).astype(int)
                if x<0 or x>=w or y<0 or y>=h:
                    ok[i]=False
                elif abs(float(b['depth'][y,x])-float(target['camera_depth_m'][i]))>1e-3:
                    ok[i]=False
            return dict(start_pixel_xy=p,end_pixel_xy=target['pixel_xy'],
                camera_only_end_pixel_xy=stationary['pixel_xy'],valid=ok)
        return convert

    def replay(env,fork,collector,binding,folder,snapshot,cause,*,save_segments=True):
        runtime['frames']=[]
        first=old_replay(env,fork,collector,binding,folder,snapshot,cause,save_segments=save_segments)
        caches=runtime['frames'];frames=first[0];key=str(folder)
        if key in runtime['seen']:return first
        runtime['seen'].add(key)
        runtime['frames']=[]
        repeated=old_replay(env,fork,collector,binding,folder,snapshot,cause,save_segments=False)
        repeat_cache=runtime['frames']
        delta=np.abs(np.c_[first[3],first[4]]-np.c_[repeated[3],repeated[4]])
        atoms=('left_finger_target_contact','right_finger_target_contact','target_support_contact','target_anchor_contact')
        contact_same=all(all(a['physics'][n]==b['physics'][n] for n in atoms) for a,b in zip(frames,repeated[0]))
        if delta.max()>1e-3 or np.quantile(delta,.95)>1e-4 or not contact_same:
            raise RuntimeError('Same-candidate observation repeat failed original frozen physical tolerance')
        active_support_keys=set()
        for step,frame in enumerate(frames):
            evidence=frame['observation_evidence'];evidence['same_candidate_repeat_qc']=dict(physics_passed=True,
                qpos_qvel_max_abs=float(delta.max()),qpos_qvel_p95=float(np.quantile(delta,.95)),all_contact_atoms_match=True)
            # Only counterparts actually encountered in the current/past prefix
            # are attached here. Future contact does not establish an earlier
            # support identity or retroactively populate its role/noise record.
            active_support_keys.update(contact['physics_contact']['support_instance_key']
                for view_record in evidence['views'].values() for contact in view_record['contact_interfaces']
                if contact['physics_contact']['role']=='support' and contact['physics_contact'].get('support_instance_key'))
            for view in ('primary','wrist'):
                a,b=caches[step][view],repeat_cache[step][view];record=evidence['views'][view]
                noise={}
                for role in ('target','left','right','eef','anchor'):
                    mask=a['roles'][role]|b['roles'][role]
                    desc_a=descriptor(a['roles'][role],a['actual'],a['rendered']);desc_b=descriptor(b['roles'][role],b['actual'],b['rendered'])
                    ca,cb=desc_a['centroid_xy'],desc_b['centroid_xy']
                    noise[role]=dict(centroid_noise_px=float(np.linalg.norm(np.asarray(ca)-cb)) if ca and cb else None,
                        rgb_repeat=local_rgb_registration(a['rendered'],b['rendered'],mask))
                record['repeat_noise']=noise
                typed={}
                for support_key in sorted(active_support_keys):
                    instance=binding['typed_support_instances'].get(support_key)
                    if instance is None:continue
                    ma=support_instance_mask(a,instance);mb=support_instance_mask(b,instance)
                    da=descriptor(ma,a['actual'],a['rendered']);db=descriptor(mb,b['actual'],b['rendered'])
                    ca,cb=da['centroid_xy'],db['centroid_xy']
                    repeated_noise=dict(centroid_noise_px=float(np.linalg.norm(np.asarray(ca)-cb)) if ca and cb else None,
                        rgb_repeat=local_rgb_registration(a['rendered'],b['rendered'],ma|mb))
                    typed[support_key]=dict(da,support_instance_key=support_key,
                        body_id=instance['body_id'],body_name=instance['body_name'],
                        identity_ambiguous=instance['identity_ambiguous'],instance_annotation_only=True,
                        repeat_noise=repeated_noise)
                record['support_instances']=typed
            # Every interval is causal and ends at this frame. Frame0 is not
            # substituted with a re-rendered future or a different query input.
            intervals=[]
            for start_step in causal_motion_start_steps(step):
                for view in ('primary','wrist'):
                    tracks={role:track_rgb_role(caches[start_step][view]['actual'],caches[step][view]['actual'],
                        caches[start_step][view]['roles'][role],caches[step][view]['roles'][role],
                        projected_endpoints=endpoints(caches[start_step],caches[step],view,role)) for role in TRACK_ROLES}
                    noises={role:[evidence['views'][view]['repeat_noise'][role]['centroid_noise_px'],
                        frames[start_step]['observation_evidence']['views'][view]['repeat_noise'][role]['centroid_noise_px']] for role in TRACK_ROLES}
                    typed_tracks={}
                    for support_key in sorted(active_support_keys):
                        instance=binding['typed_support_instances'].get(support_key)
                        if instance is None:continue
                        before=frames[start_step]['observation_evidence']['views'][view]['support_instances'].get(support_key)
                        current=evidence['views'][view]['support_instances'].get(support_key)
                        if instance['identity_ambiguous']:
                            tracked=dict(verified=False,geometry_motion_support=False,
                                metric_world_motion_certified=False,count=0,tracks=[],
                                reason='world_body_is_not_a_unique_support_surface_instance')
                        elif before is None or current is None:
                            tracked=dict(verified=False,geometry_motion_support=False,
                                metric_world_motion_certified=False,count=0,tracks=[],
                                reason='typed_support_reference_not_observed_in_causal_prefix')
                        else:
                            start_mask=support_instance_mask(caches[start_step][view],instance)
                            end_mask=support_instance_mask(caches[step][view],instance)
                            tracked=track_rgb_role(caches[start_step][view]['actual'],caches[step][view]['actual'],
                                start_mask,end_mask,projected_endpoints=endpoints(caches[start_step],caches[step],view,support_key))
                        noise_values=[record.get('repeat_noise',{}).get('centroid_noise_px') if record else None
                            for record in (before,current)]
                        typed_tracks[support_key]=dict(support_instance_key=support_key,
                            body_id=instance['body_id'],body_name=instance['body_name'],
                            identity_ambiguous=instance['identity_ambiguous'],track=tracked,
                            same_candidate_repeat_noise_px=max(noise_values) if all(v is not None for v in noise_values) else None,
                            reference_identity_must_be_certified_causally_at_start=True)
                    intervals.append(dict(start_step=start_step,end_step=step,view=view,tracks=tracks,
                        typed_support_tracks=typed_tracks,
                        same_candidate_repeat_noise={role:max(values) if all(v is not None for v in values) else None for role,values in noises.items()},
                        geometry_correspondence_uses_private_depth_only_for_annotation=True,
                        correspondence_camera_pose_accounted=True,
                        interval_kind='cumulative_from_real_step2' if start_step==2 and step>4 else 'existing_causal_window',
                        no_future_identity_backfill=True,
                        finger_identity_requires_independent_same_side_actual_RGB_tracks=True,
                        camera_motion_compensated_by_per_frame_projection=False))
            evidence['actual_rgb_temporal_intervals']=json_ready(intervals)
        return first

    base.bind,base.capture,base.replay=bind,capture,replay
    return runtime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pilot-only',action='store_true')
    parser.add_argument('--reuse-pilot',type=Path);args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    code=root/'research_runs'/BASE_VERSION/'replay_temporal_recovery_evidence.py'
    if sha(code)!=BASE_SHA:raise RuntimeError('Frozen replay module SHA changed')
    if out.exists():raise RuntimeError('Refuse overwrite or implicit resume')
    geometry_sha=sha(Path(inspect.getfile(track_rgb_role)))
    if args.reuse_pilot:
        pilot=args.reuse_pilot.resolve()
        manifest=json.loads((pilot/'observation_adapter_manifest.json').read_text())
        if (manifest.get('adapter_sha256')!=sha(Path(__file__)) or
            manifest.get('geometry_sha256')!=geometry_sha or
            manifest.get('schema')!='actual_rgb_recovery_observation_evidence_v1'):
            raise RuntimeError('Only identical observation adapter/geometry pilot may be reused')
        for record in json.loads((pilot/'records.json').read_text()):
            teacher=json.loads((pilot/record['directory']/'temporal_teacher.json').read_text())
            if len(teacher['frames'])!=17 or any('observation_evidence' not in f for f in teacher['frames']):
                raise RuntimeError('Pure physical or incomplete pilot is not observation evidence')
    spec=importlib.util.spec_from_file_location('frozen_observation_base',code)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    attach(module,root)
    argv=[str(code),'--root',str(root),'--output',str(out)]
    if args.pilot_only:argv.append('--pilot-only')
    if args.reuse_pilot:argv+=['--reuse-pilot',str(args.reuse_pilot.resolve())]
    sys.argv=argv
    try:module.main()
    finally:
        if out.exists():
            provenance=dict(schema='actual_rgb_recovery_observation_evidence_v1',base_script_sha256=BASE_SHA,
                adapter_sha256=sha(Path(__file__)),geometry_sha256=geometry_sha,
                role='offline_supervision_only',new_policy_queries=0,
                new_candidate_actions=0,training_started=False,old_physical_teacher_masks_unchanged=True,
                certified_labels_are_separate_derived_sidecars=True,
                repeated_candidate_ids='all candidate IDs, not only pool candidate0',
                added_real_step2_cumulative_windows=True,
                typed_support_counterpart_instances=True,
                untyped_all_world_support_union_removed=True,
                world_body_support_instance_ambiguity_masks_tracking=True,
                causal_support_reference_identity_not_backfilled=True,
                independent_left_right_actual_RGB_tracks=True,
                private_contact_visual_roles_kept_distinct=True)
            module.dump(out/'observation_adapter_manifest.json',provenance)
            hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('status.json','SHA256SUMS.txt')}
            (out/'SHA256SUMS.txt').write_text(''.join(f'{h}  {p}\n' for p,h in sorted(hashes.items())))


if __name__=='__main__':main()
