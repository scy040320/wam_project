"""Bounded development repair: real first-block recovery for every candidate.

Existing query pools and public source snapshots remain immutable. All arms
restore the same runtime snapshot. This is not a new independent test or a
terminal-success dataset. Physical holding/contact predicates remain masked.
"""
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
import traceback

import numpy as np
from PIL import Image
from wam_reranking.candidate_effects import localize_candidate_visual_evidence
from wam_reranking.observed_recovery import labels_from_actual
from wam_reranking.recovery_label_contract import audit_recovery_labels, audit_recovery_membership
from wam_reranking.shared_query_cache import file_sha

SOURCE = 'candidate_reranking_known_task_closedloop_mechanism_20261004_v3_shared_source'
GROUPS = {0: (31, 34), 9: (10, 13), 46: (10, 13), 57: (10, 13)}
CAUSES = ('normal', 'visual_occlusion', 'object_shift', 'execution_contact_deviation', 'unknown')
AT_SHA = '68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec'

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n'); tmp.replace(path)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module

def result_from_folder(folder):
    q = json.loads((folder/'query.json').read_text())
    return dict(actions=np.load(folder/'actions.npy', allow_pickle=False),
        future_image_predictions=dict(future_image=np.load(folder/'predicted_primary.npy', allow_pickle=False),
            future_wrist_image=np.load(folder/'predicted_wrist.npy', allow_pickle=False)),
        value_prediction=q['value']), q

def record_actual(ex, result, snapshot, start, folder, *, cause='normal'):
    ex.restore(snapshot)
    state = np.asarray(ex.env.get_sim_state()).copy()
    if not np.array_equal(state, snapshot['sim_state']): raise RuntimeError('Exact arm start failed')
    sequence = {v: [np.asarray(start[v+'_image']).copy()] for v in ('primary', 'wrist')}
    proprio = [np.asarray(start['proprio']).copy()]; requested=[]; applied=[]; done=False
    for action in np.asarray(result['actions'], dtype=np.float32):
        raw, _, done, _ = ex.env.step(action.tolist())
        # A persistent screen occluder / view conflict is part of this new
        # collection contract. It cannot disappear just because an arm runs.
        if cause == 'visual_occlusion': raw, _ = ex.base.InterventionEnv._mask_primary(raw)
        if cause == 'unknown': raw, _ = ex.base.InterventionEnv._shift_primary_view(raw)
        obs=ex.prep(raw); requested.append(action.copy()); applied.append(action.copy())
        for v in sequence: sequence[v].append(np.asarray(obs[v+'_image']).copy())
        proprio.append(np.asarray(obs['proprio']).copy())
        if done: break
    folder.mkdir(parents=True, exist_ok=False)
    arrays={}
    for v, frames in sequence.items():
        np.save(folder/(v+'.npy'), np.stack(frames)); arrays[v+'_sequence']=v+'.npy'
    for name, data in (('requested', requested), ('applied', applied)):
        np.save(folder/(name+'.npy'), np.asarray(data)); arrays[name+'_actions']=name+'.npy'
    np.save(folder/'proprio.npy', np.asarray(proprio)); arrays['proprio_sequence']='proprio.npy'
    np.save(folder/'proprio_start.npy', proprio[0]); arrays['proprio_start']='proprio_start.npy'
    np.save(folder/'proprio_end.npy', proprio[-1]); arrays['proprio_end']='proprio_end.npy'
    np.save(folder/'exact_start_state.npy', state)
    dump(folder/'block_observation.json', dict(executed_length=len(requested), terminated=bool(done),
        only_first_block=True, terminal_task_success_not_used_for_recovery_labels=True))
    return obs, done, arrays, len(requested)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--pilot-only', action='store_true')
    a=ap.parse_args(); root=a.root.resolve(); out=a.output.resolve()
    old_output=(root/'outputs'/SOURCE).resolve()
    if out==old_output or old_output in out.parents: raise RuntimeError('New output may not be inside frozen source')
    if out.exists(): raise RuntimeError('Refuse overwrite, selective recollection or implicit resume')
    out.mkdir(parents=True); started=time.time(); touched={}; rows=[]; pools=[]; short=[]; qc=[]
    protocol=dict(schema='paired_candidate_observed_recovery_v1',model_alias='V9',
        group_states=GROUPS, train_first_state=True, val_second_state=True, causes=CAUSES,
        k=4,pools=40,candidate_first_blocks=160,clean_b_qc=8,source=SOURCE,
        label_schema='candidate_recovery_labels_v2',label_inherits='candidate_recovery_supervision_v1',
        frozen_attributor_sha256=AT_SHA,quality_threshold=.5,first_block_length=16,
        unknown_variant='view_conflict',known_candidate_queries_reused_immutable=True,
        unknown_new_queries=32,unknown_seed_base=400000000,
        persistent_primary_mask_for_occlusion=True,persistent_shifted_primary_for_unknown=True,
        physical_recovery_masked=True,full_terminal_success_not_collected=True,
        consumed_development_only=True,no_32_or_128_validation_states=True,
        all_failed_or_no_recovery_variance_pools_retained=True,
        clean_pair_thresholds=dict(sim_max_abs=1e-3,sim_p95=1e-4,proprio_max_abs=1e-3,
            image_mean_abs=2,image_p95=8,image_fraction_gt5=.06,image_psnr=30),
        pilot=dict(task=0,state=31,pools=5,candidates=20,
            gate='pairing/evidence/hashes only; no success or recovery-variance selection'))
    dump(out/'protocol.json',protocol)
    legacy_code=root/'research_runs'/SOURCE; sys.path.insert(0,str(legacy_code))
    legacy=load('recovery_legacy_shared_source',legacy_code/'run_mechanism.py')
    # Redirect EVERY legacy output global before construction. Never write V3.
    legacy.OUT=out/'runtime'; legacy.OUT.mkdir()
    backbone=root/'outputs/known_task_evidence_residual_repair_20261005_v2/candidate_backbone.json'
    (legacy.OUT/'candidate_only_model.json').write_bytes(backbone.read_bytes())
    try:ex=legacy.Experiment(0,protocol)
    except Exception as error:
        dump(out/'failure.json',dict(stage='runtime_initialization',error_type=type(error).__name__,message=str(error),
            raw_candidates_completed=0,original_source_untouched=True))
        raise
    if not ex.attributor.validation_replay_exact: raise RuntimeError('Frozen attributor replay failed')
    def remember(path):
        path=path.resolve(); key=str(path.relative_to(root))
        h=file_sha(path)
        if key in touched and touched[key]!=h: raise RuntimeError('Source changed mid-collection')
        touched[key]=h; return h
    try:
        for task, states in GROUPS.items():
            if ex.env is not None: ex.env.close()
            ex.task_id=task; ex.task=ex.suite.get_task(task); ex.states=ex.suite.get_task_init_states(task)
            ex.env,ex.language=legacy.parent.get_libero_env(ex.task,ex.cfg.model_family,resolution=ex.cfg.env_img_res)
            target,kind,ex.subgoal,_=ex.taskspec[str(task)]['0']
            ex.base.TARGET_OBJECT=target; ex.base.TARGET_INTERVENTION_KIND=kind
            ex.base.SHIFT_VECTOR=np.asarray([.0625,-.03,.005]); ex.relation=ex.bundle.FROZEN_RELATIONS[(task,0)]
            ex.attributor.set_context(task,target,ex.relation)
            for si,state in enumerate(states):
                if a.pilot_only and (task,state)!=(0,31): continue
                split='train' if si==0 else 'val'
                group=root/'outputs'/SOURCE/'groups'/f'task{task}_state{state}'
                remember(group/'source.pkl'); remember(group/'source_manifest.json')
                payload,manifest=legacy.read_source(group,task,state,ex.base.sha_array,ex.base.obs_hash)
                if payload.get('early_terminal'): raise RuntimeError('Frozen source is terminal; do not swap state')
                # Same restore path for control A and B; raw/cache discrepancy
                # is diagnostic, not a substituted physics-pair threshold.
                clean=[]
                for arm in ('clean_a','clean_b'):
                    start=copy.deepcopy(payload['query_obs'])
                    end,done,_,n=record_actual(ex,payload['source_result'],payload['snapshot'],start,
                        out/'qc'/f'task{task}_state{state}'/arm)
                    if n!=16 or done: raise RuntimeError('Frozen QC source became short/terminal')
                    clean.append(dict(observation=end,sim=np.r_[ex.env.sim.data.qpos.copy(),ex.env.sim.data.qvel.copy()]))
                checked=legacy.parent.compare_pair(*clean)
                dump(out/'qc'/f'task{task}_state{state}'/'strict_audit.json',checked); qc.append(checked)
                if not checked['passed']: raise RuntimeError('Clean-A/B frozen pairing threshold failed')
                for cause in CAUSES:
                    pool_id=f'task{task}_state{state}_{cause}'; pool_dir=out/'pools'/pool_id
                    pool_dir.mkdir(parents=True,exist_ok=False)
                    if cause=='unknown':
                        ex.restore(payload['snapshot'])
                        wrapper=ex.base.InterventionEnv(ex.env,dict(name='unknown',label='unknown',variant='view_conflict'))
                        wrapper.step_count=42;wrapper.arm(block_index=2,block_start=42,offset=4)
                        start,done,n,ep,_=ex.execute(payload['source_result'],16,pool_dir/'source_block',payload['query_obs'],2,wrapper)
                        if n!=16 or done or wrapper.event is None: raise RuntimeError('Unknown source invalid; do not substitute')
                        snapshot=ex.fork.capture_runtime_snapshot(ex.env); record=ex.attributor.infer(ep)
                        dump(pool_dir/'offline_intervention.json',dict(event=wrapper.event,supervision_only=True))
                        candidates=[]
                        for cid in range(4):
                            seed=400000000+task*100000+state*100+cid
                            result,seconds=ex.query(start,seed); candidates.append(result)
                            cd=pool_dir/f'query_{cid}';cd.mkdir()
                            np.save(cd/'actions.npy',result['actions'])
                            for name,key in (('predicted_primary','future_image'),('predicted_wrist','future_wrist_image')):
                                np.save(cd/(name+'.npy'),result['future_image_predictions'][key])
                            dump(cd/'query.json',dict(seed=seed,value=float(result['value_prediction']),seconds=seconds))
                        query_source='new_frozen_schedule'; source_sha={}
                    else:
                        condition='clean' if cause=='normal' else cause
                        src=root/'outputs'/SOURCE/'scenarios'/f'task{task}_state{state}_{condition}'
                        snap_path=src/'root_runtime_snapshot.pkl'; remember(snap_path)
                        with snap_path.open('rb') as f:snapshot=pickle.load(f)
                        bd=src/'source_block/evidence/block'
                        start={k:np.load(bd/(n+'.npy'),allow_pickle=False) for k,n in (
                            ('primary_image','actual_observed_primary'),('wrist_image','actual_observed_wrist'),('proprio','actual_observed_proprio'))}
                        for n in ('actual_observed_primary','actual_observed_wrist','actual_observed_proprio'):remember(bd/(n+'.npy'))
                        rp=src/'source_block/evidence/attribution.json';remember(rp);record=json.loads(rp.read_text())
                        pp=src/'shared_root_pool';mp=pp/'pool.json';remember(mp);meta=json.loads(mp.read_text())
                        if meta['k']!=4 or meta['observation_sha256']!=ex.base.obs_hash(start):raise RuntimeError('Original query input mismatch')
                        candidates=[]
                        for cid in range(4):
                            cd=pp/f'candidate_{cid}'
                            for f in cd.iterdir():
                                if f.is_file():remember(f)
                            result,_=result_from_folder(cd); candidates.append(result)
                        query_source=str(pp.relative_to(root));source_sha={str(mp.relative_to(root)):file_sha(mp),str(snap_path.relative_to(root)):file_sha(snap_path)}
                    with (pool_dir/'snapshot.pkl').open('xb') as f:pickle.dump(snapshot,f)
                    for view in ('primary','wrist'):np.save(pool_dir/(view+'_before.npy'),start[view+'_image'])
                    np.save(pool_dir/'proprio_before.npy',start['proprio']);dump(pool_dir/'attribution.json',record)
                    before_sim_hash=ex.base.sha_array(snapshot['sim_state']); input_hash=ex.base.obs_hash(start)
                    candidate_rows=[]
                    for cid,result in enumerate(candidates):
                        cd=pool_dir/f'candidate_{cid}'
                        end,done,arrays,n=record_actual(ex,result,snapshot,start,cd,cause=cause)
                        np.save(cd/'planned_actions.npy',result['actions'])
                        for name,key in (('predicted_primary','future_image'),('predicted_wrist','future_wrist_image')):
                            np.save(cd/(name+'.npy'),result['future_image_predictions'][key])
                        visual=localize_candidate_visual_evidence(localizer=ex.attributor.localizer,
                            current_primary=Image.fromarray(start['primary_image']),current_wrist=Image.fromarray(start['wrist_image']),
                            predicted_primary=Image.fromarray(np.asarray(result['future_image_predictions']['future_image']).astype(np.uint8)),
                            predicted_wrist=Image.fromarray(np.asarray(result['future_image_predictions']['future_wrist_image']).astype(np.uint8)),
                            target_prompt=ex.relation.subject,anchor_prompt=ex.relation.anchor,relation=ex.relation.relation)
                        dump(cd/'candidate_visual.json',asdict(visual))
                        identity=dict(dataset=out.name,suite='libero90',task=task,state=state,candidate_id=cid,
                            observed_block_id=pool_id+'/first_block')
                        cr=dict(candidate_id=cid,executed_length=n,complete=n==16,split=split,
                            value=float(result['value_prediction']),query_observation_sha256=input_hash,
                            exact_start_sim_sha256=before_sim_hash,planned_actions_sha256=ex.base.sha_array(result['actions']),
                            directory=str(cd.relative_to(out)))
                        if n==16:
                            actual=[Image.fromarray(np.load(cd/(v+'.npy'),allow_pickle=False)[step]) for step in (0,16) for v in ('primary','wrist')]
                            sm=ex.attributor.localizer.relevance(actual,ex.relation.subject)
                            am=ex.attributor.localizer.relevance(actual,ex.relation.anchor)
                            labels=labels_from_actual(identity=identity,subject_maps=sm,anchor_maps=am,
                                requested=np.load(cd/'requested.npy'),applied=np.load(cd/'applied.npy'))
                            dump(cd/'labels.json',labels);np.savez_compressed(cd/'observer_maps.npz',subject=sm,anchor=am)
                            rec=dict(schema='candidate_recovery_labels_v2',inherits='candidate_recovery_supervision_v1',
                                identity=identity,role='paired_recovery_supervision',split=split,executed_length=16,
                                step_indices=list(range(17)),arrays=arrays,labels_path='labels.json')
                            ac=audit_recovery_labels(rec,directory=cd)
                            if not ac['ready']:raise RuntimeError(json.dumps(ac))
                            dump(cd/'record.json',rec);rows.append(rec)
                        else:short.append(dict(identity=identity,executed_length=n,retained=True))
                        candidate_rows.append(cr)
                        dump(out/'status.json',dict(phase='pilot' if len(pools)<5 else 'full',
                            completed_pools=len(pools),completed_candidates=4*len(pools)+len(candidate_rows),
                            total_pools=40,total_candidates=160,task=task,state=state,cause=cause,candidate_id=cid,
                            elapsed_seconds=time.time()-started))
                    pm=dict(pool_id=pool_id,task=task,state=state,split=split,cause=cause,k=4,
                        query_source=query_source,query_observation_sha256=input_hash,exact_start_sim_sha256=before_sim_hash,
                        source_sha256=source_sha,relation=ex.relation.relation,candidates=candidate_rows)
                    dump(pool_dir/'pool.json',pm);pools.append(pm)
                    if len(pools)==5:
                        checks=dict(clean_pair=all(q['passed'] for q in qc),
                            pools=len(pools)==5,candidates=sum(len(p['candidates']) for p in pools)==20,
                            membership=audit_recovery_membership(rows)['ready'],
                            each_cause_complete=all(any(c['complete'] for p in pools if p['cause']==cause for c in p['candidates']) for cause in CAUSES))
                        pilot_hashes={str(f.relative_to(out)):file_sha(f) for f in (out/'pools').rglob('*') if f.is_file()}
                        checks['nonempty_files']=all((out/f).stat().st_size>0 for f in pilot_hashes)
                        dump(out/'pilot_sha256.json',pilot_hashes)
                        pa=dict(passed=all(checks.values()),pools=5,candidates=20,checks=checks,
                            complete_recovery_records=len(rows),short_retained=len(short),not_success_selected=True)
                        dump(out/'pilot_audit.json',pa)
                        if not pa['passed']:raise RuntimeError('Pilot pairing/evidence audit failed')
                    print(json.dumps(dict(pools=len(pools),candidates=4*len(pools),last=pool_id)),flush=True)
        membership=audit_recovery_membership(rows)
        if not membership['ready']:raise RuntimeError(json.dumps(membership))
        if not a.pilot_only and (len(pools)!=40 or sum(len(p['candidates']) for p in pools)!=160 or len(qc)!=8):raise RuntimeError('Frozen cardinality mismatch')
        coverage={cause:sum(r['complete'] for p in pools if p['cause']==cause for r in p['candidates']) for cause in CAUSES}
        if any(n==0 for n in coverage.values()):raise RuntimeError('Missing observed five-cause coverage')
        for path,sha in touched.items():
            if file_sha(root/path)!=sha:raise RuntimeError('Original source mutated')
        dump(out/'pools.json',pools)
        hashes={str(f.relative_to(out)):file_sha(f) for f in out.rglob('*') if f.is_file() and f.name!='status.json'}
        dump(out/'derived_sha256.json',hashes)
        report=dict(passed=True,pools=len(pools),candidate_first_blocks=4*len(pools),complete_labels=len(rows),
            short_blocks=short,clean_b_qc=len(qc),membership=membership,five_cause_complete_coverage=coverage,
            source_hashes=touched,source_unchanged=True,physical_recovery_certified=False,
            terminal_success_ranking_labels_added=0,training_started=False,elapsed_seconds=time.time()-started)
        dump(out/'completion_audit.json',report);dump(out/'status.json',dict(phase='complete',**report))
    finally:
        if ex.env is not None:ex.env.close()

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
