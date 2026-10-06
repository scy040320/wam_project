"""Reuse frozen selected-arm trajectories as masked recovery auxiliary silver.

Source files never change. Success/condition/attribution cannot enter labelling;
frozen predicted attribution is used ONLY to build deployment-time features.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from wam_reranking.contracts import (AttributionOutput,BeliefFact,BeliefState,
    CandidateVisualEvidence,CoarseCause,ConsistencyFactor,EvidenceQuality,Stage,TriValue)
from wam_reranking.belief import DependencyGraph
from wam_reranking.candidate_effects import parse_candidate_effect
from wam_reranking.evidence_residual import typed_gate_effect
from wam_reranking.observed_recovery import labels_from_actual,map_endpoint
from wam_reranking.recovery_contract import (FEATURE_NAMES,FEATURE_SCHEMA,CurrentFactEvidence,
    prepare_recovery_context,trace_candidate_recovery)
from wam_reranking.recovery_label_contract import audit_recovery_labels,audit_recovery_membership
from wam_reranking.shared_query_cache import file_sha
from wam_reranking.target_localization import CLIPSegTargetLocalizer

BINDINGS={0:('top drawer','wooden cabinet frame','articulated'),
    9:('black bowl','plate','on'),46:('alphabet soup','basket','inside'),
    57:('cream cheese','tray','inside')}
SOURCE='known_task_trusted_evidence_arbitration_20261005_v8'

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def attribution(record):
    return AttributionOutput(record['factor_probs'],record['class_probs'],CoarseCause(record['projected_cause']),
        float(record['confidence']),float(record['entropy']),EvidenceQuality(**record['evidence_quality']),
        record['source_block_id'],{k:TriValue(v) for k,v in record['factor_states'].items()},record['factor_confidences'])

def prior_from_snapshot(snapshot,task):
    b=BeliefState(task,*BINDINGS[task][:2])
    if not snapshot:return b
    b.task_stage=Stage(snapshot.get('task_stage','uncertain'))
    for name,fact in snapshot.get('facts',{}).items():
        b.facts[name]=BeliefFact(TriValue(fact['value']),float(fact['confidence']),fact['source'],
            int(fact.get('updated_at_block',0)),tuple(fact.get('evidence_ids',())))
    return b

def actual_observations(labels,subject_maps,anchor_maps):
    obs=[]
    for name in ('target_visible','receptacle_visible'):
        v=labels['predicates'][name]
        if v['before_mask']:
            maps=subject_maps if name=='target_visible' else anchor_maps
            view='primary' if map_endpoint(maps[0])['quality']>=map_endpoint(maps[1])['quality'] else 'wrist'
            obs.append(CurrentFactEvidence(name,TriValue(v['before']),v['before_quality'],
                'actual_before:'+name,view=view))
    return obs

def label_targets(labels):
    items={**labels['predicates'],**labels['observability'],'relation_support_2d':labels['relation_support_2d']}
    names=[];values=[];masks=[]
    for name,label in items.items():
        for endpoint in ('after','transition'):
            names.append(name+'.'+endpoint)
            mask=label[endpoint+'_mask'];masks.append(bool(mask))
            if not mask:values.append(0.);continue
            def scalar(value):return float(value) if name=='relation_support_2d' else float(value=='true')
            after=scalar(label['after'])
            values.append(after if endpoint=='after' else after-scalar(label['before']))
    return names,values,masks

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise RuntimeError('Refuse overwrite or automatic resume')
    out.mkdir(parents=True);started=time.time()
    audit_path=root/'outputs/known_task_recovery_contract_audit_20261006_v9_r2/historical_selected_arm_transport_audit.json'
    old=json.loads(audit_path.read_text());rows=[r for r in old['entries'] if r['potential_auxiliary']]
    if len(rows)!=797:raise RuntimeError('Frozen historical count changed')
    # Last state of each historical task remains val; no trajectory/random split.
    val_states={0:34,9:13,46:13,57:13}
    if a.limit:rows=rows[:a.limit]
    protocol=dict(schema='recovery_aux_relabelling_v2',model_alias='V9',source=SOURCE,
        frozen_source_audit_sha256=file_sha(audit_path),expected_complete_blocks=797,
        schema_inherits='candidate_recovery_supervision_v1',features=FEATURE_SCHEMA,
        feature_names=FEATURE_NAMES,val_states=val_states,task_bindings=BINDINGS,
        silver_label_source='frozen_CLIPSeg_actual_RGB_and_exact_measured_commands',
        quality_threshold=.5,physical_grasp_lift_placement_masked=True,
        no_final_success_for_labels=True,no_attribution_prediction_for_labels=True,
        no_counterfactual_labels_for_unexecuted_candidates=True,role='historical_selected_arm_auxiliary')
    dump(out/'protocol.json',protocol)
    detector=CLIPSegTargetLocalizer(device='cuda',model_path=str(root.parent/'model_cache/clipseg-rd64-refined'))
    records=[];examples=[];coverage=Counter();classes=Counter();feature_rows=[];target_names=None
    for i,row in enumerate(rows):
        path=root/row['path'];folder=path.parent;bd=folder/'evidence/block';task=row['task'];state=row['state']
        block=int(folder.name.split('_')[-1]);decision=json.loads((folder.parent/f'decision_{block}.json').read_text())
        cid=int(decision['selected_candidate_id']);identity=dict(dataset=SOURCE,suite='libero90',
            task=task,state=state,candidate_id=cid,observed_block_id=str(folder.relative_to(root)))
        split='val' if state==val_states[task] else 'train'
        dest=out/'records'/f'task{task}_state{state}_{row["condition"]}_block{block}'
        dest.mkdir(parents=True,exist_ok=False)
        with np.load(path,allow_pickle=False) as traj:
            arrays={}
            for view in ('primary','wrist'):
                first=np.load(bd/f'O_t_{view}.npy',allow_pickle=False)
                sequence=np.concatenate([first[None],traj[view]],axis=0)
                if sequence.shape!=(17,256,256,3):raise RuntimeError('Actual image alignment changed')
                np.save(dest/(view+'.npy'),sequence);arrays[view+'_sequence']=view+'.npy'
            for name in ('requested','applied'):
                np.save(dest/(name+'.npy'),traj[name]);arrays[name+'_actions']=name+'.npy'
            requested=traj['requested'].copy();applied=traj['applied'].copy()
        for prefix,timepoint in (('O_t','start'),('actual_observed','end')):
            np.save(dest/f'proprio_{timepoint}.npy',np.load(bd/f'{prefix}_proprio.npy',allow_pickle=False))
            arrays['proprio_'+timepoint]=f'proprio_{timepoint}.npy'
        actual=[Image.fromarray(np.load(dest/(v+'.npy'),allow_pickle=False)[step])
                for step in (0,16) for v in ('primary','wrist')]
        subject,anchor,relation=BINDINGS[task]
        sm=detector.relevance(actual,subject);am=detector.relevance(actual,anchor)
        labels=labels_from_actual(identity=identity,subject_maps=sm,anchor_maps=am,requested=requested,applied=applied)
        dump(dest/'labels.json',labels)
        np.savez_compressed(dest/'observer_maps.npz',subject=sm,anchor=am)
        record=dict(schema='candidate_recovery_labels_v2',inherits='candidate_recovery_supervision_v1',
            identity=identity,role='historical_selected_arm_auxiliary',split=split,executed_length=16,
            step_indices=list(range(17)),arrays=arrays,labels_path='labels.json')
        checked=audit_recovery_labels(record,directory=dest)
        if not checked['ready']:raise RuntimeError(json.dumps(checked))
        dump(dest/'record.json',record);records.append(record)
        attr=attribution(decision['attribution']);classes[attr.projected_cause.value]+=1
        belief=prior_from_snapshot(decision.get('belief_before'),task)
        visual=CandidateVisualEvidence(**decision['candidate_visual_evidence'][cid])
        planned=np.load(bd/'planned_actions.npy',allow_pickle=False)
        effect=typed_gate_effect(parse_candidate_effect(cid,planned,visual_evidence=visual),relation)
        context=prepare_recovery_context(prior=belief,attribution=attr,
            observations=actual_observations(labels,sm,am),block_index=block)
        trace=trace_candidate_recovery(context,effect,planned)
        empty=prepare_recovery_context(prior=belief,attribution=attr,
            observations=actual_observations(labels,sm,am),block_index=block,graph=DependencyGraph(()))
        nodag=trace_candidate_recovery(empty,effect,planned)
        names,values,masks=label_targets(labels)
        if target_names is None:target_names=names
        if names!=target_names:raise RuntimeError('Target schema drift')
        coverage.update(n for n,m in zip(names,masks) if m)
        feature_rows.append(dict(sample_id=str(dest.relative_to(out)),split=split,
            identity=identity,features=trace.features.tolist(),no_dag_features=nodag.features.tolist(),
            targets=values,masks=masks))
        if len(examples)<16:examples.append(str(dest.relative_to(out)))
        dump(out/'status.json',dict(phase='relabeling',completed=i+1,total=len(rows),
            current_identity=identity,elapsed_seconds=time.time()-started))
        if i%20==0:print(json.dumps(dict(completed=i+1,total=len(rows))),flush=True)
    membership=audit_recovery_membership(records)
    if not membership['ready']:raise RuntimeError(json.dumps(membership))
    # Verify every original recorded input hash after labelling.
    for path,sha in old['source_hashes'].items():
        if file_sha(root/path)!=sha:raise RuntimeError('Original V8 source mutated')
    dump(out/'auxiliary_rows.json',dict(feature_schema=FEATURE_SCHEMA,feature_names=FEATURE_NAMES,
        target_names=target_names,rows=feature_rows))
    report=dict(passed=True,role='auxiliary_only_not_complete_candidate_counterfactuals',
        blocks=len(rows),source_total=834,complete_source=797,short_source_retained=37,
        split=dict(Counter(r['split'] for r in records)),membership=membership,
        observed_target_coverage=dict(coverage),predicted_context_classes=dict(classes),
        unknown_context_is_not_unknown_ground_truth=True,physical_recovery_certified=False,
        no_old_source_mutation=True,training_started=False,elapsed_seconds=time.time()-started,
        examples=examples)
    dump(out/'completion_audit.json',report);dump(out/'status.json',dict(phase='complete',**report))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
