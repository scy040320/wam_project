"""Build V9 inputs with raw recovery labels separated from deployable features.

Old endpoint silver and V8 artifacts are read-only. Positive-close LIBERO
timing is rebuilt, not silently substituted into old feature hashes. Terminal
outcomes are retained only in rank supervision/report records.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from wam_reranking.belief import DependencyGraph
from wam_reranking.candidate_effects import parse_candidate_effect
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.contracts import BeliefState, CandidateVisualEvidence, TriValue
from wam_reranking.evidence_residual import candidate_backbone_features, typed_gate_effect
from wam_reranking.observed_recovery import map_endpoint, labels_from_actual
from wam_reranking.recovery_contract import FEATURE_NAMES, FEATURE_SCHEMA, CurrentFactEvidence, prepare_recovery_context, trace_candidate_recovery
from wam_reranking.recovery_label_contract import audit_recovery_membership, audit_recovery_labels
from wam_reranking.shared_query_cache import file_sha
from wam_reranking.target_localization import CLIPSegTargetLocalizer
try:
    from relabel_archived_recovery import attribution, prior_from_snapshot, label_targets, BINDINGS
except ModuleNotFoundError:
    from scripts.relabel_archived_recovery import attribution, prior_from_snapshot, label_targets, BINDINGS

OLD='candidate_reranking_d21_frozen_d18v8_development_20261004_v2_unobserved_explicit'
HIST='known_task_trusted_evidence_arbitration_20261005_v8'
BACKBONE_SHA='d0780e6680bb99da07d7ced440f2b6604d8bcf489d3bb8455d46499024ac939f'

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def before_observations(subject,anchor):
    """Uses only map(0/1), not future actual arrays, silver masks or success."""
    result=[]
    for predicate,maps in (('target_visible',subject),('receptacle_visible',anchor)):
        qualities=[map_endpoint(m)['quality'] for m in maps[:2]]
        view=int(np.argmax(qualities));quality=qualities[view]
        if quality>=.5:
            result.append(CurrentFactEvidence(predicate,TriValue.TRUE,quality,'actual_before:'+predicate,
                view=('primary','wrist')[view]))
    return result

def traces(prior,attr,observations,block,visual,actions,cid,relation,shuffled_attr):
    # The frozen backbone keeps its historical interpretation. ONLY the new
    # recovery parser/timing uses the verified LIBERO command convention.
    effect=typed_gate_effect(parse_candidate_effect(cid,actions,visual_evidence=visual,close_when_negative=False),relation)
    full=prepare_recovery_context(prior=prior,attribution=attr,observations=observations,block_index=block)
    nodag=prepare_recovery_context(prior=prior,attribution=attr,observations=observations,block_index=block,graph=DependencyGraph(()))
    shuffled=prepare_recovery_context(prior=prior,attribution=shuffled_attr,observations=observations,block_index=block)
    return [trace_candidate_recovery(c,effect,actions).features.tolist() for c in (full,nodag,shuffled)]

def shuffled_index(rows,key,scope):
    """Fixed seed permutes evidence records, never labels, within split/task."""
    groups=defaultdict(list)
    for row in rows:groups[scope(row)].append(key(row))
    result={};rng=np.random.default_rng(917)
    for group,ids in sorted(groups.items()):
        ids=sorted(ids);order=rng.permutation(len(ids))
        for src,position in zip(ids,order):result[src]=ids[int(position)]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--auxiliary',type=Path,required=True);p.add_argument('--derived-audit',type=Path,required=True)
    p.add_argument('--paired',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists():raise RuntimeError('Refuse overwrite training input')
    out.mkdir(parents=True);hashed={}
    def remember(path):
        path=path.resolve();hashed[str(path)]=file_sha(path);return hashed[str(path)]
    derived=json.loads((a.derived_audit/'derived_audit.json').read_text())
    if not derived.get('passed'):raise RuntimeError('797 derived evidence audit not passed')
    paired_audit=json.loads((a.paired/'completion_audit.json').read_text())
    if not paired_audit.get('passed') or paired_audit['candidate_first_blocks']!=160:raise RuntimeError('Bounded paired collection not complete')
    remember(a.paired/'completion_audit.json');remember(a.derived_audit/'derived_audit.json')
    for path,sha in json.loads((a.paired/'derived_sha256.json').read_text()).items():
        if file_sha(a.paired/path)!=sha:raise RuntimeError('Paired immutable evidence hash changed')
    bp=root/'outputs/known_task_evidence_residual_repair_20261005_v2/candidate_backbone.json'
    if remember(bp)!=BACKBONE_SHA:raise RuntimeError('Frozen backbone changed')
    backbone=CandidateUtilityModel.from_record(json.loads(bp.read_text()))
    reference=root/'outputs/known_task_evidence_residual_repair_20261005_v2/training_input.json'
    remember(reference)
    backbone_reference={tuple(r['key']):{c['candidate_id']:c['backbone_features'] for c in r['candidates']}
        for r in json.loads(reference.read_text())['rows']}
    bundle=root/'outputs'/OLD/'training_bundle';helper=load('recovery_original_bundle',root/'research_runs'/OLD/'prepare_d21_training_bundle_v7.py')
    recfile=bundle/'paired_scenarios.json';remember(recfile);old_rows=json.loads(recfile.read_text())
    if len(old_rows)!=192:raise RuntimeError('Original 192 cardinality drift')
    old_index={(int(r['task_id']),int(r['state']),int(r['moment_id']),r['condition']):r for r in old_rows}
    perm=shuffled_index(old_rows,lambda r:(int(r['task_id']),int(r['state']),int(r['moment_id']),r['condition']),
        lambda r:(int(r['task_id']),int(r['moment_id']),'train' if int(r['state'])<=5 else 'val'))
    import torch
    # Match the independent historical silver job, not the numerical flags
    # Cosmos modifies while generating candidates. Frozen attributor inference
    # has its own separately scoped precision and is not called here.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    precision=dict(float32_matmul_precision='highest',cuda_matmul_allow_tf32=False,cudnn_allow_tf32=True)
    detector=CLIPSegTargetLocalizer(device='cuda',model_path=str(root.parent/'model_cache/clipseg-rd64-refined'))
    rank=[];aux=[];members=[];names=None
    for fp in sorted((bundle/'candidate_outcomes').glob('*/results.json')):
        remember(fp)
        for r in json.loads(fp.read_text()):
            task,state,moment=int(r['task_id']),int(r['state']),int(r['moment_id'])
            key=(task,state,moment,r['condition']);rec=old_index[key];split='train' if state<=5 else 'val'
            attr=helper.attribution_output(rec['learned_attribution'],f'original_development:{key}')
            shuffled=helper.attribution_output(old_index[perm[key]]['learned_attribution'],f'shuffled_development:{perm[key]}')
            relation=helper.FROZEN_RELATIONS[(task,moment)]
            images=[]
            for field in ('current_primary_path','current_wrist_path'):
                pp=fp.parent/r[field];remember(pp);images.append(Image.fromarray(np.load(pp,allow_pickle=False)))
            sm=detector.relevance(images,relation.subject);am=detector.relevance(images,relation.anchor)
            obs=before_observations(sm,am);prior=BeliefState(task,relation.subject,relation.anchor)
            for c,v in zip(r['candidates'],rec['candidate_visual_evidence'],strict=True):
                pp=fp.parent/c['actions_path'];remember(pp);actions=np.load(pp,allow_pickle=False)
                visual=CandidateVisualEvidence(**v);cid=int(c['candidate_id'])
                full,nodag,shuf=traces(prior,attr,obs,3,visual,actions,cid,relation.relation,shuffled)
                # Reproduce the old default parser for the old backbone score.
                old_effect=parse_candidate_effect(cid,actions,visual_evidence=visual)
                bx=candidate_backbone_features(old_effect,float(c['value']))
                np.testing.assert_array_equal(bx,np.asarray(backbone_reference[key][cid],dtype=np.float64),
                    err_msg='Frozen candidate-backbone feature replay changed')
                outcome=c['outcome'];clean_outcome={k:outcome[k] for k in ('success','executed_steps','continuation_wam_calls','risk_proxy') if k in outcome}
                rank.append(dict(pool_id=OLD+':'+str(key),suite='libero90',task=task,state=state,split=split,
                    candidate_id=cid,features=full,no_dag_features=nodag,shuffled_features=shuf,
                    backbone_features=bx.tolist(),backbone_score=backbone.score(bx),value=float(c['value']),
                    success=bool(outcome['success']),outcome=clean_outcome))
            dump(out/'status.json',dict(phase='original_rank_features',complete_pools=len(rank)//4,total_pools=192))
    # Historical labels are complete observed blocks but selected-arm only.
    haf=a.auxiliary/'auxiliary_rows.json';remember(haf);historical=json.loads(haf.read_text())['rows']
    if len(historical)!=797:raise RuntimeError('797 auxiliary cardinality drift')
    # Independent, deterministic bitwise teacher replay checks before reusing
    # old maps. Do not silently replace old silver if this invariant fails.
    for r in historical[::80][:10]:
        cd=a.auxiliary/r['sample_id'];task=r['identity']['task']
        images=[Image.fromarray(np.load(cd/(v+'.npy'),allow_pickle=False)[step]) for step in (0,16) for v in ('primary','wrist')]
        subject,anchor,_=BINDINGS[task]
        sm=detector.relevance(images,subject);am=detector.relevance(images,anchor)
        with np.load(cd/'observer_maps.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(sm,z['subject'],err_msg='Historical frozen teacher precision replay changed')
            np.testing.assert_array_equal(am,z['anchor'],err_msg='Historical frozen teacher precision replay changed')
    decisions={}
    for r in historical:
        folder=root/r['identity']['observed_block_id'];block=int(folder.name.split('_')[-1])
        path=folder.parent/f'decision_{block}.json';remember(path);decisions[r['sample_id']]=json.loads(path.read_text())
    hmap={r['sample_id']:r for r in historical};hp=shuffled_index(historical,lambda r:r['sample_id'],lambda r:(r['identity']['task'],r['split']))
    for r in historical:
        dest=a.auxiliary/r['sample_id'];rec=json.loads((dest/'record.json').read_text());members.append(rec)
        for name in ('record.json','labels.json','observer_maps.npz'):remember(dest/name)
        label=json.loads((dest/'labels.json').read_text())
        checked=audit_recovery_labels(rec,directory=dest)
        if not checked['ready']:raise RuntimeError(json.dumps(checked))
        with np.load(dest/'observer_maps.npz',allow_pickle=False) as z:sm=z['subject'];am=z['anchor']
        dec=decisions[r['sample_id']];idn=r['identity'];task=idn['task'];folder=root/idn['observed_block_id'];block=int(folder.name.split('_')[-1])
        pp=folder/'evidence/block/planned_actions.npy';remember(pp);actions=np.load(pp,allow_pickle=False)
        attr=attribution(dec['attribution']);shuf=attribution(decisions[hp[r['sample_id']]]['attribution'])
        full,nodag,shuffled=traces(prior_from_snapshot(dec.get('belief_before'),task),attr,before_observations(sm,am),block,
            CandidateVisualEvidence(**dec['candidate_visual_evidence'][int(idn['candidate_id'])]),actions,int(idn['candidate_id']),
            'articulated' if task==0 else ('on' if task==9 else 'inside'),shuf)
        tn,values,masks=label_targets(label)
        if names is None:names=tn
        if names!=tn:raise RuntimeError('Target order drift')
        aux.append(dict(row_id=HIST+':'+r['sample_id'],identity=idn,role='historical_selected_arm_auxiliary',split=r['split'],
            features=full,no_dag_features=nodag,shuffled_features=shuffled,targets=values,masks=masks))
    pf=a.paired/'pools.json';remember(pf);pools=json.loads(pf.read_text())
    pi={r['pool_id']:r for r in pools};sp=shuffled_index(pools,lambda r:r['pool_id'],lambda r:(r['task'],r['split']))
    changed_labels=0;paired_targets_by_pool={}
    for pool in pools:
        pool_dir=a.paired/'pools'/pool['pool_id'];task=pool['task']
        attr_path=pool_dir/'attribution.json';remember(attr_path);attr=helper.attribution_output(json.loads(attr_path.read_text()),pool['pool_id'])
        shuffled_path=a.paired/'pools'/sp[pool['pool_id']]/'attribution.json';remember(shuffled_path)
        shuf=helper.attribution_output(json.loads(shuffled_path.read_text()),sp[pool['pool_id']])
        for c in pool['candidates']:
            if not c['complete']:continue # Retained in collection audit, never padded.
            cd=a.paired/c['directory'];rec=json.loads((cd/'record.json').read_text());members.append(rec)
            for name in ('record.json','labels.json','observer_maps.npz'):remember(cd/name)
            if not audit_recovery_labels(rec,directory=cd)['ready']:raise RuntimeError('Paired label audit failed')
            label=json.loads((cd/'labels.json').read_text())
            # Raw executions stay immutable. Generate precision-canonical
            # sidecars; hardlinks reference the exact arrays, never modify them.
            canonical=out/'canonical_paired'/pool['pool_id']/f'candidate_{c["candidate_id"]}'
            canonical.mkdir(parents=True,exist_ok=False)
            for filename in rec['arrays'].values():
                os.link(cd/filename,canonical/filename)
            images=[Image.fromarray(np.load(cd/(v+'.npy'),allow_pickle=False)[step]) for step in (0,16) for v in ('primary','wrist')]
            subject_prompt,anchor_prompt,_=BINDINGS[task]
            sm=detector.relevance(images,subject_prompt);am=detector.relevance(images,anchor_prompt)
            original_label=label
            label=labels_from_actual(identity=rec['identity'],subject_maps=sm,anchor_maps=am,
                requested=np.load(cd/'requested.npy',allow_pickle=False),applied=np.load(cd/'applied.npy',allow_pickle=False))
            changed_labels+=label!=original_label
            dump(canonical/'labels.json',label);dump(canonical/'record.json',rec)
            np.savez_compressed(canonical/'observer_maps.npz',subject=sm,anchor=am)
            if not audit_recovery_labels(rec,directory=canonical)['ready']:raise RuntimeError('Canonical precision labels invalid')
            pp=cd/'planned_actions.npy';remember(pp);actions=np.load(pp,allow_pickle=False)
            vp=cd/'candidate_visual.json';remember(vp);visual=CandidateVisualEvidence(**json.loads(vp.read_text()))
            subject,anchor=helper.FROZEN_RELATIONS[(task,0)].subject,helper.FROZEN_RELATIONS[(task,0)].anchor
            full,nodag,shuffled=traces(BeliefState(task,subject,anchor),attr,before_observations(sm,am),3,visual,
                actions,c['candidate_id'],pool['relation'],shuf)
            tn,values,masks=label_targets(label)
            if names!=tn:raise RuntimeError('Paired target order drift')
            paired_targets_by_pool.setdefault(pool['pool_id'],[]).append((values,masks))
            aux.append(dict(row_id=a.paired.name+':'+c['directory'],identity=rec['identity'],role='paired_recovery_supervision',split=pool['split'],
                features=full,no_dag_features=nodag,shuffled_features=shuffled,targets=values,masks=masks))
    membership=audit_recovery_membership(members)
    if not membership['ready']:raise RuntimeError(json.dumps(membership))
    group_roles={}
    for r in rank+aux:
        ident=r.get('identity',r);group=('libero90',int(ident['task']),int(ident['state']))
        if group in group_roles and group_roles[group]!=r['split']:raise RuntimeError('Cross-source group leakage')
        group_roles[group]=r['split']
    if len(rank)!=768 or Counter(r['split'] for r in rank)!=dict(train=576,val=192):raise RuntimeError('Original rank split drift')
    for path,sha in hashed.items():
        if file_sha(Path(path))!=sha:raise RuntimeError('Original input changed')
    contract=Path(__file__).with_name('recovery_supervision_contract_v9.json')
    remember(contract)
    audit=dict(schema='candidate_recovery_supervision_v1',recovery_training_ready=True,group_leakage=False,
        deployment_gt_leakage=False,feature_schema=FEATURE_SCHEMA,feature_count=len(FEATURE_NAMES),
        historical_labels=797,paired_complete_labels=paired_audit['complete_labels'],
        original_rank_candidates=768,rank_split=dict(Counter(r['split'] for r in rank)),
        expected_rank_rows=len(rank),expected_auxiliary_rows=len(aux),
        auxiliary_split=dict(Counter(r['split'] for r in aux)),membership=membership,
        unobservable_physical_targets_masked=True,ordinary_success_gain_not_automatically_attribution_gain=True,
        new_features_positive_close=True,old_backbone_default_parser_preserved=True,
        shuffled_seed=917,shuffled_labels=False,source_hashes=hashed,contract_sha256=file_sha(contract))
    variation={}
    for pid,values_masks in paired_targets_by_pool.items():
        varied=[]
        for i,name in enumerate(names):
            observed=[v[i] for v,m in values_masks if m[i]]
            if len(observed)>=2 and max(observed)!=min(observed):varied.append(name)
        variation[pid]=dict(complete_candidates=len(values_masks),varying_observable_targets=varied,
            no_observed_recovery_variation=not varied,retained=True)
    audit['observer_numeric_precision']=precision
    audit['canonical_label_prompts']=BINDINGS
    audit['historical_teacher_bitwise_replay_examples']=10
    audit['paired_labels_canonicalized_before_fit']=paired_audit['complete_labels']
    audit['paired_numerical_label_changes']=changed_labels
    audit['paired_recovery_variation']=variation
    dump(out/'uniform_precision_audit.json',dict(passed=True,precision=precision,
        old_teacher_replay_exact=True,original_old_and_new_sidecars_preserved=True,
        canonical_paired_label_changes=changed_labels,no_new_label_rule_or_threshold=True,
        canonical_prompts=BINDINGS,old_prompt_contract_inherited=True,
        shared_actual_raw_arrays_immutable=True))
    dump(out/'training_audit.json',audit)
    dump(out/'training_input.json',dict(schema='recovery_ranker_training_input_v1',suite='libero90',
        feature_schema=FEATURE_SCHEMA,feature_names=FEATURE_NAMES,target_names=names,
        target_encoding='raw_signed_transition; fit maps 0.5+0.5*d',training_audit=audit,
        rank_rows=rank,auxiliary_rows=aux))
    dump(out/'status.json',dict(phase='complete',**audit))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
