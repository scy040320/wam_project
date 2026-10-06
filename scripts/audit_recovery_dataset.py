"""Audit the frozen candidate pool for a genuinely connected recovery V9.

No model fitting, rollouts, collection, or deployment mutation occurs here.
Missing actual first-block evidence stays masked; legacy episode outcomes are
retained as auxiliary supervision, never relabelled as predicate recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from wam_reranking import CandidateVisualEvidence, parse_candidate_effect
from wam_reranking.contracts import BeliefState, CoarseCause, PREDICATES, TriValue
from wam_reranking.evidence_residual import typed_gate_effect
from wam_reranking.recovery_contract import (
    SCHEMA, FEATURE_NAMES, prepare_recovery_context, trace_candidate_recovery,
    require_recovery_training_ready,
)
from wam_reranking.recovery_supervision import audit_recovery_supervision
from wam_reranking.shared_query_cache import file_sha

SOURCE = 'candidate_reranking_d21_frozen_d18v8_development_20261004_v2_unobserved_explicit'


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.item()
        if isinstance(x,np.generic) else x.value if hasattr(x,'value') else str(x))+'\n')


def module(path):
    spec=importlib.util.spec_from_file_location('frozen_recovery_bindings',path)
    obj=importlib.util.module_from_spec(spec);sys.modules[spec.name]=obj;spec.loader.exec_module(obj)
    return obj


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--request-training',action='store_true',
        help='Only validate eligibility; never fit a model in this audit entry point')
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    hashes={}
    def tracked(path):
        path=Path(path).resolve()
        if not path.is_file() or not path.stat().st_size:raise ValueError('Missing critical source: '+str(path))
        if root not in path.parents:raise ValueError('Source escaped cloud project: '+str(path))
        hashes[str(path.relative_to(root))]=file_sha(path)
        return path
    contract=json.loads(tracked(args.contract).read_text())
    if contract['schema']!=SCHEMA:raise ValueError('Frozen recovery contract mismatch')
    dump(out/'frozen_recovery_contract.json',contract)
    bundle=root/'outputs'/SOURCE/'training_bundle'
    records=json.loads(tracked(bundle/'paired_scenarios.json').read_text())
    tracked(bundle/'learned_predictions.json')
    helper_path=root/'research_runs'/SOURCE/'prepare_d21_training_bundle_v7.py'
    helper=module(tracked(helper_path))
    index={}
    for p in sorted((bundle/'candidate_outcomes').glob('*/results.json')):
        for r in json.loads(tracked(p).read_text()):
            key=(r['task_id'],r['state'],r['moment_id'],r['condition'])
            if key in index:raise ValueError('Duplicate source-scoped pool')
            index[key]=(p.parent,r)
    split=Counter(r['split'] for r in records)
    if split!=Counter({'train':144,'val':48}) or len(index)!=192:
        raise ValueError('Frozen source count drift')
    groups={s:{('libero90',r['task_id'],r['state']) for r in records if r['split']==s} for s in split}
    if groups['train'] & groups['val']:raise ValueError('Group leakage')
    label_counts=Counter();predicted_counts=Counter();files=Counter();missing=Counter()
    mixed=Counter();ready_by_route=Counter();candidates=[];pool_traces=[];seen=set()
    questionable_claims=0
    for r in sorted(records,key=lambda a:(a['task_id'],a['state'],a['moment_id'],a['condition'])):
        key=(r['task_id'],r['state'],r['moment_id'],r['condition'])
        folder,source=index[key]
        if key in seen or r['state']>7:raise ValueError('Duplicate or consumed acceptance state')
        seen.add(key);label_counts[source['label']]+=1
        attr=helper.attribution_output(r['learned_attribution'],'recovery_contract:'+str(key))
        predicted_counts[attr.projected_cause.value]+=1
        # Do not import historic initial_belief's unsupported true constants.
        binding=helper.TASK_BINDINGS[r['task_id']]
        prior=BeliefState(binding.task_id,binding.target_object,binding.receptacle)
        context=prepare_recovery_context(prior=prior,attribution=attr,block_index=3)
        current=[np.load(tracked(folder/source['current_'+view+'_path']),allow_pickle=False)
                 for view in ('primary','wrist')]
        if any(a.ndim!=3 or a.shape[-1]!=3 or not np.isfinite(a).all() for a in current):
            raise ValueError('Invalid actual current image')
        entry=root/source['evidence_block']
        block=json.loads(tracked(entry/'block.json').read_text())
        control=root/index[(*key[:3],'clean')][1]['evidence_block']
        clean=json.loads(tracked(control/'block.json').read_text())
        for field in ('query_observation_sha256','planned_actions_sha256','start_state_sha256'):
            if block[field]!=clean[field]:raise ValueError('Physical pairing contract drift')
        relation=helper.FROZEN_RELATIONS[(key[0],key[2])].relation
        if len(source['candidates'])!=4 or len(r['candidate_visual_evidence'])!=4:
            raise ValueError('Candidate count drift')
        outcomes=[bool(c['outcome']['success']) for c in source['candidates']]
        if any(outcomes) and not all(outcomes):mixed[r['split']]+=1
        row_trace=[]
        ids=set()
        for c,v in zip(source['candidates'],r['candidate_visual_evidence'],strict=True):
            cid=c['candidate_id']
            if cid in ids:raise ValueError('Duplicate candidate')
            ids.add(cid)
            actions=np.load(tracked(folder/c['actions_path']),allow_pickle=False)
            if actions.shape!=(16,7) or not np.isfinite(actions).all():raise ValueError('Bad action chunk')
            for view in ('primary','wrist'):
                prediction=np.load(tracked(folder/c['predicted_'+view+'_path']),allow_pickle=False)
                if prediction.ndim!=3 or prediction.shape[-1]!=3 or not np.isfinite(prediction).all():
                    raise ValueError('Bad predicted image')
            effect=parse_candidate_effect(cid,actions,visual_evidence=CandidateVisualEvidence(**v))
            typed=typed_gate_effect(effect,relation)
            trace=trace_candidate_recovery(context,typed,actions)
            # Existing endpoint-derived pre-release hints are not temporal witnesses.
            questionable_claims+=bool(effect.evidence.get('establishes_place_ready_before_release',0.))
            actual=c.get('recovery_execution')
            audit=audit_recovery_supervision(actual,directory=(folder/c['actions_path']).parent,tracked=tracked)
            if actual and actual.get('candidate_id')!=cid:
                audit['ready']=False;audit['missing'].append('candidate_identity')
            missing.update(audit['missing'])
            if audit['ready']:ready_by_route[source['label']]+=1
            files['complete_recovery_records']+=int(audit['ready'])
            files['candidate_actions']+=1;files['predicted_dual_views']+=1
            info=dict(source_key=SOURCE,suite='libero90',key=list(key),split=r['split'],candidate_id=cid,
                role='recovery_supervised' if audit['ready'] else 'legacy_terminal_success_auxiliary',
                recovery_audit=audit,terminal_outcome_available=True)
            candidates.append(info)
            # Feature envelope contains no condition, success, sim state or label.
            row_trace.append(dict(candidate_id=cid,features=trace.features.tolist(),
                hard_reasons=trace.hard_reasons,unresolved_requirements=trace.unresolved_requirements,
                command_order=trace.command_order,facts=trace.facts))
        pool_traces.append(dict(key=list(key),split=r['split'],role='audit_only_not_training',
            belief=context.belief.snapshot(),candidate_traces=row_trace))
    coverage={c.value:ready_by_route[c.value] for c in CoarseCause}
    ready=files['complete_recovery_records']==768 and all(n>0 for n in coverage.values())
    problems=[]
    if files['complete_recovery_records']!=768:problems.append('actual_candidate_first_block_recovery_evidence_missing')
    if any(n==0 for n in coverage.values()):problems.append('five_route_observable_supervision_not_covered')
    report=dict(schema=SCHEMA,experiment_alias='recovery_conditioned_ranker_V9',
        source=SOURCE,legacy_integrity_passed=True,recovery_training_ready=ready,
        group_leakage=False,deployment_gt_leakage=False,pools=192,candidates=768,
        split=dict(split),mixed_pools=dict(mixed),protocol_label_pools=dict(label_counts),
        predicted_attribution_pools=dict(predicted_counts),recovery_supervised_candidates=dict(coverage),
        inventory=dict(files),missing_fields=dict(missing),blocking_reasons=problems,
        endpoint_pre_release_claims_without_intermediate_witness=int(questionable_claims),
        feature_count=len(FEATURE_NAMES),all_five_routes_implemented=True,
        unknown_prediction_is_not_unknown_supervision=True,
        no_prediction_to_actual_label_conversion=True,no_final_success_to_predicate_conversion=True,
        no_data_deleted=True,no_collection=True,no_deployment_update=True,
        independent128_started=False,training_started=False,
        legacy_data_role='terminal_success_auxiliary_only_until_actual_recovery_evidence_exists')
    for path,sha in hashes.items():
        if file_sha(root/path)!=sha:raise ValueError('Frozen input changed during audit')
    dump(out/'candidate_supervision_index.json',candidates)
    dump(out/'coupled_candidate_traces.json',pool_traces)
    dump(out/'source_sha256.json',hashes)
    dump(out/'recovery_data_audit.json',report)
    dump(out/'status.json',dict(phase='audit_complete' if ready else 'audit_complete_training_blocked',
        training_started=False,blocking_reasons=problems,independent128_started=False))
    print(json.dumps(report,indent=2))
    if args.request_training:require_recovery_training_ready(report)


if __name__=='__main__':main()
