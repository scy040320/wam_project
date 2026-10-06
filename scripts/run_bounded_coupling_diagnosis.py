"""One preregistered CPU diagnosis on existing 144/48 candidate pools.

No rollout, collector, attributor training or deployment update is performed.
The protocol-supervised oracle is an upper diagnostic, not human causal truth.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from wam_reranking import CandidateVisualEvidence, initial_belief, parse_candidate_effect
from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.contracts import AttributionOutput, CoarseCause, ConsistencyFactor, EvidenceQuality, TriValue
from wam_reranking.coupling_diagnostics import GAINS, check_membership, fit_diagnostic
from wam_reranking.contextual_cause_residual import contextual_cause_features, fit_context_residual
from wam_reranking.evidence_residual import candidate_backbone_features, cause_residual_features, typed_gate_effect
from wam_reranking.shared_query_cache import file_sha
from wam_reranking.source_scoped_policy import prepare_source_scoped_decisions

SOURCE = 'candidate_reranking_d21_frozen_d18v8_development_20261004_v2_unobserved_explicit'
PRIOR = 'known_task_evidence_residual_repair_20261005_v2'
PANELS = ('ranking_only', 'current_variant_gate')
VARIANTS = ('none', 'predicted', 'protocol_oracle', 'shuffled')
SEED = 20261006


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    return mod


def protocol_oracle(condition, world_observable, execution_measured):
    """Offline diagnostic-only intervention cause with observability masking."""
    values = {f.value:1. for f in ConsistencyFactor}
    cause = {'clean':CoarseCause.NORMAL,'visual_occlusion':CoarseCause.VISUAL_OCCLUSION,
        'object_shift':CoarseCause.OBJECT_SHIFT,'execution_contact_deviation':CoarseCause.EXECUTION_CONTACT_DEVIATION}[condition]
    states = {f.value:TriValue.TRUE for f in ConsistencyFactor}
    quality = EvidenceQuality(True,True,True)
    if condition == 'visual_occlusion':
        values['observation_reliable']=0.; states['observation_reliable']=TriValue.FALSE
        quality=EvidenceQuality(False,True,True)
    elif condition == 'object_shift':
        if world_observable:
            values['world_state_consistent']=0.;states['world_state_consistent']=TriValue.FALSE
        else:
            cause=CoarseCause.UNKNOWN;values['cause_resolved']=0.;states['cause_resolved']=TriValue.FALSE
            values['world_state_consistent']=.5;states['world_state_consistent']=TriValue.UNKNOWN
    elif condition == 'execution_contact_deviation':
        if execution_measured:
            values['execution_contact_consistent']=0.;states['execution_contact_consistent']=TriValue.FALSE
        else:
            cause=CoarseCause.UNKNOWN;values['cause_resolved']=0.;states['cause_resolved']=TriValue.FALSE
            values['execution_contact_consistent']=.5;states['execution_contact_consistent']=TriValue.UNKNOWN
    return AttributionOutput(values,{c.value:float(c is cause) for c in CoarseCause},cause,1.,0.,quality,
        'protocol_oracle_offline_only_not_human_ground_truth',states,
        {f.value:1. if states[f.value] is not TriValue.UNKNOWN else 0. for f in ConsistencyFactor})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--contextual-repair',action='store_true')
    args=ap.parse_args();root=args.root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    paths={}
    def tracked(path):
        path=Path(path);paths[str(path.relative_to(root))]=file_sha(path);return path
    bundle=root/'outputs'/SOURCE/'training_bundle';code=root/'research_runs'/SOURCE
    backbone=CandidateUtilityModel.from_record(json.loads(tracked(root/'outputs'/PRIOR/'candidate_backbone.json').read_text()))
    for filename in ('prepare_d21_training_bundle_v7.py','train_d21_candidate_utility_v7.py'):tracked(code/filename)
    helper=load_module('bounded_frozen_binding',code/'prepare_d21_training_bundle_v7.py')
    loader=load_module('bounded_frozen_pool_loader',code/'train_d21_candidate_utility_v7.py')
    records=json.loads(tracked(bundle/'paired_scenarios.json').read_text())
    tracked(bundle/'learned_predictions.json')
    old=json.loads(tracked(root/'outputs'/PRIOR/'training_input.json').read_text())['rows']
    old_index={tuple(r['key']):r for r in old}
    pool_index={}
    for p in sorted((bundle/'candidate_outcomes').glob('*/results.json')):
        for r in json.loads(tracked(p).read_text()):pool_index[(r['task_id'],r['state'],r['moment_id'],r['condition'])]=(p.parent,r)
    scenarios=loader.load_scenarios(bundle)
    scenario_index={tuple(r['key']):r for r in scenarios}
    if len(records)!=192 or len(pool_index)!=192 or len(scenario_index)!=192:raise ValueError('Expected 192 unique pools')
    ordered=sorted(records,key=lambda r:(r['task_id'],r['state'],r['moment_id'],r['condition']))
    pred=[helper.attribution_output(r['learned_attribution'],f'diagnostic:{i}') for i,r in enumerate(ordered)]
    rng=np.random.default_rng(SEED);donors={}
    for split in ('train','val'):
        ix=np.asarray([i for i,r in enumerate(ordered) if r['split']==split])
        shuffled=rng.permutation(ix)
        for i,j in zip(ix,shuffled,strict=True):donors[int(i)]=int(j)
    oracle,raw,sources=[],[],[]
    for i,rec in enumerate(ordered):
        key=(rec['task_id'],rec['state'],rec['moment_id'],rec['condition']);folder,r=pool_index[key]
        clean=pool_index[(*key[:3],'clean')][1]
        evidence=root/r['evidence_block'];control=root/clean['evidence_block']
        b=json.loads(tracked(evidence/'block.json').read_text());bc=json.loads(tracked(control/'block.json').read_text())
        for name in ('query_observation_sha256','planned_actions_sha256','start_state_sha256'):
            if b[name]!=bc[name]:raise RuntimeError('Pair contract mismatch: '+str(key)+' '+name)
        def array(directory,name):return np.load(tracked(directory/name),allow_pickle=False)
        visible_difference=max(float(np.abs(array(evidence,'actual_observed_'+view+'.npy').astype(float)
            -array(control,'actual_observed_'+view+'.npy').astype(float)).mean()) for view in ('primary','wrist'))
        applied=array(evidence,'applied_actions.npy');requested=array(evidence,'requested_actions.npy')
        execution_difference=float(np.max(np.abs(applied-requested)))
        oracle.append(protocol_oracle(key[3],visible_difference>2.,execution_difference>1e-8))
        sc=scenario_index[key];effects=[]
        for c,v in zip(sc['candidates'],rec['candidate_visual_evidence'],strict=True):
            stored=next(ca for ca in r['candidates'] if ca['candidate_id']==c['candidate_id'])
            effects.append(parse_candidate_effect(c['candidate_id'],array(folder,stored['actions_path']),
                visual_evidence=CandidateVisualEvidence(**v)))
        relation=helper.FROZEN_RELATIONS[(key[0],key[2])].relation
        raw.append(dict(key=list(key),split=rec['split'],relation=relation,effects=effects,candidates=sc['candidates']))
        sources.append(dict(key=list(key),oracle_cause=oracle[-1].projected_cause.value,
            predicted_cause=pred[i].projected_cause.value,image_mean_abs_max=visible_difference,
            applied_requested_max_abs=execution_difference,shuffled_donor_key=[ordered[donors[i]][k] for k in ('task_id','state','moment_id','condition')]))
        # Exact backbone reconstruction prevents a changed candidate parser
        # from being credited as an attribution effect.
        for c,e in zip(sc['candidates'],effects,strict=True):
            baseline=next(x for x in old_index[key]['candidates'] if x['candidate_id']==c['candidate_id'])
            if not np.array_equal(candidate_backbone_features(e,c['value']),baseline['backbone_features']):
                raise RuntimeError('Frozen backbone feature drift: '+str(key))
    for p in Path(__file__).parent.rglob('*.py'):
        if '__pycache__' not in str(p):tracked(p)
    protocol=dict(role='bounded_consumed_development_coupling_diagnosis_not_independent',
        counts={'train':144,'val':48,'pools':192,'candidates':768},
        variants=list(VARIANTS),panels=list(PANELS),gain_grid=list(GAINS),shuffle_seed=SEED,
        gain_selection='train_only_success_max_then_smallest_gain_zero_success_harm',
        oracle='protocol_intervention_cause_not_human_ground_truth; render<=2 unknown; absent command deviation unknown',
        oracle_not_deployable=True,frozen_inputs=paths,
        validation_is_consumed_development=True,acceptance32_not_loaded=True,
        no_rollouts=True,no_collection=True,no_attributor_update=True,no_auto_deploy=True,no_auto_128=True,
        success_only_loss=True,costs_are_reported_not_optimized=True,
        contextual_repair=args.contextual_repair,context_feature_count=70 if args.contextual_repair else 10,
        no_additional_hyperparameter_search=True,loss_l2=1.,optimizer_steps=2000,learning_rate=.05)
    dump(out/'frozen_diagnostic_protocol.json',protocol)
    for p,sha in paths.items():
        if file_sha(root/p)!=sha:raise RuntimeError('Frozen input changed before fitting')
    variant_rows={}
    for variant in VARIANTS:
        rows=[]
        for i,datum in enumerate(raw):
            attribution=pred[i] if variant=='predicted' else oracle[i] if variant=='protocol_oracle' else pred[donors[i]] if variant=='shuffled' else None
            effects,cs=datum['effects'],datum['candidates']
            if attribution is None:
                from wam_reranking.contracts import CandidateDecision
                decisions=[CandidateDecision(c['candidate_id'],True,c['value'],c['value'],(),{}) for c in cs]
            else:
                decisions=prepare_source_scoped_decisions(belief=initial_belief(datum['key'][0],bindings=helper.TASK_BINDINGS),
                    attribution=attribution,block_index=3,effects=effects,values=[c['value'] for c in cs],
                    relation=datum['relation'],backbone=backbone)
            features={};supervision={}
            for c,e,d in zip(cs,effects,decisions,strict=True):
                cid=c['candidate_id'];features[cid]=dict(backbone=candidate_backbone_features(e,c['value']),
                    cause=np.zeros(10) if attribution is None else cause_residual_features(typed_gate_effect(e,datum['relation']),attribution,d),decision=d)
                if args.contextual_repair:
                    features[cid]['cause']=contextual_cause_features(features[cid]['backbone'],features[cid]['cause'],attribution)
                supervision[cid]=dict(success=bool(c['outcome']['success']),steps=float(c['outcome']['executed_steps']),calls=float(c['outcome']['continuation_wam_calls']))
            rows.append(dict(key=datum['key'],split=datum['split'],features=features,supervision=supervision,attribution=attribution))
        check_membership(rows)
        if Counter(row['split'] for row in rows) != Counter({'train':144,'val':48}):
            raise RuntimeError('Frozen diagnostic split count mismatch')
        if any(len(row['features']) != 4 for row in rows):
            raise RuntimeError('Frozen diagnostic candidate count mismatch')
        variant_rows[variant]=rows
    reports={};missed=[]
    for panel in PANELS:
        reports[panel]={}
        for variant in VARIANTS:
            result=fit_diagnostic(variant_rows[variant],backbone,panel=panel,
                **({'fitter':fit_context_residual} if args.contextual_repair else {}))
            reports[panel][variant]=result['report']
            dump(out/panel/variant/'diagnostic_model_do_not_deploy.json',dict(oracle_or_shuffled_never_deploy=True,
                role='development_diagnostic_only',variant=variant,model=result['model'].to_record()))
    for panel,variants in reports.items():
        entries=variants['none']['train']['entries']+variants['none']['validation']['entries']
        for e in entries:
            if not e['covered'] or e['backbone_success']:continue
            key=tuple(e['key']);index=next(i for i,r in enumerate(raw) if tuple(r['key'])==key)
            detail=dict(panel=panel,key=list(key),split=raw[index]['split'],source=sources[index],variants={})
            for variant,rows in variant_rows.items():
                detail['variants'][variant]=[dict(candidate_id=cid,backbone_score=backbone.score(c['backbone']),
                    cause_features=c['cause'].tolist(),accepted=c['decision'].accepted,
                    rejection_reasons=c['decision'].rejection_reasons,
                    supervised_success=rows[index]['supervision'][cid]['success']) for cid,c in rows[index]['features'].items()]
            missed.append(detail)
    gate_panel={variant:{split:reports['current_variant_gate'][variant][split]['summary'] for split in ('train','validation')} for variant in VARIANTS}
    def incremental(variant):
        r=gate_panel[variant]
        return all(r[s]['gains_vs_backbone']>0 and r[s]['harms_vs_backbone']==r[s]['harms_vs_value']==r[s]['unobserved_fallback']==0 for s in r)
    summary=dict(quality_passed=True,comparisons=gate_panel,
        pure_ranking={v:{s:reports['ranking_only'][v][s]['summary'] for s in ('train','validation')} for v in VARIANTS},
        predicted_increment_train_and_val_zero_harm=incremental('predicted'),
        protocol_oracle_increment_train_and_val_zero_harm=incremental('protocol_oracle'),
        attribution_confusion=dict(Counter(s['oracle_cause']+' -> '+s['predicted_cause'] for s in sources)),
        consumed_validation_not_independent=True,oracle_not_human_truth=True,
        deployment_models_unchanged=True,independent_128_not_started=True,
        feature_reconstruction_exact=True)
    dump(out/'source_and_oracle_audit.json',dict(passed=True,entries=sources))
    dump(out/'diagnostic_report.json',reports);dump(out/'missed_opportunity_cases.json',missed)
    dump(out/'diagnostic_summary.json',summary)
    for p,sha in paths.items():
        if file_sha(root/p)!=sha:raise RuntimeError('Source changed during diagnostic fitting')
    dump(out/'status.json',dict(phase='complete',quality_passed=True,no_rollouts=True,no_auto_128=True))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
