"""Isolated access to the COMPLETE immutable 64-scene V8, never the Oct-1 ranker."""
from __future__ import annotations
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('/root/autodl-tmp/wam_project/cosmos-policy')
V8='known_task_trusted_evidence_arbitration_20261005_v8'
V8_CODE=ROOT/'research_runs'/V8
V8_OUT=ROOT/'outputs'/V8
ORIGINAL=ROOT/'outputs/candidate_reranking_known_task_closedloop_mechanism_20261004_v3_shared_source'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(V8_CODE/'overlay'))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

frozen=load('immutable_complete_v8_execution',V8_CODE/'run_arbitrated_closedloop.py')
api=load('wam_reranking.terminal_conditioned_residual',HERE/'terminal_conditioned_residual.py')
wrapper=load('wam_reranking.full_v8_conditioned_residual',HERE/'full_v8_conditioned_residual.py')
from wam_reranking import CandidateUtilityModel,CandidateVisualEvidence,initial_belief,parse_candidate_effect
from wam_reranking.contracts import AttributionOutput,BeliefChange,BeliefFact,BeliefState,CoarseCause,EvidenceQuality,Stage,TriValue
from wam_reranking.evidence_arbitration import prepare_arbitrated_decisions,select_evidence_arbitration
from wam_reranking.evidence_residual import EvidenceResidualModel,candidate_backbone_features,cause_residual_features,typed_gate_effect
from wam_reranking.reranker import select_candidate

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+'\n');tmp.replace(path)

def frozen_models():
    return (CandidateUtilityModel.from_record(read(V8_OUT/'candidate_only_model.json')),
            EvidenceResidualModel.from_record(read(V8_OUT/'evidence_residual_model.json')))

def attribution_from_snapshot(row):
    r=copy.deepcopy(row);r['projected_cause']=CoarseCause(r['projected_cause'])
    r['evidence_quality']=EvidenceQuality(**r['evidence_quality'])
    r['factor_states']={k:TriValue(v) for k,v in r.get('factor_states',{}).items()}
    return AttributionOutput(**r)

def belief_from_snapshot(row):
    facts={k:BeliefFact(TriValue(v['value']),float(v['confidence']),v['source'],
                       v['updated_at_block'],tuple(v['evidence_ids'])) for k,v in row['facts'].items()}
    history=[]
    for item in row['history']:
        v=dict(item);v['old_value']=TriValue(v['old_value']);v['new_value']=TriValue(v['new_value'])
        v['propagation_path']=tuple(v['propagation_path']);history.append(BeliefChange(**v))
    return BeliefState(row['task_id'],row['target_object'],row['receptacle'],Stage(row['task_stage']),facts,history)

def build_context(prior,attr,plans,values,visuals,relation,block,backbone,residual):
    effects=[parse_candidate_effect(i,a,visual_evidence=v) for i,(a,v) in enumerate(zip(plans,visuals,strict=True))]
    # Exactly ONE original belief update in the live selection path.
    decisions=prepare_arbitrated_decisions(belief=prior,attribution=attr,block_index=block,
        effects=effects,values=values,relation=relation,backbone=backbone)
    xs={i:candidate_backbone_features(e,values[i]) for i,e in enumerate(effects)}
    zs={i:cause_residual_features(typed_gate_effect(e,relation),attr,decisions[i]) for i,e in enumerate(effects)}
    kw=dict(decisions=decisions,backbone_features=xs,cause_features=zs,backbone=backbone,
            frozen_residual=residual,attribution=attr)
    chosen,scores,_=wrapper.select_full_v8_conditioned(**kw,deltas=np.zeros(len(plans)))
    fallback=None
    if chosen is None:_,fallback=select_candidate(decisions,attr)
    return effects,kw,chosen,scores,fallback

def load_residual(path):
    r=read(path)
    if r['schema']!=api.SCHEMA or tuple(r['feature_names'])!=api.FEATURE_NAMES or r['cap']!=api.CAP:
        raise ValueError('Frozen terminal residual interface changed')
    return api.TerminalResidual(np.asarray(r['scale']),np.asarray(r['weights']),np.asarray(r['support']),r['gain'])

def redirect_execution(out,version):
    frozen.OUT=out;frozen.VERSION=version
    for m in (frozen.paired,frozen.paired.repair,frozen.paired.legacy):m.OUT=out;m.VERSION=version

def audit_original_decisions(out):
    """All recorded blocks of all 64 scenes; no outcomes are used for fitting."""
    backbone,residual=frozen_models();protocol=read(V8_OUT/'frozen_protocol.json')
    models={str(p):sha(p) for p in (V8_OUT/'candidate_only_model.json',V8_OUT/'evidence_residual_model.json')}
    if models[str(V8_OUT/'candidate_only_model.json')]!='d0780e6680bb99da07d7ced440f2b6604d8bcf489d3bb8455d46499024ac939f':
        raise ValueError('Not the complete V8 candidate backbone')
    checks=[];pins={**models};bundle=frozen.paired.legacy.load('v10_immutable_conversion',frozen.paired.legacy.parent.REF/'prepare_d21_training_bundle_v7.py')
    for task in protocol['tasks']:
        for state in protocol['task_states'][str(task)]:
            for condition in protocol['conditions']:
                name=f'task{task}_state{state}_{condition}';arm=V8_OUT/'scenarios'/name/'full_repaired'
                outcome=read(arm/'outcome.json');pins[str(arm/'outcome.json')]=sha(arm/'outcome.json')
                expected={int(x['block']):x['candidate_id'] for x in outcome['selected_blocks']}
                journals=list(arm.glob('decision_*.json'))
                selected=set()
                for path in sorted(journals):
                    block=int(path.name.split('_')[1].split('.')[0]);record=read(path);pins[str(path)]=sha(path)
                    if path.stem.endswith('_rejected'):
                        folder=ORIGINAL/'scenarios'/name/'shared_root_pool' if block==3 else arm/f'pool_{block}'
                    else:
                        folder=arm/f'recovery_pool_{block}'
                        if not folder.exists():folder=ORIGINAL/'scenarios'/name/'shared_root_pool' if block==3 else arm/f'pool_{block}'
                    plans=[];values=[]
                    for cid in range(4):
                        p=folder/f'candidate_{cid}'/'actions.npy';pins[str(p)]=sha(p)
                        plans.append(np.load(p,allow_pickle=False));values.append(float(read(p.parent/'query.json')['value']))
                    visuals=[CandidateVisualEvidence(**v) for v in record['candidate_visual_evidence']]
                    before=belief_from_snapshot(record['belief_before']);attr=attribution_from_snapshot(record['attribution'])
                    effects,kw,chosen,scores,fallback=build_context(before,attr,plans,values,visuals,
                        bundle.FROZEN_RELATIONS[(task,0)].relation,block,backbone,residual)
                    cid=None if chosen is None else chosen.candidate_id
                    if cid!=record['selected_candidate_id'] or fallback!=record['fallback']:
                        raise ValueError(f'Complete V8 exact decision replay mismatch: {name}/{path.name}')
                    old_scores={int(k):float(v) for k,v in record['scores'].items()}
                    if scores.keys()!=old_scores.keys() or any(abs(scores[k]-old_scores[k])>1e-10 for k in scores):
                        raise ValueError(f'Complete V8 score replay mismatch: {name}/{path.name}')
                    original_kw={k:v for k,v in kw.items() if k!='frozen_residual'}
                    old,_=select_evidence_arbitration(**original_kw,residual=residual)
                    if old!=chosen:raise ValueError('Zero conditional wrapper is not original arbitration')
                    if not path.stem.endswith('_rejected'):
                        if expected.get(block)!=cid:raise ValueError('Executed candidate and decision journal disagree')
                        selected.add(block)
                    checks.append(dict(scene=name,block=block,journal=path.name,candidate=cid,fallback=fallback))
                if selected!=set(expected):raise ValueError('Missing recorded executed decision')
    scenes={c['scene'] for c in checks}
    if len(scenes)!=64:raise ValueError('Not all original 64 scenes replayed')
    report=dict(passed=True,scenes=64,decision_journals=len(checks),zero_residual_exact=True,
        original_strategy='complete_v8_trusted_evidence_arbitration',checks=checks,
        outcome_use='identity audit only; no 64-scene outcome used to fit or calibrate')
    dump(out/'complete_v8_zero_replay_audit.json',report);dump(out/'complete_v8_replay_sources.json',pins)
    return report,bundle
