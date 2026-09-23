#!/usr/bin/env python3
import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

root = Path('outputs/paired_native16_recovery_v1')
manifest = list(csv.DictReader(open(root/'manifest.csv', encoding='utf-8')))
rows=[]
expected={
 'normal': {'continue_next_chunk','continue'},
 'visual_occlusion': {'pause_and_reobserve','reobserve'},
 'object_shift': {'local_state_update','invalidate_dependency_closure'},
 'action_noise': {'local_action_correction','correct_next_action'},
 'unknown': {'safe_stop_global_refresh','safe_stop'},
}
for m in manifest:
 p=root/'runs'/f"{m['cell_id']}_{m['approach']}"/'online_decision_log.json'
 d=json.load(open(p,encoding='utf-8'))
 z=next((x for x in d['decisions'] if x.get('intervention_injected')), {})
 rows.append({**m,'success':int(bool(d['success'])),'safety_stopped':int(bool(d['safety_stopped'])),
  'wam_calls':d['wam_calls'],'global_refreshes':d['global_refreshes'],'invalidated_node_total':d['invalidated_node_total'],
  'environment_steps':d['environment_steps'],'recovery_action':z.get('recovery_action',''),'cause':z.get('cause',''),
  'recovery_correct':int(z.get('recovery_action','') in expected[m['condition']])})

primary=[r for r in rows if r['analysis_set']=='primary_heldout']
by=defaultdict(list)
for r in primary: by[r['approach']].append(r)
overall={}
for method,rs in sorted(by.items()):
 overall[method]={
  'n':len(rs),'success_rate':sum(r['success'] for r in rs)/len(rs),
  'safety_stop_rate':sum(r['safety_stopped'] for r in rs)/len(rs),
  'recovery_correct_rate':sum(r['recovery_correct'] for r in rs)/len(rs),
  'mean_global_refreshes':sum(r['global_refreshes'] for r in rs)/len(rs),
  'mean_invalidated_nodes':sum(r['invalidated_node_total'] for r in rs)/len(rs),
  'mean_wam_calls':sum(r['wam_calls'] for r in rs)/len(rs),
 }

condition_method=defaultdict(list)
for r in primary: condition_method[(r['condition'],r['approach'])].append(r)
detail=[]
for (cond,method),rs in sorted(condition_method.items()):
 detail.append({'condition':cond,'approach':method,'n':len(rs),
  'success_rate':sum(r['success'] for r in rs)/len(rs),
  'safety_stop_rate':sum(r['safety_stopped'] for r in rs)/len(rs),
  'recovery_correct_rate':sum(r['recovery_correct'] for r in rs)/len(rs),
  'mean_global_refreshes':sum(r['global_refreshes'] for r in rs)/len(rs),
  'mean_invalidated_nodes':sum(r['invalidated_node_total'] for r in rs)/len(rs),
  'recovery_actions':dict(Counter(r['recovery_action'] for r in rs)),
  'causes':dict(Counter(r['cause'] for r in rs))})

# This is an engineering/protocol pass, not a new threshold-selection gate.
d={(x['condition'],x['approach']):x for x in detail}
criteria={
 'integrity_320_of_320': len(rows)==320,
 'primary_288': len(primary)==288,
 'd_clean_success_100pct': d['normal','D_dependency_aware']['success_rate']==1.0,
 'd_known_nonshift_success_100pct': all(d[c,'D_dependency_aware']['success_rate']==1.0 for c in ['visual_occlusion','action_noise']),
 'd_unknown_safe_stop_100pct': d['unknown','D_dependency_aware']['safety_stop_rate']==1.0,
 'd_reduces_invalidations_vs_b_every_condition': all(d[c,'D_dependency_aware']['mean_invalidated_nodes'] < d[c,'B_uniform_subgraph']['mean_invalidated_nodes'] for c in expected),
 'd_object_shift_task_success': d['object_shift','D_dependency_aware']['success_rate']>0.0,
}
report={'primary_only':True,'overall':overall,'by_condition_method':detail,'criteria':criteria,
 'decision':'PASS_WITH_LIMITATION' if all(v for k,v in criteria.items() if k!='d_object_shift_task_success') else 'FAIL',
 'limitation':'All methods, including D, have 0 task success under object_shift. Do not retune on held-out data; carry this as a preregistered recovery-cue validation target.'}
(root/'comparison_acceptance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
lines=['# Paired recovery comparison acceptance report','','**Decision: '+report['decision']+'**','',report['limitation'],'','## Primary aggregate','',
'| method | n | success | recovery correct | safety stop | global refresh | invalidated nodes | WAM calls |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for m,x in overall.items(): lines.append(f"| {m} | {x['n']} | {x['success_rate']:.3f} | {x['recovery_correct_rate']:.3f} | {x['safety_stop_rate']:.3f} | {x['mean_global_refreshes']:.2f} | {x['mean_invalidated_nodes']:.2f} | {x['mean_wam_calls']:.2f} |")
lines += ['','## Acceptance criteria','']+[f"- [{'x' if v else ' '}] {k}" for k,v in criteria.items()]
(root/'COMPARISON_ACCEPTANCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
