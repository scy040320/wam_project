"""Read-only reuse inventory; never converts a missing event to a negative.

Reports existing atom certificates at inherited plan-use deadlines. Atom
counts are NOT predicate admissions: changed joint is not current pose;
carrying sufficient evidence is not a complementary held classifier.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib,importlib,json,sys,types
from pathlib import Path
import numpy as np

NATIVE='known_task_recovery_v9_20261006_v5_paired_recovery'
SOURCE='known_task_recovery_observation_audit_20261006_v5_motion_typed_support_full'
LABELS=SOURCE+'_labels_v3_perspective'
HIST='known_task_recovery_v9_20261006_v2_label_masks_auxiliary'
TRAINING='known_task_recovery_v9_training_20261006_v2_canonical_precision_inputs'
CODE='known_task_recovery_v9_training_20261006_v2_canonical_precision'
EXPECTED_CODE={'recovery_contract.py':'26c13d935ecc5f77c94695a3af700e845a19d8270e6ba5bd77ce0bacba78dbc3',
 'candidate_effects.py':'55b11811e36269ac08927d53fe66f260a84ba0fed401766ffd25b633dd4e1c38',
 'evidence_residual.py':'51cb2fba7e266435ba6cc5d5c74e0c266e7db1b42c2413ae5d279aef7ec3f5c2',
 'contracts.py':'95ccf099dd42adc600ba477a1948fea248287141a3727439bb68e3f0d8664f75'}
HEADS=('target_visible','target_pose_current','target_reachable','grasped','lifted','receptacle_visible','place_ready')
ATOMS=('left_finger_target_contact','right_finger_target_contact','target_support_contact','carried_sufficient_evidence','lifted_sufficient_evidence','released_sufficient_evidence','joint_state_changed')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()

def table(path):
 return dict((line.split('  ',1)[1],line.split('  ',1)[0]) for line in (path/'SHA256SUMS.txt').read_text().splitlines())

def read(path,refs,*,base=None,manifest=None):
 p=Path(path);digest=sha(p)
 if manifest is not None and manifest[str(p.relative_to(base))]!=digest:raise ValueError('SHA mismatch: '+str(p))
 refs[str(p)]=digest
 return json.loads(p.read_text())

def atom_status(atom):
 if atom.get('new_mask') is not True:return 'masked'
 if atom.get('physical_value') is True:return 'positive'
 if atom.get('physical_value') is False:return 'negative'
 return 'masked'

def endpoint_status(label,endpoint):
 if label.get(endpoint+'_mask') is not True:return 'masked'
 value=label.get(endpoint)
 if value=='true':return 'positive'
 if value=='false':return 'negative'
 return 'masked'

def run(root):
 root=Path(root).resolve();refs={};code=root/'research_runs'/CODE/'overlay/wam_reranking'
 for name,digest in EXPECTED_CODE.items():
  if sha(code/name)!=digest:raise ValueError('Frozen use-time code SHA mismatch: '+name)
 package=types.ModuleType('wam_reranking');package.__path__=[str(code)]
 sys.modules['wam_reranking']=package
 from wam_reranking.contracts import CandidateVisualEvidence
 from wam_reranking.candidate_effects import parse_candidate_effect
 from wam_reranking.evidence_residual import typed_gate_effect
 from wam_reranking.recovery_contract import command_requirements
 src=root/'outputs'/SOURCE;lab=root/'outputs'/LABELS;native=root/'outputs'/NATIVE
 sm,lm=table(src),table(lab)
 records=read(src/'records.json',refs,base=src,manifest=sm)
 original_hashes=read(src/'source_sha256.json',refs,base=src,manifest=sm)
 native_pools=read(native/'pools.json',refs)
 frozen_audit=read(lab/'observability_label_audit.json',refs,base=lab,manifest=lm)
 pools={x['pool_id']:x for x in native_pools}
 if len(records)!=160 or len(pools)!=40:raise ValueError('Frozen native cardinality differs')
 use_rows=[];fields=Counter();stages=Counter();atom_usage=defaultdict(Counter);step_counts=defaultdict(Counter)
 step_atom_counts=defaultdict(Counter);pool_events=defaultdict(list);contact_inventory=Counter();complete_names=True
 for r in records:
  pool,cid,split=r['pool_id'],r['candidate_id'],r['split'];p=pools[pool]
  d=src/r['directory'];teacher=read(d/'temporal_teacher.json',refs,base=src,manifest=sm)
  cert=read(lab/'certificates'/pool/f'candidate_{cid}'/'observability_certificates.json',refs,base=lab,manifest=lm)
  identity=teacher['identity']
  if identity!=cert['identity'] or teacher['split']!=split or cert['split']!=split or identity['candidate_id']!=cid:raise ValueError('Identity/split mismatch')
  original=root/teacher['source_folder'];plan=original/'planned_actions.npy';refs[str(plan)]=sha(plan)
  declared=next(x for x in p['candidates'] if x['candidate_id']==cid)
  # The native membership sha_array hashes decoded arrays, not .npy bytes.
  # Verify the exact original file by its independently frozen source table.
  if refs[str(plan)]!=original_hashes[str(plan.relative_to(root))]:raise ValueError('Original planned-actions file SHA changed')
  a=np.load(plan,allow_pickle=False);visual=read(original/'candidate_visual.json',refs)
  effect=typed_gate_effect(parse_candidate_effect(cid,a,visual_evidence=CandidateVisualEvidence(**visual),close_when_negative=False),p['relation'])
  uses,_=command_requirements(effect,a);stages[effect.stage.value]+=1
  for f in teacher['frames']:
   ev=f['observation_evidence'];mapping=ev['private_collision_vs_visual_mapping']
   fields['native_frames']+=1
   fields['all_scene_contact_geom_name_pairs']+=int(isinstance(f.get('contact_geom_pairs'),list))
   fields['complete_hand_visual_binding']+=int(any(k in mapping for k in ('hand','full_hand','palm','full_gripper')))
   fields['attachment_or_equality_exclusion']+=int(any(k in f.get('physics',{}) for k in ('target_attached','target_hand_attachment','target_welded','active_attachment_constraints')))
   target={x['geom_name'] for x in mapping['target']}
   pair_names=f.get('contact_geom_pairs',[])
   if any(len(pair)!=2 or any(not isinstance(x,str) for x in pair) for pair in pair_names):complete_names=False
   else:
    touching=[(x,y) for x,y in pair_names if (x in target and y.startswith('gripper')) or (y in target and x.startswith('gripper'))]
    contact_inventory['target_any_named_gripper_contact_positive' if touching else 'no_target_named_gripper_contact_in_saved_all_scene_pairs']+=1
   if f['step']>0:
    for name in ATOMS:
     status=atom_status(cert['labels'][f['step']]['atoms'][name]);step_atom_counts[name][split+'.'+status]+=1
  for u in uses:
   step_counts[u.predicate][str(u.at_step)]+=1
   # Proposed correspondence is diagnostic ONLY, never an admitted new label.
   mapped={'grasped':'carried_sufficient_evidence','lifted':'lifted_sufficient_evidence'}.get(u.predicate)
   status=atom_status(cert['labels'][u.at_step]['atoms'][mapped]) if mapped else 'no_semantic_mapping'
   atom_usage[u.predicate][split+'.'+status]+=1
   q=dict(pool_id=pool,candidate_id=cid,split=split,predicate=u.predicate,use_step=u.at_step,stage=effect.stage.value,
       source_identity=identity,diagnostic_atom=mapped,atom_status=status,admitted=False)
   use_rows.append(q);pool_events[(pool,split,u.predicate,u.at_step)].append(q)
 # Same-use contrasts only count explicit certified opposite values.
 contrasts=defaultdict(Counter)
 for (pool,split,pred,step),rows in pool_events.items():
  states={x['atom_status'] for x in rows}
  if 'positive' in states and 'negative' in states:contrasts[pred][split]+=1
 aux=root/'outputs'/HIST
 hist_report=read(aux/'completion_audit.json',refs)
 hist_rows=read(aux/'auxiliary_rows.json',refs)['rows']
 if len(hist_rows)!=797:raise ValueError('Historical 797 changed')
 endpoint_counts=defaultdict(Counter);historical_fields=Counter();hist_use_counts=defaultdict(Counter);history_sources=[]
 for row in hist_rows:
  recdir=aux/row['sample_id'];rec=read(recdir/'record.json',refs);label=read(recdir/'labels.json',refs)
  if rec['identity']!=row['identity'] or rec['split']!=row['split']:raise ValueError('Historical identity mismatch')
  split=row['split'];identity=row['identity'];bd=root/identity['observed_block_id'];idx=int(bd.name.rsplit('_',1)[1])
  decision=read(bd.parent/f'decision_{idx}.json',refs);plan=bd/'evidence/block/planned_actions.npy';refs[str(plan)]=sha(plan)
  actions=np.load(plan,allow_pickle=False);task=identity['task'];relation='articulated' if task==0 else ('on' if task==9 else 'inside')
  visual=decision['candidate_visual_evidence'][identity['candidate_id']]
  effect=typed_gate_effect(parse_candidate_effect(identity['candidate_id'],actions,visual_evidence=CandidateVisualEvidence(**visual),close_when_negative=False),relation)
  uses,_=command_requirements(effect,actions)
  for u in uses:hist_use_counts[u.predicate][str(u.at_step)]+=1
  for group in ('predicates','observability'):
   for name,item in label[group].items():
    for endpoint in ('before','after'):
     endpoint_counts[name+'.'+endpoint][split+'.'+endpoint_status(item,endpoint)]+=1
  keys=rec['arrays'];historical_fields['complete_blocks']+=1
  historical_fields['dual_rgb_17frames']+=int(rec['step_indices']==list(range(17)) and 'primary_sequence' in keys and 'wrist_sequence' in keys)
  historical_fields['requested_applied_16steps']+=int('requested_actions' in keys and 'applied_actions' in keys)
  historical_fields['endpoint_proprio_only']+=int('proprio_start' in keys and 'proprio_end' in keys and 'proprio_sequence' not in keys)
  historical_fields['per_step_physics_or_contacts']+=int(any('contact' in k or 'physics' in k for k in keys))
  historical_fields['full_hand_mapping_or_repeat_noise']+=int(any('hand' in k or 'repeat' in k for k in keys))
  history_sources.append(dict(identity=identity,split=split,record_sha256=refs[str(recdir/'record.json')],label_sha256=refs[str(recdir/'labels.json')]))
 # Boundary16 is a distinct future recovery-goal audit, not a moved use time.
 boundary16=defaultdict(Counter);by_deadline=defaultdict(Counter);mixed=defaultdict(Counter)
 for r in records:
  cert=read(lab/'certificates'/r['pool_id']/f'candidate_{r["candidate_id"]}'/'observability_certificates.json',refs,base=lab,manifest=lm)
  for atom in ATOMS:boundary16[atom][r['split']+'.'+atom_status(cert['labels'][16]['atoms'][atom])]+=1
 for atom in ATOMS:
  for p in pools.values():
   values=[]
   for cid in range(4):
    c=read(lab/'certificates'/p['pool_id']/f'candidate_{cid}'/'observability_certificates.json',refs,base=lab,manifest=lm)
    values.append(atom_status(c['labels'][16]['atoms'][atom]))
   if 'positive' in values and 'negative' in values:mixed[atom][p['split']]+=1
 return dict(schema='v10_existing_supervision_reuse_audit_v1',source_native=NATIVE,temporal_source=SOURCE,labels_source=LABELS,
  native_candidate_count=160,native_pool_count=40,historical_complete_blocks=797,historical_split=hist_report['split'],
  inherited_requirement_stage_counts=dict(stages),native_plan_use_deadline_counts={k:dict(v) for k,v in step_counts.items()},
  historical_plan_use_deadline_counts={k:dict(v) for k,v in hist_use_counts.items()},
  native_atom_status_at_inherited_use={k:dict(v) for k,v in atom_usage.items()},
  native_explicit_atom_opposition_same_pool_same_use={k:dict(v) for k,v in contrasts.items()},
  native_frame_field_inventory=dict(fields),saved_contact_name_inventory=dict(contact_inventory),all_contact_pair_names_complete=complete_names,
  historical_field_inventory=dict(historical_fields),historical_endpoint_existing_silver={k:dict(v) for k,v in endpoint_counts.items()},
  all_step1_to16_certified_atom_counts={k:dict(v) for k,v in step_atom_counts.items()},
  boundary16_atom_counts={k:dict(v) for k,v in boundary16.items()},boundary16_opposition_pools={k:dict(v) for k,v in mixed.items()},
  original_event_deadline_report=frozen_audit['event_by_deadline_coverage'],
  source_code_sha256=EXPECTED_CODE,source_file_reference_count=len(refs),source_file_reference_table_sha256=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest(),
  use_time_table_sha256=hashlib.sha256(json.dumps(use_rows,sort_keys=True).encode()).hexdigest(),
  historical_identity_table_sha256=hashlib.sha256(json.dumps(history_sources,sort_keys=True).encode()).hexdigest(),
  explicitly_not_admitted=['joint_changed_is_not_target_pose_current','finger_contact_is_not_held','missing_carry_is_not_held_false',
    'released_is_not_placed','2D_relation_is_not_physical_support','endpoint16_is_not_earlier_use_time','hand_union_is_not_complete_hand'],
  source_fields_only_not_new_teacher=True,new_labels_generated=False,old_masks_unchanged=True,training_started=False,new_queries=0,new_replays=0,
  training_ready=False,admitted_heads=[])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 print(json.dumps(run(p.parse_args().root),sort_keys=True))
