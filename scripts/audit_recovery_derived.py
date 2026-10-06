"""Read-only derived-recovery/teacher audit; write only a new audit directory.

The labels remain algorithmic observer silver. Hashes establish reproducible
inputs and outputs, not human truth or physical contact certificates. The CLI
must name the actual frozen teacher and source roots; no inferred model version
is substituted. Repeating --code-root lists source overlays in import priority.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from wam_reranking import observed_recovery, recovery_label_contract
from wam_reranking.recovery_label_contract import (
    IDENTITY_FIELDS, TARGETS, OBSERVABILITY_TARGETS, audit_recovery_labels,
    audit_recovery_membership,
)

AUDIT_SCHEMA = 'derived_recovery_integrity_audit_v1'
FROZEN_FEATURE_SCHEMA = 'cause_dependency_recovery_features_v2'
SOURCE_FILES = (
    'wam_reranking/observed_recovery.py', 'wam_reranking/target_localization.py',
    'wam_reranking/candidate_effects.py', 'wam_reranking/recovery_contract.py',
    'wam_reranking/recovery_label_contract.py', 'wam_reranking/contracts.py',
    'wam_reranking/belief.py', 'wam_reranking/source_scoped_policy.py',
    'scripts/relabel_archived_recovery.py',
)
ROW_FIELDS = {'sample_id', 'split', 'identity', 'features', 'no_dag_features', 'targets', 'masks'}
TEACHER_EXTENSIONS = {'.bin', '.safetensors', '.json', '.txt', '.model', '.vocab', '.tiktoken'}


def streaming_sha256(path: Path, chunk_bytes=1024*1024):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(chunk_bytes),b''): digest.update(chunk)
    return digest.hexdigest()


def _dump(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def _inside(path, directory):
    path=path.resolve();directory=directory.resolve()
    return path==directory or directory in path.parents


def _identity_key(identity):
    return tuple(identity[name] for name in IDENTITY_FIELDS)


def target_vectors(labels):
    """Signed transitions are regression targets in [-1,1], not BCE classes."""
    ordered = list(TARGETS)+list(OBSERVABILITY_TARGETS)+['relation_support_2d']
    names=[];values=[];masks=[]
    for name in ordered:
        item=(labels['predicates'][name] if name in TARGETS else
              labels['observability'][name] if name in OBSERVABILITY_TARGETS else labels[name])
        def scalar(value):return float(value) if name=='relation_support_2d' else float(value=='true')
        for endpoint in ('after','transition'):
            names.append(name+'.'+endpoint);mask=item[endpoint+'_mask'];masks.append(mask)
            if not mask:values.append(0.);continue
            after=scalar(item['after'])
            values.append(after if endpoint=='after' else after-scalar(item['before']))
    return names,values,masks


def _teacher_inventory(teacher_root: Path):
    teacher_root=teacher_root.resolve()
    if not teacher_root.is_dir():raise ValueError('Explicit teacher-root is not a directory')
    paths=[p for p in sorted(teacher_root.rglob('*')) if p.is_file()
           and p.suffix in TEACHER_EXTENSIONS and not any(part.startswith('.') for part in p.relative_to(teacher_root).parts)]
    if not any(p.suffix in ('.bin','.safetensors') for p in paths):
        raise ValueError('No actual frozen teacher weight file in explicit teacher-root')
    if not any(p.suffix=='.json' and 'config' in p.name for p in paths):
        raise ValueError('No actual teacher config in explicit teacher-root')
    inventory={}
    for path in paths:
        inventory[str(path.relative_to(teacher_root))]=dict(
            resolved_path=str(path.resolve()),bytes=path.stat().st_size,sha256=streaming_sha256(path))
    return inventory


def _source_inventory(code_roots: Sequence[Path]):
    roots=[Path(p).resolve() for p in code_roots]
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError('Explicit code-root directory is required')
    inventory={}
    for name in SOURCE_FILES:
        relative=Path(name);choices=[]
        for root in roots:
            choices.extend([root/relative,root/'overlay'/relative])
            if relative.parts[0]=='scripts':choices.append(root/relative.name)
        source=next((p for p in choices if p.is_file()),None)
        if source is None:raise ValueError('Required observer/recovery source missing: '+name)
        inventory[name]=dict(path=str(source.resolve()),bytes=source.stat().st_size,
                             sha256=streaming_sha256(source))
    # Imported semantics must be the same bytes as the explicitly audited code.
    # Equal copied snapshots are allowed; silently auditing unrelated code is not.
    for module in (observed_recovery,recovery_label_contract):
        name='wam_reranking/'+Path(module.__file__).name
        if streaming_sha256(Path(module.__file__))!=inventory[name]['sha256']:
            raise ValueError('Imported code does not match explicit frozen source: '+name)
    return inventory


def audit_derived(input_dir: Path, *, expected_count: int, teacher_root: Path,
                  code_roots: Sequence[Path], expected_feature_count=604, excluded_groups=()):
    """Inspect complete pilot (2) or full (797) without modifying its artifacts."""
    if expected_count not in (2,797):raise ValueError('Only explicitly frozen pilot2/full797 are accepted')
    source=input_dir.resolve()
    if not source.is_dir():raise ValueError('Derived input directory missing')
    teacher=_teacher_inventory(teacher_root);code=_source_inventory(code_roots)
    input_hashes={}; problems=[]
    def tracked(path):
        path=Path(path).resolve()
        if not _inside(path,source) or not path.is_file() or not path.stat().st_size:
            raise ValueError('Missing/escaped derived source file: '+str(path))
        key=str(path.relative_to(source));sha=streaming_sha256(path)
        if key in input_hashes and input_hashes[key]!=sha:
            raise ValueError('Derived file changed during audit: '+key)
        input_hashes[key]=sha
        return path
    def read_json(path):return json.loads(tracked(path).read_text(encoding='utf-8'))
    protocol=read_json(source/'protocol.json')
    status=read_json(source/'status.json')
    completion=read_json(source/'completion_audit.json')
    bundle=read_json(source/'auxiliary_rows.json')
    if status.get('phase')!='complete' or not completion.get('passed'):
        problems.append('relabeling_not_complete_or_failed')
    if completion.get('blocks')!=expected_count or status.get('blocks')!=expected_count:
        problems.append('completion_count_mismatch')
    if protocol.get('schema')!='recovery_aux_relabelling_v2':problems.append('protocol_schema')
    if bundle.get('feature_schema')!=FROZEN_FEATURE_SCHEMA or protocol.get('features')!=FROZEN_FEATURE_SCHEMA:
        problems.append('feature_schema')
    frozen_names=protocol.get('feature_names',[])
    if expected_feature_count!=604 or len(frozen_names)!=604 or len(set(frozen_names))!=604:
        problems.append('frozen_604_feature_count')
    if bundle.get('feature_names')!=frozen_names:
        problems.append('feature_name_identity')
    rows=bundle.get('rows',[])
    if not isinstance(rows,list):raise ValueError('Auxiliary row mapping invalid')
    if len(rows)!=expected_count:problems.append('auxiliary_row_count')
    index={};mapped_paths=set()
    for row in rows:
        if not isinstance(row,dict) or set(row)!=ROW_FIELDS:
            problems.append('unexpected_or_gt_feature_envelope');continue
        try:key=_identity_key(row['identity'])
        except (KeyError,TypeError):problems.append('auxiliary_source_identity');continue
        if key in index:problems.append('duplicate_auxiliary_identity')
        index[key]=row
        sample_id=row['sample_id']
        if not isinstance(sample_id,str) or not sample_id:
            problems.append('sample_path');continue
        path=(source/sample_id).resolve()
        if not _inside(path,source):problems.append('sample_path_escape');continue
        if path in mapped_paths:problems.append('duplicate_sample_path')
        mapped_paths.add(path)
    folders=sorted((source/'records').rglob('record.json'))
    if len(folders)!=expected_count:problems.append('record_directory_count')
    records=[];per_record={};masks=Counter();transitions=Counter();mapped=set();regenerated_count=0
    for record_path in folders:
        record=read_json(record_path);folder=record_path.parent
        checked=audit_recovery_labels(record,directory=folder,tracked=tracked)
        if not checked['ready']:
            problems.extend(str(record_path.relative_to(source))+':'+p for p in checked['problems'])
            continue
        records.append(record);identity=record['identity'];key=_identity_key(identity)
        row=index.get(key)
        sample_id=str(folder.relative_to(source))
        files=[record_path,folder/record['labels_path']]+[folder/p for p in record['arrays'].values()]
        maps_path=tracked(folder/'observer_maps.npz');files.append(maps_path)
        labels=read_json(folder/record['labels_path'])
        with np.load(maps_path,allow_pickle=False) as archive:
            if set(archive.files)!={'subject','anchor'}:
                problems.append(sample_id+':observer_map_fields');continue
            sm,am=archive['subject'],archive['anchor']
            if sm.ndim!=3 or sm.shape[0]!=4 or am.shape!=sm.shape or min(sm.shape[1:])<1:
                problems.append(sample_id+':observer_map_shape');continue
            if not np.isfinite(sm).all() or not np.isfinite(am).all():
                problems.append(sample_id+':observer_map_nonfinite');continue
            requested=np.load(folder/record['arrays']['requested_actions'],allow_pickle=False)
            applied=np.load(folder/record['arrays']['applied_actions'],allow_pickle=False)
            regenerated=observed_recovery.labels_from_actual(identity=identity,subject_maps=sm,
                anchor_maps=am,requested=requested,applied=applied)
        if labels!=regenerated:problems.append(sample_id+':observer_silver_not_deterministic')
        else:regenerated_count+=1
        names,expected_values,expected_masks=target_vectors(labels)
        if bundle.get('target_names')!=names:problems.append('target_name_identity')
        if row is None:
            problems.append(sample_id+':missing_auxiliary_row');continue
        mapped.add(key)
        if row['sample_id']!=sample_id or row['split']!=record['split']:
            problems.append(sample_id+':row_record_membership_mismatch')
        for feature in ('features','no_dag_features'):
            try:array=np.asarray(row[feature],dtype=np.float64)
            except (TypeError,ValueError):problems.append(sample_id+':'+feature+'.nonnumeric');continue
            if array.shape!=(expected_feature_count,) or not np.isfinite(array).all():
                problems.append(sample_id+':'+feature+'.shape_or_nonfinite')
        try:values=np.asarray(row['targets'],dtype=np.float64)
        except (TypeError,ValueError):values=np.asarray([])
        row_masks=row['masks']
        if values.shape!=(len(names),) or not np.isfinite(values).all():
            problems.append(sample_id+':target_shape_or_nonfinite')
        elif not np.array_equal(values,np.asarray(expected_values,dtype=np.float64)):
            problems.append(sample_id+':target_value_label_mismatch')
        if not isinstance(row_masks,list) or any(type(m) is not bool for m in row_masks):
            problems.append(sample_id+':mask_type')
        elif row_masks!=expected_masks:problems.append(sample_id+':mask_label_mismatch')
        if values.shape==(len(names),):
            for name,value,masked in zip(names,values,expected_masks):
                if name.endswith('.transition'):
                    if not -1<=value<=1:problems.append(sample_id+':signed_transition_range')
                    if masked:transitions['negative' if value<0 else 'positive' if value>0 else 'zero']+=1
                elif not 0<=value<=1:problems.append(sample_id+':endpoint_probability_range')
                if masked:masks[name]+=1
        manifest={str(p.resolve().relative_to(source)):streaming_sha256(p) for p in files}
        for p,sha in manifest.items():
            if p in input_hashes and input_hashes[p]!=sha:raise ValueError('Derived source changed: '+p)
            input_hashes[p]=sha
        per_record[sample_id]=dict(identity=identity,split=record['split'],role=record['role'],
            derived_sha256=manifest,physical_recovery_certified=False)
    if set(index)!=mapped:problems.append('unmatched_or_unverified_auxiliary_rows')
    membership=audit_recovery_membership(records,excluded_groups=excluded_groups)
    if not membership['ready']:problems.extend(membership['problems'])
    if len(records)!=expected_count:problems.append('contract_valid_record_count')
    # End-of-audit consistency is checked for every derived input, including
    # metadata; a still-running writer cannot slip through a completed audit.
    for path,sha in input_hashes.items():
        if streaming_sha256(source/path)!=sha:raise ValueError('Derived source mutated during audit: '+path)
    for name,item in teacher.items():
        if streaming_sha256(Path(item['resolved_path']))!=item['sha256']:
            raise ValueError('Frozen teacher changed during audit: '+name)
    for name,item in code.items():
        if streaming_sha256(Path(item['path']))!=item['sha256']:
            raise ValueError('Frozen source code changed during audit: '+name)
    report=dict(schema=AUDIT_SCHEMA,passed=not problems,problems=problems,input_dir=str(source),
        expected_count=expected_count,records=len(records),feature_count=expected_feature_count,
        membership=membership,split=dict(Counter(r['split'] for r in records)),
        observable_supervision_counts=dict(masks),signed_transition_counts=dict(transitions),
        observer_labels_regenerated=regenerated_count==expected_count,
        observer_regenerated_records=regenerated_count,teacher_inference_replayed=False,
        observer_is_algorithmic_silver_not_human_truth=True,
        frozen_feature_schema=FROZEN_FEATURE_SCHEMA,
        frozen_v2_feature_role='diagnostic_only_not_new_v3_training_inputs',
        reusable_outputs='actual_arrays_observer_maps_and_masked_labels',
        physical_recovery_certified=False,
        masked_targets_finite=not any('target_shape_or_nonfinite' in p for p in problems),
        original_input_mutation=False,training_started=False,
        teacher_root=str(teacher_root.resolve()),code_roots=[str(Path(p).resolve()) for p in code_roots])
    manifest=dict(schema=AUDIT_SCHEMA,input_dir=str(source),input_sha256=input_hashes,
        per_record=per_record,teacher_sha256=teacher,source_code_sha256=code,
        audit_tool_sha256=streaming_sha256(Path(__file__).resolve()))
    return report,manifest


def write_new_audit(input_dir: Path, output_dir: Path, **kwargs):
    source=input_dir.resolve();out=output_dir.resolve()
    if out.exists() or _inside(out,source) or _inside(out,Path(kwargs['teacher_root'])):
        raise ValueError('Audit must use a NEW directory outside all derived input and teacher files')
    report,manifest=audit_derived(source,**kwargs)
    out.mkdir(parents=True,exist_ok=False)
    _dump(out/'derived_audit.json',report);_dump(out/'derived_sha256_manifest.json',manifest)
    files=('derived_audit.json','derived_sha256_manifest.json')
    (out/'SHA256SUMS.txt').write_text(''.join(streaming_sha256(out/name)+'  '+name+'\n' for name in files),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-count',type=int,choices=(2,797),required=True)
    parser.add_argument('--teacher-root',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,action='append',required=True)
    parser.add_argument('--excluded-groups-json',type=Path)
    args=parser.parse_args()
    excluded=json.loads(args.excluded_groups_json.read_text(encoding='utf-8')) if args.excluded_groups_json else ()
    report=write_new_audit(args.input_dir,args.output_dir,expected_count=args.expected_count,
        teacher_root=args.teacher_root,code_roots=args.code_root,excluded_groups=excluded)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(2)


if __name__=='__main__':main()
