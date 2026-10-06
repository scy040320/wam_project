import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_recovery_derived import (
    audit_derived, write_new_audit, streaming_sha256, target_vectors, FROZEN_FEATURE_SCHEMA,
)
from wam_reranking.observed_recovery import labels_from_actual
from wam_reranking.recovery_contract import FEATURE_NAMES


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def visible_map():
    arr=np.zeros((8,8),dtype=np.float32);arr[2:4,2:4]=1.;return arr


def synthetic(root):
    source=root/'aux';source.mkdir()
    teacher=root/'teacher';teacher.mkdir()
    (teacher/'model.safetensors').write_bytes(b'synthetic_teacher_bytes_for_integrity_test_only')
    save_json(teacher/'config.json',{'test_fixture':True})
    rows=[];names=None
    for index in range(2):
        folder=source/'records'/f'block_{index}';folder.mkdir(parents=True)
        identity=dict(dataset='historical_aux',suite='libero90',task=46,state=10,
                      candidate_id=0,observed_block_id='source/block_'+str(index))
        maps=np.stack([visible_map() for _ in range(4)])
        maps[2 if index==0 else 0:4 if index==0 else 2]=0.
        sm=maps.copy();am=np.stack([visible_map() for _ in range(4)])
        requested=np.zeros((16,7),dtype=np.float32);applied=requested.copy()
        arrays=dict(primary_sequence='primary.npy',wrist_sequence='wrist.npy',
                    requested_actions='requested.npy',applied_actions='applied.npy',
                    proprio_start='start.npy',proprio_end='end.npy')
        for name,file in arrays.items():
            arr=(np.zeros((17,2,2,3),dtype=np.uint8) if 'sequence' in name else
                 requested if 'requested' in name else applied if 'applied' in name else np.zeros(8))
            np.save(folder/file,arr)
        np.savez_compressed(folder/'observer_maps.npz',subject=sm,anchor=am)
        labels=labels_from_actual(identity=identity,subject_maps=sm,anchor_maps=am,
                                 requested=requested,applied=applied)
        save_json(folder/'labels.json',labels)
        record=dict(schema='candidate_recovery_labels_v2',inherits='candidate_recovery_supervision_v1',
            identity=identity,role='historical_selected_arm_auxiliary',split='train',executed_length=16,
            step_indices=list(range(17)),arrays=arrays,labels_path='labels.json')
        save_json(folder/'record.json',record)
        names,values,masks=target_vectors(labels)
        rows.append(dict(sample_id=str(folder.relative_to(source)),split='train',identity=identity,
            features=[0.]*604,no_dag_features=[0.]*604,targets=values,masks=masks))
    save_json(source/'protocol.json',dict(schema='recovery_aux_relabelling_v2',
        features=FROZEN_FEATURE_SCHEMA,feature_names=list(FEATURE_NAMES)))
    save_json(source/'auxiliary_rows.json',dict(feature_schema=FROZEN_FEATURE_SCHEMA,
        feature_names=list(FEATURE_NAMES),target_names=names,rows=rows))
    save_json(source/'completion_audit.json',dict(passed=True,blocks=2))
    save_json(source/'status.json',dict(phase='complete',blocks=2))
    return source,teacher


class DerivedRecoveryAuditTests(unittest.TestCase):
    code_root=Path(__file__).resolve().parents[1]

    def audit(self,change=None):
        with tempfile.TemporaryDirectory() as directory:
            source,teacher=synthetic(Path(directory))
            if change:change(source,teacher)
            return audit_derived(source,expected_count=2,teacher_root=teacher,
                                 code_roots=[self.code_root])

    def modify_bundle(self,source,change):
        path=source/'auxiliary_rows.json';bundle=json.loads(path.read_text())
        change(bundle);save_json(path,bundle)

    def test_pilot_complete_and_hashes_present(self):
        report,manifest=self.audit()
        self.assertTrue(report['passed'],report['problems'])
        self.assertEqual(len(manifest['per_record']),2)
        self.assertIn('model.safetensors',manifest['teacher_sha256'])
        self.assertIn('wam_reranking/observed_recovery.py',manifest['source_code_sha256'])
        self.assertFalse(report['physical_recovery_certified'])
        self.assertEqual(report['frozen_v2_feature_role'],'diagnostic_only_not_new_v3_training_inputs')

    def test_signed_transition_both_directions_preserved(self):
        report,_=self.audit();self.assertTrue(report['passed'])
        self.assertGreater(report['signed_transition_counts']['negative'],0)
        self.assertGreater(report['signed_transition_counts']['positive'],0)

    def test_feature604_and_nonfinite_rejected(self):
        for mutation in ('shape','nonfinite'):
            def change(source,teacher):
                def update(bundle):
                    if mutation=='shape':bundle['rows'][0]['features']=[0.]*603
                    else:bundle['rows'][0]['no_dag_features'][0]=float('nan')
                self.modify_bundle(source,update)
            with self.subTest(mutation=mutation):
                report,_=self.audit(change);self.assertFalse(report['passed'])

    def test_masked_targets_must_be_finite_too(self):
        def change(source,teacher):
            def update(bundle):
                index=bundle['target_names'].index('grasped.after')
                bundle['rows'][0]['targets'][index]=float('nan')
            self.modify_bundle(source,update)
        report,_=self.audit(change);self.assertFalse(report['passed'])
        self.assertTrue(any('target_shape_or_nonfinite' in p for p in report['problems']))

    def test_mask_mismatch_and_label_target_mismatch_detected(self):
        for mutation in ('mask','target'):
            def change(source,teacher):
                def update(bundle):
                    if mutation=='mask':bundle['rows'][0]['masks'][0]=not bundle['rows'][0]['masks'][0]
                    else:bundle['rows'][0]['targets'][0]=.1234
                self.modify_bundle(source,update)
            with self.subTest(mutation=mutation):self.assertFalse(self.audit(change)[0]['passed'])

    def test_finalsuccess_cannot_enter_feature_envelope(self):
        def change(source,teacher):
            self.modify_bundle(source,lambda b:b['rows'][0].update(success=True))
        report,_=self.audit(change);self.assertFalse(report['passed'])
        self.assertIn('unexpected_or_gt_feature_envelope',report['problems'])

    def test_independently_regenerated_silver_detects_tampered_labels(self):
        def change(source,teacher):
            path=source/'records/block_0/labels.json';labels=json.loads(path.read_text())
            labels['observability']['target_evidence_available']['after']='true';save_json(path,labels)
        report,_=self.audit(change);self.assertFalse(report['passed'])
        self.assertTrue(any('observer_silver_not_deterministic' in p for p in report['problems']))

    def test_group_leakage_with_matching_row_split_detected(self):
        def change(source,teacher):
            path=source/'records/block_1/record.json';record=json.loads(path.read_text())
            record['split']='val';save_json(path,record)
            self.modify_bundle(source,lambda b:b['rows'][1].update(split='val'))
        report,_=self.audit(change);self.assertFalse(report['passed']);self.assertIn('group_leakage',report['problems'])

    def test_missing_actual_teacher_or_config_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            source,teacher=synthetic(Path(directory));(teacher/'model.safetensors').unlink()
            with self.assertRaises(ValueError):audit_derived(source,expected_count=2,
                teacher_root=teacher,code_roots=[self.code_root])

    def test_streaming_hash_matches_full_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'large.bin';data=b'abcd'*900000;path.write_bytes(data)
            self.assertEqual(streaming_sha256(path,chunk_bytes=4096),hashlib.sha256(data).hexdigest())

    def test_only_new_directory_written_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,teacher=synthetic(root)
            before={str(p.relative_to(source)):streaming_sha256(p) for p in source.rglob('*') if p.is_file()}
            out=root/'new_audit'
            report=write_new_audit(source,out,expected_count=2,teacher_root=teacher,code_roots=[self.code_root])
            self.assertTrue(report['passed']);self.assertTrue((out/'SHA256SUMS.txt').is_file())
            after={str(p.relative_to(source)):streaming_sha256(p) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            with self.assertRaises(ValueError):write_new_audit(source,out,expected_count=2,
                teacher_root=teacher,code_roots=[self.code_root])
            with self.assertRaises(ValueError):write_new_audit(source,source/'unsafe',expected_count=2,
                teacher_root=teacher,code_roots=[self.code_root])

    def test_running_or_incomplete_source_not_certified(self):
        def change(source,teacher):save_json(source/'status.json',dict(phase='relabeling',completed=1))
        report,_=self.audit(change);self.assertFalse(report['passed'])
        self.assertIn('relabeling_not_complete_or_failed',report['problems'])


if __name__=='__main__':unittest.main()
