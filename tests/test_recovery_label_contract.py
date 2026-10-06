import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wam_reranking.recovery_label_contract import (
    SCHEMA, INHERITS, TARGETS, OBSERVABILITY_TARGETS, masked_label,
    audit_recovery_labels, audit_recovery_membership,
)


def bundle(folder):
    identity = dict(dataset='frozen_v8', suite='libero90', task=46, state=10,
                    candidate_id=0, observed_block_id='block_3')
    arrays = dict(primary_sequence='primary.npy', wrist_sequence='wrist.npy',
                  requested_actions='requested.npy', applied_actions='applied.npy',
                  proprio_start='start.npy', proprio_end='end.npy')
    for name, file in arrays.items():
        shape = (17, 2, 2, 3) if 'sequence' in name else (16, 7) if 'actions' in name else (8,)
        np.save(folder/file, np.zeros(shape, dtype=np.float32))
    record = dict(schema=SCHEMA, inherits=INHERITS, identity=identity,
        role='historical_selected_arm_auxiliary', split='train', executed_length=16,
        step_indices=list(range(17)), arrays=arrays, labels_path='labels.json')
    visible = dict(before='unknown', after='true', before_mask=False, after_mask=True,
        transition_mask=False, before_quality=0., after_quality=.8,
        source='observed_image_support', evidence_steps=[16], evidence_ids=['wrist_sequence'])
    labels = dict(schema=SCHEMA, inherits=INHERITS, role='supervision_only',
        identity=identity, predicates={name:masked_label() for name in TARGETS},
        observability={name:masked_label() for name in OBSERVABILITY_TARGETS},
        relation_support_2d=dict(before=None, after=None, before_mask=False, after_mask=False,
            transition_mask=False, before_quality=0., after_quality=0.,
            source='observed_image_support', evidence_steps=[], evidence_ids=[], units='image_space_proxy'))
    labels['predicates']['target_visible'] = visible
    return record, labels


class LabelContractTests(unittest.TestCase):
    def run_audit(self, change=None):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name); record,labels=bundle(folder)
            if change:change(record,labels,folder)
            (folder/'labels.json').write_text(json.dumps(labels),encoding='utf-8')
            return audit_recovery_labels(record,directory=folder)

    def test_unknown_before_observed_after_is_endpoint_not_transition(self):
        result=self.run_audit()
        self.assertTrue(result['ready'],result['problems'])
        self.assertEqual(result['supervision_masks']['target_visible'],
            dict(before_mask=False,after_mask=True,transition_mask=False))
        self.assertFalse(result['physical_recovery_certified'])

    def test_intermediate_proprio_not_required(self):
        result=self.run_audit()
        self.assertTrue(result['ready']);self.assertFalse(result['intermediate_proprio_present'])
        self.assertTrue(result['endpoint_proprio_present'])

    def test_no_proprio_still_supports_images(self):
        def change(record,labels,folder):
            del record['arrays']['proprio_start'];del record['arrays']['proprio_end']
        self.assertTrue(self.run_audit(change)['ready'])

    def test_hidden_before_cannot_certify_transition(self):
        def change(record,labels,folder):labels['predicates']['target_visible']['transition_mask']=True
        result=self.run_audit(change)
        self.assertFalse(result['ready'])
        self.assertIn('target_visible.transition_requires_both_observed_endpoints',result['problems'])

    def test_unknown_after_cannot_be_unmasked(self):
        def change(record,labels,folder):labels['predicates']['target_visible']['after']='unknown'
        self.assertFalse(self.run_audit(change)['ready'])

    def test_quality_gate_not_lowered(self):
        def change(record,labels,folder):labels['predicates']['target_visible']['after_quality']=.49
        self.assertFalse(self.run_audit(change)['ready'])

    def test_availability_transition_is_not_physical_recovery(self):
        def change(record,labels,folder):
            labels['observability']['target_evidence_available']=dict(before='false',after='true',
                before_mask=True,after_mask=True,transition_mask=True,before_quality=.8,after_quality=.8,
                source='observed_image_support',evidence_steps=[0,16],
                evidence_ids=['primary_sequence','wrist_sequence'])
        result=self.run_audit(change);self.assertTrue(result['ready'],result['problems'])
        self.assertTrue(result['supervision_masks']['target_evidence_available']['transition_mask'])
        self.assertFalse(result['physical_recovery_certified'])

    def test_physical_proxy_default_masked(self):
        result=self.run_audit()
        for name in ('grasped','lifted','placed','execution_consistent','target_pose_current'):
            self.assertFalse(any(result['supervision_masks'][name].values()))

    def test_command_does_not_prove_grasp(self):
        def change(record,labels,folder):
            labels['predicates']['grasped']=dict(before='unknown',after='true',before_mask=False,
                after_mask=True,transition_mask=False,before_quality=0.,after_quality=1.,
                source='measured_command_difference',evidence_steps=[16],
                evidence_ids=['requested_actions','applied_actions'])
        result=self.run_audit(change);self.assertFalse(result['ready'])
        self.assertIn('grasped.command_does_not_certify_contact',result['problems'])

    def test_recorded_command_channel_is_separate(self):
        def change(record,labels,folder):
            labels['predicates']['command_consistent']=dict(before='unknown',after='true',
                before_mask=False,after_mask=True,transition_mask=False,before_quality=0.,after_quality=1.,
                source='measured_command_difference',evidence_steps=[16],
                evidence_ids=['requested_actions','applied_actions'])
        self.assertTrue(self.run_audit(change)['ready'])

    def test_terminal_success_and_predicted_attribution_not_label_sources(self):
        for source in ('final_success','predicted_attribution','intervention_label','simulator_contact'):
            def change(record,labels,folder):labels['predicates']['target_visible']['source']=source
            with self.subTest(source=source):self.assertFalse(self.run_audit(change)['ready'])

    def test_relation_proxy_is_separate_and_not_contact(self):
        def change(record,labels,folder):
            labels['relation_support_2d']=dict(before=.3,after=.7,before_mask=True,after_mask=True,
                transition_mask=True,before_quality=.8,after_quality=.8,source='observed_image_support',
                evidence_steps=[0,16],evidence_ids=['primary_sequence','wrist_sequence'],units='image_space_proxy')
        result=self.run_audit(change);self.assertTrue(result['ready'],result['problems'])
        self.assertFalse(result['physical_recovery_certified'])

    def test_relation_proxy_cannot_claim_physical_contact(self):
        def change(record,labels,folder):labels['relation_support_2d']['units']='physical_contact'
        self.assertFalse(self.run_audit(change)['ready'])

    def test_label_identity_must_match_observed_block(self):
        def change(record,labels,folder):
            labels['identity']=copy.deepcopy(labels['identity']);labels['identity']['observed_block_id']='other'
        self.assertFalse(self.run_audit(change)['ready'])

    def test_actions_and_images_are_mandatory(self):
        for field in ('primary_sequence','wrist_sequence','requested_actions','applied_actions'):
            def change(record,labels,folder):del record['arrays'][field]
            with self.subTest(field=field):self.assertFalse(self.run_audit(change)['ready'])

    def test_short_blocks_retained_not_formal(self):
        def change(record,labels,folder):record['executed_length']=8
        result=self.run_audit(change);self.assertFalse(result['formal_eligible'])
        self.assertIn('short_block_retained_excluded_formal',result['problems'])

    def test_before_and_after_witnesses_are_separate(self):
        def change(record,labels,folder):labels['predicates']['target_visible']['evidence_steps']=[0]
        self.assertFalse(self.run_audit(change)['ready'])

    def test_reference_cannot_escape_candidate_directory(self):
        def change(record,labels,folder):record['arrays']['wrist_sequence']='../outside.npy'
        self.assertFalse(self.run_audit(change)['ready'])

    def test_empty_image_array_is_not_actual_evidence(self):
        def change(record,labels,folder):np.save(folder/'primary.npy',np.zeros((17,0,0,3)))
        self.assertFalse(self.run_audit(change)['ready'])

    def test_no_legacy_768_completeness_requirement(self):
        record=dict(identity=dict(dataset='aux',suite='libero90',task=46,state=10,
            candidate_id=0,observed_block_id='block_0'),split='train',role='historical_selected_arm_auxiliary')
        result=audit_recovery_membership([record]);self.assertTrue(result['ready']);self.assertEqual(result['records'],1)

    def test_group_leakage_cannot_hide_behind_dataset_name(self):
        identity=dict(dataset='aux',suite='libero90',task=46,state=10,candidate_id=0,observed_block_id='block_0')
        a=dict(identity=identity,split='train',role='historical_selected_arm_auxiliary')
        b=copy.deepcopy(a);b['identity']['dataset']='new';b['split']='val'
        self.assertTrue(audit_recovery_membership([a,b])['group_leakage'])

    def test_source_scope_avoids_bare_id_collision(self):
        identity=dict(dataset='aux',suite='libero90',task=46,state=10,candidate_id=0,observed_block_id='block_0')
        a=dict(identity=identity,split='train',role='historical_selected_arm_auxiliary')
        b=copy.deepcopy(a);b['identity']['dataset']='new'
        self.assertTrue(audit_recovery_membership([a,b])['ready'])

    def test_sealed32_and_reserved128_cannot_enter_fit(self):
        for task,state,extra in ((0,35,()),(46,22,(('libero90',46,22),))):
            record=dict(identity=dict(dataset='aux',suite='libero90',task=task,state=state,
                candidate_id=0,observed_block_id='block_0'),split='train',role='historical_selected_arm_auxiliary')
            self.assertFalse(audit_recovery_membership([record],excluded_groups=extra)['ready'])

    def test_suite_alias_cannot_hide_a_reserved_group(self):
        record=dict(identity=dict(dataset='aux',suite='libero_90',task=0,state=35,
            candidate_id=0,observed_block_id='block_0'),split='train',role='historical_selected_arm_auxiliary')
        self.assertFalse(audit_recovery_membership([record])['ready'])

    def test_confirmation_role_and_split_cannot_be_promoted_implicitly(self):
        record=dict(identity=dict(dataset='aux',suite='libero90',task=46,state=10,
            candidate_id=0,observed_block_id='block_0'),split='confirmation',role='confirmation')
        self.assertFalse(audit_recovery_membership([record])['ready'])


if __name__=='__main__':unittest.main()
