"""Pure CPU tests; no unpickling, simulator, network, query or training."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from wam_reranking import temporal_recovery_teacher as teacher

spec=importlib.util.spec_from_file_location('_tail_replay_cpu',Path(__file__).resolve().parents[1]/'scripts/replay_archived_release_tail.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def contract():
    entries=[]
    for (task,state,context,split),f in m.FIXED.items():
        entries.append(dict(task=task,state=state,context=context,split=split,suite='libero90',
            observed_block=7,valid_length=f['length'],candidate_id=f['cid'],
            trajectory_sha256=f['trajectory'],terminal_metadata_sha256=f['terminal'],
            previous_block_source=dict(observed_block=6,candidate_id=f['previous_cid'],same_actual_execution_chain=True)))
    return dict(schema='archived_terminal_short_auxiliary_scope_v1',source=m.SOURCE,entries=entries)


def history(length=3,release=True):
    frames=[];lineage=[]
    for i in range(65+length):
        tail=i>=65;eef=[i*.003,0.,.5];target=[i*.003,0.,.48]
        if tail and release:target=[64*.003,0.,.48]
        frames.append(dict(step=i,physics=dict(eef_pos_m=eef,target_pos_m=target,
            gripper_aperture_m=.038 if tail and release else .03,
            left_finger_target_contact=not(tail and release),right_finger_target_contact=not(tail and release),
            target_support_contact=False),observability={}))
        lineage.append(dict(observed_block_id='block_7' if tail else 'block_6',candidate_id=1 if tail else 3,
            local_step=i-64 if tail else i,chain_step=i))
    return frames,lineage


class TailTests(unittest.TestCase):
    def test_exact_two_preapproved_short_scopes(self):
        self.assertEqual([x['valid_length'] for x in m.validate_contract(contract())],[3,7])

    def test_extra_scope_or_duplicate_not_allowed(self):
        c=contract();c['entries'].append(copy.deepcopy(c['entries'][0]))
        with self.assertRaises(ValueError):m.validate_contract(c)

    def test_split_and_candidate_not_transplanted_from_previous_block(self):
        for key,value in (('split','val'),('candidate_id',3)):
            c=contract();c['entries'][0][key]=value
            with self.assertRaises(ValueError):m.validate_contract(c)

    def test_hash_or_length_change_rejected(self):
        for key,value in (('trajectory_sha256','unverified'),('valid_length',16)):
            c=contract();c['entries'][0][key]=value
            with self.assertRaises(ValueError):m.validate_contract(c)

    def test_explicit_prior_candidate_link_required(self):
        c=contract();c['entries'][0]['previous_block_source']['same_actual_execution_chain']=False
        with self.assertRaises(ValueError):m.validate_contract(c)

    def test_real_length_arrays_not_padding(self):
        a={v:np.zeros((3,256,256,3),np.uint8) for v in ('primary','wrist')}
        a.update(requested=np.zeros((3,7)),applied=np.zeros((3,7)))
        m.validate_arrays(a,3)
        with self.assertRaises(ValueError):m.validate_arrays(a,16)

    def test_nonfinite_commands_rejected(self):
        a={v:np.zeros((3,256,256,3),np.uint8) for v in ('primary','wrist')}
        a.update(requested=np.zeros((3,7)),applied=np.full((3,7),np.nan))
        with self.assertRaises(ValueError):m.validate_arrays(a,3)

    def test_terminal_exact_length_not_a_release_label(self):
        m.check_terminal(False,1,3);m.check_terminal(True,3,3)
        with self.assertRaises(ValueError):m.check_terminal(True,2,3)
        with self.assertRaises(ValueError):m.check_terminal(False,3,3)

    def test_cross_block_release_uses_actual_past_and_own_candidate(self):
        h,line=history();r=m.derive_tail_windows(teacher,h,line,3)
        self.assertTrue(r['physical_release_exists'])
        event=r['first_release_event'];self.assertEqual(event['current_source']['local_step'],2)
        self.assertEqual(event['current_source']['candidate_id'],1)
        self.assertEqual(event['prior_carried_source']['candidate_id'],3)
        self.assertLess(event['prior_carried_source']['chain_step'],event['current_source']['chain_step'])
        self.assertFalse(event['physics']['released_sufficient_evidence']['supervision_mask'])

    def test_no_contact_loss_no_invented_release_or_negative_grasp(self):
        h,line=history(release=False);r=m.derive_tail_windows(teacher,h,line,3)
        self.assertFalse(r['physical_release_exists']);self.assertIsNone(r['first_release_event'])
        self.assertIsNone(r['frames'][-1]['physics']['holding_after_measured_release']['value'])

    def test_no_future_endpoint_backfills_first_event(self):
        h,line=history();h[-3]['physics']['gripper_aperture_m']=.03
        r=m.derive_tail_windows(teacher,h,line,3)
        self.assertEqual(r['first_release_event']['current_source']['local_step'],2)
        self.assertIsNone(r['frames'][0]['physics']['released_sufficient_evidence']['value'])

    def test_insufficient_real_past_window_rejected(self):
        h,line=history()
        with self.assertRaises(ValueError):m.derive_tail_windows(teacher,h[-17:],line[-17:],3)

    def test_repeat_gate_not_relaxed_and_no_empty_contacts_certificate(self):
        h,_=history();r=dict(frames=h[-4:],qpos=np.zeros((4,3)),qvel=np.zeros((4,3)),terminal_steps=[False,False,True])
        for f in r['frames']:f['physics']['target_anchor_contact']=False
        self.assertTrue(m.repeat_check(r,copy.deepcopy(r))['passed'])
        s=copy.deepcopy(r);s['qpos'][0,0]=.0011
        with self.assertRaises(ValueError):m.repeat_check(r,s)

    def test_frame_identity_preserves_original_short_cid_and_time(self):
        row=dict(task=46,state=10);b=dict(block=7,candidate_id=1,directory='source/block_7')
        identity=m.frame_identity(row,b,3)
        self.assertEqual(identity['chain_step'],67);self.assertEqual(identity['candidate_id'],1)
        self.assertEqual(identity['original_rgb_index'],2)

    def test_numerical_repeat_difference_does_not_change_categorical_events(self):
        h,line=history();a=m.derive_tail_windows(teacher,h,line,3);b=copy.deepcopy(a)
        b['frames'][0]['physics']['target_relative_to_eef_m']['value'][0]+=1e-10
        self.assertEqual(m.event_signature(a),m.event_signature(b))


if __name__=='__main__':unittest.main()
