"""CPU synthetic invariants; not measured event coverage or model readiness."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from wam_reranking import recovery_observability_contract as c

SPEC = importlib.util.spec_from_file_location("_observability_builder_v2_tests",
    Path(__file__).resolve().parents[1] / "scripts" / "build_observable_recovery_labels.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)

ID = dict(dataset=b.SOURCE_DATASET,suite="libero90",task=46,state=10,candidate_id=0,
    observed_block_id="task46_state10_normal/first_block")
HASH = dict(actual_rgb="a"*64,actual_proprio="b"*64,physical_measurement="c"*64)
GOOD = dict(mean_abs=0.,p95=0.,fraction_gt5=0.,psnr=99.)


def frames():
    role = dict(rendered_pixels=100,local_rgb=GOOD,identity_rgb_unique=True,identity_quality=.8,
        identity_basis="audited_actual_rgb_instance_correspondence",actual_rgb_structure_nonconstant=True)
    return [c.certify_frame(dict(identity=ID,step=s,proprio_max_abs=0.,source_hashes_verified=True,
        evidence_sha256=HASH,views={"primary":dict(global_rgb=GOOD,
        roles={r:deepcopy(role) for r in c.ROLE_NAMES},interfaces={})})) for s in range(17)]


def motion(start=2,end=4,subject=(6.,2.),reference=(-8.,8.)):
    return dict(identity=ID,same_candidate_repeat_identity=ID,start_step=start,end_step=end,view="primary",
        source_hashes_verified=True,evidence_sha256=HASH,repeat_noise_verified=True,repeat_noise_px=.1,
        camera_motion_compensated=True,actual_rgb_tracking_verified=True,
        actual_rgb_tracking_basis="actual_rgb_geometric_correspondence",actual_rgb_tracking_points=3,
        actual_subject_delta_px=list(subject),actual_reference_delta_px=list(reference),
        projected_subject_delta_px=list(subject),projected_reference_delta_px=list(reference))


def carry_atom(start=2,end=4,aperture=.064):
    rows = []
    for step in range(start,end+1):
        eef = [.003*(step-start),0.,.1]
        rows.append(dict(step=step,identity=deepcopy(ID),physics=dict(eef_pos_m=eef,
            target_pos_m=[eef[0],0.,.12],gripper_aperture_m=aperture,
            left_finger_target_contact=True,right_finger_target_contact=True,target_support_contact=False),
            actual_proprio=dict(source="original_hashed_proprio_npy",sha256=HASH["actual_proprio"],frame=step,
                gripper_qpos_m=[aperture/2,-aperture/2],eef_pos_m=eef,replay_proprio_max_abs=0.)))
    return dict(name="carried_sufficient_evidence",identity=deepcopy(ID),step=end,value=True,
        measurement_valid=True,supervision_mask=False,source="offline_simulator_measurement",
        source_hashes_verified=True,evidence_sha256=deepcopy(HASH),entity_kind="rigid",physical_carry_window=rows)


def certify(a=None,m=None,fs=None):
    fs = frames() if fs is None else fs
    a = carry_atom() if a is None else a
    m = c.certify_temporal_motion_v2(fs,motion() if m is None else m)
    return c.certify_supervision_v2(a,fs,motion_cert=m)


class CarryV2Tests(unittest.TestCase):
    def test_different_depth_surfaces_need_not_have_equal_pixel_motion(self):
        fs=frames(); m=c.certify_temporal_motion_v2(fs,motion())
        self.assertFalse(m["pixel_delta_equality_diagnostic"])
        self.assertTrue(m["independent_surfaces_motion_supported"])
        result=c.certify_supervision_v2(carry_atom(),fs,motion_cert=m)
        self.assertTrue(result["new_mask"])
        self.assertTrue(result["carrying_silver_not_literal_contact_truth"])

    def test_carry_does_not_enable_unseen_literal_contact_heads(self):
        self.assertTrue(certify()["new_mask"])
        a=carry_atom();a["name"]="left_finger_target_contact"
        self.assertFalse(certify(a)["new_mask"])
        a["name"]="right_finger_target_contact"
        self.assertFalse(certify(a)["new_mask"])

    def test_wide_object_actual_aperture_is_not_a_failure(self):
        result=certify(carry_atom(aperture=.08))
        self.assertTrue(result["new_mask"])
        self.assertFalse(result["frozen_physical_window_metrics"]["absolute_aperture_closure_threshold_used"])

    def test_own_surface_actual_projection_threshold_not_relaxed(self):
        m=motion();m["actual_subject_delta_px"][0] += 2.01
        self.assertFalse(certify(m=m)["new_mask"])

    def test_projection_without_actual_tracking_cannot_certify(self):
        m=motion();m["actual_rgb_tracking_verified"]=None
        self.assertFalse(certify(m=m)["new_mask"])

    def test_motion_below_existing_repeat_noise_floor_masked(self):
        m=motion();m["repeat_noise_px"]=3.
        self.assertFalse(certify(m=m)["new_mask"])

    def test_command_cannot_replace_original_actual_proprio(self):
        a=carry_atom();a["physical_carry_window"][0]["actual_proprio"]["source"]="planned_gripper"
        self.assertFalse(certify(a)["new_mask"])
        a=carry_atom();a["gripper_command"]=1.
        with self.assertRaises(ValueError): certify(a)

    def test_wrong_candidate_hash_or_eef_lineage_masked(self):
        for change in ("hash","eef"):
            a=carry_atom()
            pr=a["physical_carry_window"][1]["actual_proprio"]
            if change=="hash":pr["sha256"]="d"*64
            else:pr["eef_pos_m"]=[1.,0.,.1]
            with self.subTest(change=change):self.assertFalse(certify(a)["new_mask"])

    def test_3d_relative_drift_gate_retained(self):
        a=carry_atom();a["physical_carry_window"][1]["physics"]["target_pos_m"][1]=.0031
        self.assertFalse(certify(a)["new_mask"])

    def test_single_frame_contact_and_wrong_physical_kind_masked(self):
        a=carry_atom();a["physical_carry_window"][0]["physics"]["right_finger_target_contact"]=False
        self.assertFalse(certify(a)["new_mask"])
        a=carry_atom();a["entity_kind"]="articulated"
        self.assertFalse(certify(a)["new_mask"])

    def test_cumulative_window_cannot_relabel_short_carry_event(self):
        self.assertFalse(certify(carry_atom(start=2,end=6),motion(start=2,end=6))["new_mask"])

    def test_cached_frame0_or_future_frame_never_backfills(self):
        self.assertFalse(certify(carry_atom(start=0,end=2),motion(start=0,end=2))["new_mask"])
        raw=motion();raw["used_steps"]=[2,4,16]
        with self.assertRaises(ValueError):certify(m=raw)

    def test_values_and_old_mask_preserved_no_event_is_unknown(self):
        a=carry_atom();before=deepcopy(a);r=certify(a)
        self.assertEqual(a,before)
        self.assertEqual(r["physical_value"],before["value"])
        self.assertEqual(r["old_mask"],before["supervision_mask"])
        a["value"]=None;a["measurement_valid"]=False
        r=certify(a)
        self.assertFalse(r["new_mask"])
        self.assertIsNone(r["physical_value"])

    def test_joint_cumulative_reference_is_real_step2_not_cached_start(self):
        fs=frames();m=c.certify_temporal_motion_v2(fs,motion(2,8,reference=(0.,0.)),reference_role="anchor")
        a=carry_atom(end=8);a.update(name="joint_state_changed",entity_kind="articulated",
            joint_unit="rad",joint_delta_from_reference=.02,physical_reference_step=2)
        r=c.certify_supervision_v2(a,fs,motion_cert=m)
        self.assertTrue(r["new_mask"])
        self.assertEqual(r["joint_reference_step"],2)
        self.assertEqual(r["step"],8)
        self.assertTrue(r["cumulative_reference_does_not_backfill_frame0"])

    def test_v2_joint_reference_mismatch_cannot_reuse_early_state(self):
        fs=frames();m=c.certify_temporal_motion_v2(fs,motion(2,8,reference=(0.,0.)),reference_role="anchor")
        a=carry_atom(end=8);a.update(name="joint_state_changed",entity_kind="articulated",
            joint_unit="m",joint_delta_from_reference=.02,physical_reference_step=0)
        self.assertFalse(c.certify_supervision_v2(a,fs,motion_cert=m)["new_mask"])

    def test_literal_contact_cannot_borrow_aggregate_eef_identity(self):
        role=dict(rendered_pixels=100,local_rgb=GOOD,identity_rgb_unique=True,identity_quality=.8,
            identity_basis="audited_actual_rgb_instance_correspondence",actual_rgb_structure_nonconstant=True)
        contact=dict(physics_contact=True,other_surface_role="eef",projection_in_frame=True,
            depth_error_m=0.,rgb_boundary_pixels=3,local_rgb=GOOD,surfaces_rgb_visible=True)
        raw=dict(identity=ID,step=4,proprio_max_abs=0.,source_hashes_verified=True,evidence_sha256=HASH,
            views={"primary":dict(global_rgb=GOOD,roles={r:deepcopy(role) for r in c.ROLE_NAMES},
                independent_finger_roles=dict(right=deepcopy(role)),interfaces=dict(left=contact,right=contact))})
        frame=c.certify_frame_v2(raw)
        self.assertIsNone(frame["views"]["primary"]["interfaces"]["left"]["value"])
        self.assertTrue(frame["views"]["primary"]["interfaces"]["right"]["value"])
        a=carry_atom();a["name"]="left_finger_target_contact"
        self.assertFalse(c.certify_supervision_v2(a,[frame])["new_mask"])

    def test_visible_separation_negative_is_only_its_own_literal_contact(self):
        role=dict(rendered_pixels=100,local_rgb=GOOD,identity_rgb_unique=True,identity_quality=.8,
            identity_basis="audited_actual_rgb_instance_correspondence",actual_rgb_structure_nonconstant=True)
        negative=dict(physics_contact=False,other_surface_role="eef",rgb_boundary_pixels=4,
            local_rgb=GOOD,surfaces_rgb_visible=True,actual_surface_gap_px=4.,
            separation_rgb_verified=True,own_finger_side="left",own_finger_identity_certified=True,
            actual_target_boundary_verified=True,actual_finger_boundary_verified=True,
            same_candidate_repeat_noise_px=.1)
        raw=dict(identity=ID,step=4,proprio_max_abs=0.,source_hashes_verified=True,evidence_sha256=HASH,
            views={"primary":dict(global_rgb=GOOD,roles={r:deepcopy(role) for r in c.ROLE_NAMES},
                independent_finger_roles=dict(left=deepcopy(role)),interfaces=dict(left=negative))})
        frame=c.certify_frame_v2(raw)
        a=carry_atom();a.update(name="left_finger_target_contact",value=False)
        result=c.certify_supervision_v2(a,[frame])
        self.assertTrue(result["new_mask"])
        self.assertFalse(result["physical_value"])
        a.update(name="carried_sufficient_evidence",value=None,measurement_valid=False)
        self.assertFalse(c.certify_supervision_v2(a,[frame])["new_mask"])
        for change in (dict(own_finger_side="right"),dict(same_candidate_repeat_noise_px=None),
            dict(same_candidate_repeat_noise_px=2.),dict(actual_finger_boundary_verified=False)):
            invalid=deepcopy(raw);invalid["views"]["primary"]["interfaces"]["left"].update(change)
            with self.subTest(change=change):
                self.assertIsNone(c.certify_frame_v2(invalid)["views"]["primary"]["interfaces"]["left"]["value"])

    def test_actual_rgb_identity_missing_any_window_frame_masks_carry(self):
        fs=frames();fs[3]["views"]["primary"]["roles"]["target"]["identity_certified"]=None
        self.assertFalse(certify(fs=fs)["new_mask"])


class BuilderV2Tests(unittest.TestCase):
    def proprio_fixture(self):
        values=np.zeros((17,9));values[:,:2]=[.032,-.032];values[:,2:5]=[0.,0.,.1]
        teacher=dict(frames=[dict(step=s,physics=dict(eef_pos_m=[0.,0.,.1],gripper_aperture_m=.064)) for s in range(17)])
        replay=dict(checks=[dict(step=s,proprio_max_abs=0.) for s in range(17)])
        return teacher,replay,values

    def test_hash_linked_original_proprio_timing_and_values(self):
        fixture=self.proprio_fixture();rows=b.actual_proprio_evidence(*fixture,HASH)
        self.assertTrue(rows[0]["cached_diagnostic_only"])
        self.assertFalse(rows[1]["cached_diagnostic_only"])
        self.assertEqual(rows[8]["sha256"],HASH["actual_proprio"])
        self.assertEqual(rows[8]["gripper_qpos_m"],[.032,-.032])

    def test_missing_or_nonfinite_proprio_not_silently_filled(self):
        for change in ("shape","nan","eef"):
            teacher,replay,values=self.proprio_fixture()
            if change=="shape":values=values[:16]
            elif change=="nan":values[3,0]=np.nan
            else:values[3,2]=1.
            with self.subTest(change=change),self.assertRaises(ValueError):
                b.actual_proprio_evidence(teacher,replay,values,HASH)

    def test_v5_source_and_old_v4_names_are_allowed(self):
        import re
        for name in ("known_task_recovery_observation_audit_20261006_v4_view_masks_full",
            "known_task_recovery_observation_audit_20261006_v5_motion_typed_support_pilot"):
            self.assertIsNotNone(re.fullmatch(b.OBSERVATION_SOURCE_PATTERN,name))

    def test_typed_support_requires_exact_nonworld_instance(self):
        good=dict(rendered_pixels=100,local_rgb=GOOD,identity_rgb_unique=True)
        report=dict(passed=True,supports_interface_observability=True,full_patch_in_bounds=True,
            role_reference_floor_passed=True,role_depth_matched_patch_pixels=dict(target=2,finger=2),
            shared_patch_registration={**GOOD,"passed":True},target_registration={**GOOD,"passed":True},
            finger_registration={**GOOD,"passed":True},surface_depth_tolerance_m=.002,
            actual_rgb_boundary_verified=True,actual_rgb_boundary_pixels=2)
        saved=dict(physics_contact=dict(role="support",counterpart_body_id=7,counterpart_body_name="plate",
            support_instance_key="body_7"),interface_evidence=report)
        roles=dict(target=good,support=good,eef=good)
        value=b.interface_input(saved,dict(target_support_contact=True),roles,support_instance_key="body_7")
        self.assertEqual(value["other_surface_role"],"support")
        self.assertIsNone(b.interface_input(saved,dict(target_support_contact=True),roles,support_instance_key="body_8"))
        saved["physics_contact"]["support_instance_key"]="body_0"
        self.assertIsNone(b.interface_input(saved,dict(target_support_contact=True),roles,support_instance_key="body_0"))

    def test_full_v2_builder_preserves_teacher_and_masks_unseen_contact(self):
        fixture_spec=importlib.util.spec_from_file_location("_v1_builder_fixture_for_v2",
            Path(__file__).with_name("test_observable_recovery_labels.py"))
        fixture_module=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture_module)
        teacher,measurements,replay=fixture_module.candidate_fixture()
        values=np.zeros((17,9));values[:,:2]=[.032,-.032]
        for step,frame in enumerate(teacher["frames"]):
            eef=[.003*step,0.,.1];values[step,2:5]=eef
            frame["physics"].update(eef_pos_m=eef,target_pos_m=[eef[0],0.,.12],
                gripper_aperture_m=.064,right_finger_target_contact=True)
            evidence=frame["observation_evidence"]
            for view in evidence["views"].values():
                view["roles"]["left"]=deepcopy(view["roles"]["eef"])
                view["roles"]["right"]=deepcopy(view["roles"]["eef"])
            for interval in evidence["actual_rgb_temporal_intervals"]:
                interval["tracks"]["target"]=fixture_module.tracking_report(object_delta=(6.,2.))
                interval["tracks"]["eef"]=fixture_module.tracking_report(object_delta=(-8.,8.))
                for side in ("left","right"):
                    interval["tracks"][side]=deepcopy(interval["tracks"]["eef"])
                    interval["same_candidate_repeat_noise"][side]=.1
        original=deepcopy(teacher)
        result=b.candidate_certificates(teacher,measurements,replay,HASH,c,
            certificate_version="v2",original_proprio=values)
        self.assertEqual(teacher,original)
        self.assertEqual(result["schema"],b.SCHEMA_V2)
        self.assertFalse(result["labels"][0]["atoms"]["carried_sufficient_evidence"]["new_mask"])
        self.assertTrue(result["labels"][4]["atoms"]["carried_sufficient_evidence"]["new_mask"])
        self.assertFalse(result["labels"][4]["atoms"]["right_finger_target_contact"]["new_mask"])
        self.assertTrue(result["labels"][4]["atoms"]["left_finger_target_contact"]["new_mask"])
        self.assertFalse(result["training_ready"])


if __name__ == "__main__":
    unittest.main()
