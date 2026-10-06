"""Synthetic CPU invariants, not experimental upward-motion coverage."""
from copy import deepcopy
import unittest

import numpy as np

from wam_reranking import typed_upward_motion as u
from wam_reranking.recovery_observability_contract import SCHEMA_V2

ID=dict(dataset="source160",suite="libero90",task=46,state=10,candidate_id=0,
        observed_block_id="task46_state10_normal/first_block")
HASH=dict(actual_rgb="a"*64,physical_measurement="b"*64,private_projection="c"*64)
GOOD=dict(mean_abs=0.,p95=0.,fraction_gt5=0.,psnr=99.,passed=True,
          texture_sufficient=True,actual_texture_range=20.)


def camera(position):
    position=np.asarray(position,float);back=position-np.array([0.,0.,.04]);back/=np.linalg.norm(back)
    right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right);up=np.cross(back,right)
    return dict(camera_position_m=position.tolist(),camera_rotation_world_from_camera=np.column_stack((right,up,back)).tolist(),
        fovy_degrees=45.,image_shape=[256,256],image_convention=1)


def report(cam,points,delta):
    calibrated=u._camera(cam);rows=[]
    for point in points:
        start=u._project(calibrated,np.asarray(point));end=u._project(calibrated,np.asarray(point)+delta)
        rows.append(dict(start_pixel_xy=start.tolist(),end_pixel_xy=end.tolist(),
            projected_end_pixel_xy=end.tolist(),camera_only_end_pixel_xy=start.tolist(),
            forward_backward_error_px=0.,projected_endpoint_error_px=0.))
    return dict(verified=True,count=len(rows),geometry_motion_support=True,
        evidence_kind="actual_RGB_geometry_correspondence",tracks=rows)


def fixture(delta=(0.,0.,.03)):
    key="body_7";cameras=dict(primary=camera([0.,-1.,.05]),wrist=camera([.7,-.7,.2]))
    points=[[-.04,0.,.04],[0.,.02,.04],[.04,-.02,.04]]
    support=[[-.1,0.,0.],[0.,.08,0.],[.1,-.05,0.]]
    final={v:dict(target=report(cam,points,np.asarray(delta)),support=report(cam,support,np.zeros(3))) for v,cam in cameras.items()}
    frames=[]
    for step in range(17):
        evidence=dict(step=step,role="offline_observation_supervision_only",views={},
            private_typed_support_mapping={key:dict(body_id=7,support_instance_key=key,identity_ambiguous=False)},
            same_candidate_repeat_qc=dict(physics_passed=True,all_contact_atoms_match=True,qpos_qvel_max_abs=0.,qpos_qvel_p95=0.),
            actual_rgb_temporal_intervals=[])
        for view,cam in cameras.items():
            noise=dict(centroid_noise_px=0.,rgb_repeat=deepcopy(GOOD))
            desc=dict(rendered_pixels=100,local_rgb=deepcopy(GOOD),repeat_noise=deepcopy(noise),
                support_instance_key=key,body_id=7,identity_ambiguous=False)
            evidence["views"][view]=dict(camera=deepcopy(cam),fresh_geometry_registered_to_actual_input=True,
                global_rgb=deepcopy(GOOD),roles=dict(target=dict(rendered_pixels=100,local_rgb=deepcopy(GOOD))),
                support_instances={key:desc},repeat_noise=dict(target=deepcopy(noise)))
            if step>=2:
                evidence["actual_rgb_temporal_intervals"].append(dict(start_step=1,end_step=step,view=view,
                    tracks=dict(target=deepcopy(final[view]["target"])),same_candidate_repeat_noise=dict(target=0.),
                    typed_support_tracks={key:dict(identity_ambiguous=False,support_instance_key=key,body_id=7,
                        same_candidate_repeat_noise_px=0.,track=deepcopy(final[view]["support"]))}))
            if step==6:
                interval=deepcopy(evidence["actual_rgb_temporal_intervals"][-1]);interval["start_step"]=2
                evidence["actual_rgb_temporal_intervals"].append(interval)
        frames.append(dict(step=step,identity=deepcopy(ID),proprio_max_abs=0.,observation_evidence=evidence))
    direction=dict(basis=u.CORRESPONDENCE_BASIS,actual_RGB_correspondence_audited=True,
        matches=[dict(primary_target_track_index=i,wrist_target_track_index=i,actual_RGB_same_landmark_verified=True) for i in range(3)])
    visual=dict(identity=deepcopy(ID),start_step=2,end_step=6,support_instance_key=key,
        source_hashes_verified=True,evidence_sha256=deepcopy(HASH),world_vertical_direction_evidence=direction)
    label=dict(identity=deepcopy(ID),reference_step=2,step=6,source="offline_simulator_measurement",
        measurement_valid=True,vertical_rise_from_reference_m=.03,source_hashes_verified=True,evidence_sha256=deepcopy(HASH))
    return frames,visual,label


class TypedUpwardMotionTests(unittest.TestCase):
    def test_two_actual_calibrated_views_identify_new_upward_target_only(self):
        args=fixture();before=deepcopy(args);result=u.certify_typed_upward_motion(*args)
        self.assertTrue(result["mask"]);self.assertTrue(result["visual_witness"]["world_vertical_identified"])
        self.assertEqual(result["name"],"typed_upward_motion")
        self.assertFalse(result["maps_to_literal_lifted_atom"]);self.assertFalse(result["deployment_allowed"])
        self.assertEqual(args,before);self.assertFalse(result["carried_and_raised"]["mask"])

    def test_hashed_physics_rise_without_actual_cross_view_pairs_abstains(self):
        frames,visual,label=fixture();visual.pop("world_vertical_direction_evidence")
        result=u.certify_typed_upward_motion(frames,visual,label)
        self.assertFalse(result["mask"]);self.assertIsNone(result["value"])
        self.assertIn("missing_actual_RGB_cross_view_correspondence",result["reasons"])

    def test_two_separately_detected_view_corners_are_not_same_landmarks(self):
        frames,visual,label=fixture();visual["world_vertical_direction_evidence"]["basis"]="same_simulator_body_points"
        self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_horizontal_2d_motion_does_not_certify_world_up(self):
        self.assertFalse(u.certify_typed_upward_motion(*fixture(delta=(.03,0.,0.)))["mask"])

    def test_downward_or_below_frozen_actual_1cm_not_upward(self):
        for delta in ((0.,0.,-.03),(0.,0.,.009)):
            self.assertFalse(u.certify_typed_upward_motion(*fixture(delta))["mask"])

    def test_one_hidden_or_unregistered_view_does_not_borrow_other_view(self):
        for field in ("fresh_geometry_registered_to_actual_input","support"):
            frames,visual,label=fixture();view=frames[3]["observation_evidence"]["views"]["wrist"]
            if field=="support":view["support_instances"]["body_7"]["local_rgb"]["passed"]=False
            else:view[field]=False
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_full_prefix_identity_cannot_be_backfilled_from_end(self):
        frames,visual,label=fixture();frames[2]["observation_evidence"]["actual_rgb_temporal_intervals"]=[]
        self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_no_world_body_or_different_counterpart_binding(self):
        for key in ("body_0","body_8"):
            frames,visual,label=fixture();visual["support_instance_key"]=key
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_own_noise_missing_is_never_zero_or_borrowed(self):
        for change in ("repeat","interval"):
            frames,visual,label=fixture();ev=frames[6]["observation_evidence"]
            if change=="repeat":ev["views"]["wrist"]["support_instances"]["body_7"]["repeat_noise"]["centroid_noise_px"]=None
            else:
                for interval in ev["actual_rgb_temporal_intervals"]:
                    interval["typed_support_tracks"]["body_7"]["same_candidate_repeat_noise_px"]=None
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_signed_axis_signal_must_exceed_original_3x_repeat_noise(self):
        frames,visual,label=fixture()
        for frame in frames[2:7]:
            frame["observation_evidence"]["views"]["wrist"]["support_instances"]["body_7"]["repeat_noise"]["centroid_noise_px"]=10.
        self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_degenerate_same_camera_pair_cannot_identify_3d_direction(self):
        frames,visual,label=fixture()
        for frame in frames:
            frame["observation_evidence"]["views"]["wrist"]["camera"]=deepcopy(frame["observation_evidence"]["views"]["primary"]["camera"])
        self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_duplicate_or_fewer_than_three_landmarks_not_certificate(self):
        for change in ("duplicate","count","audit"):
            frames,visual,label=fixture();direction=visual["world_vertical_direction_evidence"]
            if change=="count":direction["matches"].pop()
            elif change=="audit":direction["matches"][0]["actual_RGB_same_landmark_verified"]=False
            else:direction["matches"][1]["primary_target_track_index"]=0
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_actual_projection_fit_or_LK_threshold_not_relaxed(self):
        for change in ("fit","lk","projection","duplicates","camera_only"):
            frames,visual,label=fixture()
            for interval in frames[6]["observation_evidence"]["actual_rgb_temporal_intervals"]:
                point=interval["tracks"]["target"]["tracks"][0]
                if change=="fit":point["end_pixel_xy"][0]+=20.
                elif change=="lk":point["forward_backward_error_px"]=1.001
                elif change=="projection":point["projected_endpoint_error_px"]=1.001
                elif change=="duplicates":point["start_pixel_xy"]=interval["tracks"]["target"]["tracks"][1]["start_pixel_xy"]
                else:point.pop("camera_only_end_pixel_xy")
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_constant_role_background_structure_cannot_certify_identity(self):
        frames,visual,label=fixture();local=frames[3]["observation_evidence"]["views"]["primary"]["roles"]["target"]["local_rgb"]
        local["actual_texture_range"]=0.
        self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_fixed_RGB_numeric_gates_not_merely_passed_flag(self):
        frames,visual,label=fixture();frames[3]["observation_evidence"]["views"]["primary"]["global_rgb"]["fraction_gt5"] = .06001
        self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_repeat_QC_and_proprio_gates_not_relaxed(self):
        for change in ("qc","proprio"):
            frames,visual,label=fixture()
            if change=="qc":frames[3]["observation_evidence"]["same_candidate_repeat_qc"]["qpos_qvel_p95"]=.000101
            else:frames[3]["proprio_max_abs"]=.00101
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_causal_reference_frame1_or_future_used_frame_rejected(self):
        for change in ("start","future"):
            frames,visual,label=fixture()
            if change=="start":visual["start_step"]=label["reference_step"]=1
            else:visual["used_steps"]=[2,3,4,5,6,16]
            self.assertFalse(u.certify_typed_upward_motion(frames,visual,label)["mask"])

    def test_Y_source_reference_hash_identity_and_positive_are_independent_gates(self):
        for change in ("hash","identity","step","source","rise","valid"):
            frames,visual,label=fixture()
            if change=="hash":label["evidence_sha256"]["actual_rgb"]="e"*64
            elif change=="identity":label["identity"]["candidate_id"]=1
            elif change=="step":label["reference_step"]=0
            elif change=="source":label["source"]="command"
            elif change=="rise":label["vertical_rise_from_reference_m"]=0.
            else:label["measurement_valid"]=False
            result=u.certify_typed_upward_motion(frames,visual,label)
            self.assertIsNone(result["value"]);self.assertFalse(result["negative_lift_or_grasp_inferred"])

    def test_physics_must_not_be_smuggled_into_visual_frames(self):
        frames,visual,label=fixture();frames[3]["physics"]=dict(target_pos_m=[0.,0.,.03])
        with self.assertRaises(ValueError):u.certify_typed_upward_motion(frames,visual,label)

    def test_foreign_frame_or_forbidden_prediction_does_not_enable_label(self):
        frames,visual,label=fixture();frames[3]["identity"]["candidate_id"]=1
        with self.assertRaises(ValueError):u.certify_typed_upward_motion(frames,visual,label)
        frames,visual,label=fixture();visual["predicted_cause"]="occlusion"
        with self.assertRaises(ValueError):u.certify_typed_upward_motion(frames,visual,label)

    def test_composite_carry_is_separate_current_source_scoped_certificate(self):
        frames,visual,label=fixture();carry=dict(schema=SCHEMA_V2,name="carried_sufficient_evidence",step=6,
            identity=deepcopy(ID),new_mask=True,physical_value=True,measurement_valid=True,
            role="offline_supervision_only",deployment_allowed=False)
        result=u.certify_typed_upward_motion(frames,visual,label,carried_event_cert=carry)
        self.assertTrue(result["carried_and_raised"]["mask"])
        carry["identity"]["candidate_id"]=1
        result=u.certify_typed_upward_motion(frames,visual,label,carried_event_cert=carry)
        self.assertTrue(result["mask"]);self.assertFalse(result["carried_and_raised"]["mask"])

    def test_camera_flip_roundtrip_and_offscreen_no_clipping(self):
        cam=camera([0.,-1.,.05]);point=np.array([0.,0.,.07]);normal=u._project(u._camera(cam),point)
        cam["image_convention"]=-1;flipped=u._project(u._camera(cam),point)
        np.testing.assert_allclose(flipped,[normal[0],255-normal[1]])
        self.assertIsNone(u._ray(u._camera(cam),[-1.,128.]))
        self.assertIsNone(u._project(u._camera(cam),np.array([10.,0.,.07])))

    def test_sidecar_without_new_landmarks_preserves_teacher_and_produces_no_certificate(self):
        frames,visual,label=fixture();teacher=dict(identity=deepcopy(ID),frames=[])
        for f in frames:
            teacher["frames"].append(dict(step=f["step"],physics=dict(target_pos_m=[0.,0.,.04 if f["step"]<6 else .07]),
                observation_evidence=deepcopy(f["observation_evidence"])))
        replay=dict(checks=[dict(step=s,proprio_max_abs=0.) for s in range(17)])
        before=deepcopy(teacher)
        sidecar=u.derive_typed_upward_motion_sidecar(teacher,replay,HASH,source_hashes_verified=True)
        self.assertEqual(teacher,before);self.assertEqual(len(sidecar["labels"]),17)
        self.assertFalse(any(x["certificate"]["mask"] for x in sidecar["labels"]))
        self.assertIn("missing_actual_RGB_cross_view_correspondence",sidecar["labels"][6]["all_attempt_reasons"])
        self.assertTrue(sidecar["literal_target_support_and_lifted_masks_untouched"])
        sidecar=u.derive_typed_upward_motion_sidecar(teacher,replay,HASH,source_hashes_verified=True,
            cross_view_landmarks={(2,6,"body_7"):visual["world_vertical_direction_evidence"]})
        self.assertTrue(sidecar["labels"][6]["certificate"]["mask"])

    def test_sidecar_source_failure_and_bad_index_not_certified(self):
        frames,visual,label=fixture();teacher=dict(identity=deepcopy(ID),frames=[dict(step=f["step"],
            physics=dict(target_pos_m=[0.,0.,.04 if f["step"]<6 else .07]),observation_evidence=f["observation_evidence"]) for f in frames])
        replay=dict(checks=[dict(step=s,proprio_max_abs=0.) for s in range(17)])
        sidecar=u.derive_typed_upward_motion_sidecar(teacher,replay,HASH,source_hashes_verified=False,
            cross_view_landmarks={(2,6,"body_7"):visual["world_vertical_direction_evidence"]})
        self.assertFalse(any(x["certificate"]["mask"] for x in sidecar["labels"]))
        replay["checks"][3]["step"]=16
        with self.assertRaises(ValueError):u.derive_typed_upward_motion_sidecar(teacher,replay,HASH,source_hashes_verified=True)


if __name__ == "__main__":unittest.main()
