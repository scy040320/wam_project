"""CPU checks of label-sidecar construction, using synthetic evidence only.

These fixtures verify conservative validation and camera compensation logic;
they are not collected recovery examples or an estimate of experimental yield.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wam_reranking import recovery_observability_contract as contract


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_observable_recovery_labels.py"
SPEC = importlib.util.spec_from_file_location("_test_observable_recovery_label_builder", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

ID = dict(dataset=builder.SOURCE_DATASET, suite="libero90", task=9, state=10,
          candidate_id=1, observed_block_id="task9_state10_normal/first_block")
HASH = dict(actual_rgb="a" * 64, physical_measurement="b" * 64, private_projection="c" * 64)
GOOD = dict(mean_abs=0., p95=0., fraction_gt5=0., psnr=99., passed=True,
            texture_sufficient=True, actual_texture_range=255.)


def tracking_report(*, camera_delta=(2., 1.), object_delta=(6., 0.), count=3):
    tracks = []
    starts = ([4., 4.], [8., 4.], [4., 8.])
    for start in starts[:count]:
        camera = [start[i] + camera_delta[i] for i in (0, 1)]
        end = [camera[i] + object_delta[i] for i in (0, 1)]
        tracks.append(dict(start_pixel_xy=list(start), end_pixel_xy=end,
            projected_end_pixel_xy=list(end), camera_only_end_pixel_xy=camera,
            forward_backward_error_px=0., projected_endpoint_error_px=0.))
    return dict(verified=True, geometry_motion_support=True,
                evidence_kind="actual_RGB_geometry_correspondence", count=len(tracks), tracks=tracks)


def candidate_fixture(*, fresh_geometry=True):
    frames = []; measured = []; checks = []
    descriptor = dict(rendered_pixels=100, local_rgb=deepcopy(GOOD))
    saved_interface = dict(physics_contact=dict(role="left", counterpart_body_id=7,
        counterpart_body_name="left_finger"), interface_evidence=dict(
        passed=True, supports_interface_observability=True, full_patch_in_bounds=True,
        role_reference_floor_passed=True, role_depth_matched_patch_pixels=dict(target=3, finger=3),
        role_patch_pixels=dict(target=12, finger=17), actual_rgb_boundary_pixels=3,
        actual_rgb_boundary_verified=True, surface_depth_tolerance_m=.001,
        shared_patch_registration=deepcopy(GOOD), target_registration=deepcopy(GOOD),
        finger_registration=deepcopy(GOOD)))
    for step in range(17):
        intervals = []
        if step >= 2:
            # Current identity is based on causal actual RGB intervals, never
            # on a future interval backfilled into frame0/frame1.
            starts = [step - 1] + ([step - 2] if step >= 4 else [])
            intervals = [dict(start_step=start, end_step=step, view=view,
                tracks={role: tracking_report() for role in ("target", "eef", "anchor")},
                same_candidate_repeat_noise={role: .1 for role in ("target", "eef", "anchor")})
                for view in ("primary", "wrist") for start in starts]
        view = dict(fresh_geometry_registered_to_actual_input=fresh_geometry,
                    global_rgb=deepcopy(GOOD),
                    roles={role: deepcopy(descriptor) for role in ("target", "eef", "anchor", "support")},
                    contact_interfaces=[deepcopy(saved_interface)],
                    repeat_noise={role: dict(rgb_repeat=deepcopy(GOOD), centroid_noise_px=.1)
                                  for role in ("target", "eef", "anchor")})
        frames.append(dict(step=step,
            physics=dict(left_finger_target_contact=True, right_finger_target_contact=False,
                         target_support_contact=False),
            observation_evidence=dict(step=step, role="offline_observation_supervision_only",
                actual_rgb_temporal_intervals=intervals,
                same_candidate_repeat_qc=dict(physics_passed=True, all_contact_atoms_match=True,
                    qpos_qvel_max_abs=0., qpos_qvel_p95=0.),
                views={name: deepcopy(view) for name in ("primary", "wrist")})))
        measured.append(dict(step=step, physics={name: dict(value=True, measurement_valid=True,
            supervision_mask=False, source="offline_simulator_measurement") for name in contract.ATOM_NAMES}))
        checks.append(dict(step=step, passed=True, proprio_max_abs=0.))
    teacher = dict(identity=deepcopy(ID), frames=frames, kind="rigid", joint_unit=None)
    measurements = dict(frames=measured, first_measured_events={name: 0 for name in contract.ATOM_NAMES})
    return teacher, measurements, dict(checks=checks)


def certificates(fixture):
    return builder.candidate_certificates(*fixture, deepcopy(HASH), contract)


class TrackSummaryTests(unittest.TestCase):
    def test_actual_camera_only_flow_is_subtracted_from_actual_and_projected(self):
        report = tracking_report(camera_delta=(7., -3.), object_delta=(4., 2.))
        before = deepcopy(report)
        result = builder.track_summary(report)
        self.assertEqual(result["actual"], [4., 2.])
        self.assertEqual(result["projected"], [4., 2.])
        self.assertEqual(result["count"], 3)
        self.assertEqual(report, before)

    def test_camera_motion_alone_does_not_become_object_motion(self):
        result = builder.track_summary(tracking_report(camera_delta=(20., -5.), object_delta=(0., 0.)))
        self.assertEqual(result["actual"], [0., 0.])
        self.assertEqual(result["projected"], [0., 0.])

    def test_actual_flow_is_not_replaced_by_simulator_projection(self):
        report = tracking_report()
        for point in report["tracks"]:
            point["end_pixel_xy"][1] += .5
        result = builder.track_summary(report)
        self.assertEqual(result["actual"], [6., .5])
        self.assertEqual(result["projected"], [6., 0.])

    def test_duplicate_corner_missing_camera_or_two_points_are_not_certified(self):
        duplicate = tracking_report(); duplicate["tracks"][1]["start_pixel_xy"] = duplicate["tracks"][0]["start_pixel_xy"].copy()
        missing = tracking_report(); del missing["tracks"][1]["camera_only_end_pixel_xy"]
        unavailable = tracking_report(); unavailable["tracks"][1]["camera_only_end_pixel_xy"] = None
        for report in (duplicate, missing, unavailable, tracking_report(count=2)):
            with self.subTest(report=report): self.assertIsNone(builder.track_summary(report))

    def test_only_actual_geometry_correspondence_with_matching_count_is_accepted(self):
        for change in (dict(verified=False), dict(geometry_motion_support=False),
                       dict(evidence_kind="RGB_tracking_proxy"), dict(count=4)):
            report = tracking_report(); report.update(change)
            self.assertIsNone(builder.track_summary(report))

    def test_nonfinite_and_above_threshold_point_errors_are_rejected(self):
        for key, value in (("forward_backward_error_px", 1.01), ("projected_endpoint_error_px", 1.01),
                           ("forward_backward_error_px", -1.), ("projected_endpoint_error_px", float("nan"))):
            report = tracking_report(); report["tracks"][0][key] = value
            self.assertIsNone(builder.track_summary(report))
        report = tracking_report(); report["tracks"][0]["camera_only_end_pixel_xy"][0] = float("inf")
        self.assertIsNone(builder.track_summary(report))

    def test_current_role_needs_actual_nonconstant_structure_and_three_verified_points(self):
        descriptor = dict(rendered_pixels=100, local_rgb=deepcopy(GOOD))
        self.assertTrue(builder.current_role(builder.track_summary(tracking_report()), descriptor)["identity_rgb_unique"])
        self.assertFalse(builder.current_role(builder.track_summary(tracking_report(count=2)), descriptor)["identity_rgb_unique"])
        for key, value in (("texture_sufficient", False), ("actual_texture_range", 0.),
                           ("actual_texture_range", None), ("passed", False), ("mean_abs", 3.)):
            bad = deepcopy(descriptor); bad["local_rgb"][key] = value
            self.assertFalse(builder.current_role(builder.track_summary(tracking_report()), bad)["identity_rgb_unique"])


class CandidateCertificatesTests(unittest.TestCase):
    def test_role_patch_area_without_actual_rgb_boundary_cannot_certify_interface(self):
        fixture = candidate_fixture()
        source_view = fixture[0]["frames"][2]["observation_evidence"]["views"]["primary"]
        roles = {role: builder.current_role(builder.track_summary(tracking_report()), source_view["roles"][role])
                 for role in ("target", "eef")}
        saved = deepcopy(source_view["contact_interfaces"][0])
        del saved["interface_evidence"]["actual_rgb_boundary_pixels"]
        self.assertIsNone(builder.interface_input(saved, fixture[0]["frames"][2]["physics"], roles))

    def test_known_good_contact_chain_has_positive_control_without_frame0_or_frame1_backfill(self):
        fixture = candidate_fixture(); before = deepcopy(fixture)
        result = certificates(fixture)
        self.assertFalse(result["frame_certificates"][0]["reference_certified"])
        self.assertFalse(result["labels"][1]["atoms"]["left_finger_target_contact"]["new_mask"])
        self.assertTrue(result["labels"][2]["atoms"]["left_finger_target_contact"]["new_mask"])
        self.assertTrue(result["labels"][4]["atoms"]["carried_sufficient_evidence"]["new_mask"])
        self.assertEqual(fixture, before)
        self.assertFalse(result["training_started"])
        self.assertFalse(result["training_ready"])

    def test_fresh_geometry_not_registered_masks_all_certificates_and_preserves_gt(self):
        result = certificates(candidate_fixture(fresh_geometry=False))
        self.assertTrue(all(not f["reference_certified"] for f in result["frame_certificates"]))
        self.assertTrue(all(not atom["new_mask"] for row in result["labels"] for atom in row["atoms"].values()))
        self.assertTrue(all(atom["physical_value"] is True for row in result["labels"] for atom in row["atoms"].values()))
        self.assertTrue(all(row["certificate"]["value"] is None for row in result["motion_certificates"]))

    def test_one_unregistered_camera_does_not_mask_registered_other_view(self):
        fixture = candidate_fixture()
        for frame in fixture[0]["frames"]:
            frame["observation_evidence"]["views"]["primary"]["fresh_geometry_registered_to_actual_input"] = False
        result = certificates(fixture)
        self.assertTrue(result["labels"][2]["atoms"]["left_finger_target_contact"]["new_mask"])
        self.assertFalse(result["frame_certificates"][2]["views"]["primary"]["registered_actual"])
        self.assertTrue(result["frame_certificates"][2]["views"]["wrist"]["registered_actual"])

    def test_none_interval_repeat_noise_is_not_synthesized_as_zero(self):
        fixture = candidate_fixture()
        for frame in fixture[0]["frames"]:
            for interval in frame["observation_evidence"]["actual_rgb_temporal_intervals"]:
                interval["same_candidate_repeat_noise"]["target"] = None
        result = certificates(fixture)
        self.assertGreater(len(result["motion_certificates"]), 0)
        for row in result["motion_certificates"]:
            self.assertIsNone(row["input"]["repeat_noise_px"])
            self.assertFalse(row["input"]["repeat_noise_verified"])
            self.assertIsNone(row["certificate"]["value"])
        self.assertFalse(result["labels"][4]["atoms"]["carried_sufficient_evidence"]["new_mask"])

    def test_none_same_candidate_centroid_repeat_noise_is_not_replaced(self):
        fixture = candidate_fixture()
        for frame in fixture[0]["frames"]:
            for view in frame["observation_evidence"]["views"].values():
                view["repeat_noise"]["target"]["centroid_noise_px"] = None
        result = certificates(fixture)
        self.assertTrue(all(not row["input"]["repeat_noise_verified"] for row in result["motion_certificates"]))
        self.assertTrue(all(row["certificate"]["value"] is None for row in result["motion_certificates"]))

    def test_missing_support_or_separation_cannot_unmask_lift_release(self):
        result = certificates(candidate_fixture())
        for name in ("target_support_contact", "lifted_sufficient_evidence", "released_sufficient_evidence"):
            self.assertTrue(all(not row["atoms"][name]["new_mask"] for row in result["labels"]))
            self.assertIsNone(result["first_certified_events"][name])

    def test_full_17_frames_causal_indices_and_observation_role_required(self):
        short = candidate_fixture(); short[0]["frames"].pop()
        wrong_index = candidate_fixture(); wrong_index[2]["checks"][4]["step"] = 5
        pure_physics = candidate_fixture(); pure_physics[0]["frames"][3]["observation_evidence"]["role"] = "offline_supervision_only"
        for fixture in (short, wrong_index, pure_physics):
            with self.assertRaises(ValueError): certificates(fixture)


class SourceBoundaryTests(unittest.TestCase):
    def test_failed_source_rejected_before_manifest_or_output_writes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="observable-label-test-") as directory:
            root = Path(directory); source = root / "outputs" / "known_task_recovery_observation_audit_20261006_v4_view_masks_pilot"
            source.mkdir(parents=True); (source / "failure.json").write_text("{}", encoding="utf-8")
            output = root / "outputs" / "new_labels"
            with self.assertRaisesRegex(ValueError, "Failed/incomplete source"):
                builder.build(root, source, output)
            self.assertFalse(output.exists())
            self.assertTrue((source / "failure.json").is_file())

    def test_relative_evidence_cannot_escape_root_or_use_absolute_path(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="observable-label-test-") as directory:
            root = Path(directory) / "source"; root.mkdir()
            for path in ("../outside.npy", str(Path(directory).resolve() / "outside.npy"), "."):
                with self.subTest(path=path), self.assertRaises(ValueError): builder.contained(root, path)
            self.assertEqual(builder.contained(root, "candidate/array.npy"), root.resolve() / "candidate" / "array.npy")

    def test_external_source_and_source_descendant_output_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="observable-label-test-") as directory:
            root = Path(directory) / "root"; root.mkdir()
            source = Path(directory) / "known_task_recovery_observation_audit_20261006_v4_view_masks_pilot"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "Only the named"):
                builder.build(root, source, root / "new_labels")
            with self.assertRaisesRegex(ValueError, "Refuse overwrite"):
                builder.build(root, source, source / "mutating_output")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix="observable-label-test-") as directory:
            root = Path(directory); source = root / "outputs" / "known_task_recovery_observation_audit_20261006_v4_view_masks_pilot"
            source.mkdir(parents=True); output = root / "existing_labels"; output.mkdir()
            marker = output / "user_result.txt"; marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Refuse overwrite"):
                builder.build(root, source, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class ActualRGBBoundaryTests(unittest.TestCase):
    class Arrays(dict):
        def __enter__(self): return self
        def __exit__(self, *_): return False

    def inputs(self, *, contrast=150., repeat_noise=.1, fresh=True):
        teacher = candidate_fixture(fresh_geometry=fresh)[0]
        ys, xs = np.indices((256, 256))
        segment = np.where(xs < 128, 1, 2).astype(np.int32)
        seg = np.repeat(segment[None], 17, axis=0)
        rgb_frame = np.full((256, 256, 3), 50., dtype=float)
        rgb_frame[xs >= 128] += contrast
        rgb = np.repeat(rgb_frame[None], 17, axis=0)
        for frame in teacher["frames"]:
            evidence = frame["observation_evidence"]
            evidence["private_collision_vs_visual_mapping"] = dict(target=[dict(geom_id=1)],
                left=[dict(geom_id=2)], right=[dict(geom_id=3)])
            for view in evidence["views"].values():
                view["repeat_noise"]["left"] = dict(rgb_repeat=dict(mean_abs=repeat_noise))
                view["repeat_noise"]["target"]["rgb_repeat"]["mean_abs"] = repeat_noise
                view["contact_interfaces"][0]["interface_evidence"].update(
                    contact_pixel_xy=[128., 128.], patch_radius_px=3.)
        return teacher, seg, rgb

    def measure(self, teacher, seg, rgb):
        def saved(path, **kwargs):
            if str(path).endswith(".npz"):
                return self.Arrays(primary=seg, wrist=seg)
            return rgb
        with patch.object(builder.np, "load", side_effect=saved):
            return builder.actual_interface_boundaries(teacher, "private_teacher.npz", Path("original_candidate"))

    def report(self, result, step=2):
        return result["frames"][step]["observation_evidence"]["views"]["primary"]["contact_interfaces"][0]["interface_evidence"]

    def test_four_connected_actual_rgb_role_boundary_is_measured_without_source_mutation(self):
        teacher, seg, rgb = self.inputs(); before = deepcopy(teacher)
        result = self.measure(teacher, seg, rgb); report = self.report(result)
        self.assertTrue(report["actual_rgb_boundary_verified"])
        self.assertGreaterEqual(report["actual_rgb_boundary_pixels"], 2)
        self.assertTrue(all(c > 2. for c in report["actual_rgb_cross_role_contrasts"]))
        self.assertFalse(self.report(result, 0)["actual_rgb_boundary_verified"])
        self.assertEqual(teacher, before)

    def test_background_white_pixel_does_not_supply_missing_cross_role_boundary(self):
        teacher, seg, rgb = self.inputs(contrast=0.)
        seg[:, 126, 128] = 0; rgb[:, 126, 128] = 255.
        report = self.report(self.measure(teacher, seg, rgb))
        self.assertEqual(report["actual_rgb_boundary_pixels"], 0)
        self.assertFalse(report["actual_rgb_boundary_verified"])

    def test_diagonal_mask_cooccurrence_is_not_four_connected_boundary(self):
        teacher, seg, rgb = self.inputs()
        seg[:] = 0; seg[:, :128, :128] = 1; seg[:, 128:, 128:] = 2
        report = self.report(self.measure(teacher, seg, rgb))
        self.assertEqual(report["actual_rgb_boundary_pixels"], 0)
        self.assertFalse(report["actual_rgb_boundary_verified"])

    def test_contrast_must_exceed_frozen_floor_and_measured_repeat_noise(self):
        for contrast, noise in ((2., .1), (8., 3.)):
            report = self.report(self.measure(*self.inputs(contrast=contrast, repeat_noise=noise)))
            self.assertEqual(report["actual_rgb_boundary_pixels"], 0)
            self.assertFalse(report["actual_rgb_boundary_verified"])
            self.assertEqual(report["actual_rgb_boundary_contrast_floor"], max(2., 3. * noise))

    def test_none_repeat_noise_abstains_without_substitution_or_zero_boundary(self):
        report = self.report(self.measure(*self.inputs(repeat_noise=None)))
        self.assertIsNone(report["actual_rgb_boundary_pixels"])
        self.assertFalse(report["actual_rgb_boundary_verified"])

    def test_unregistered_geometry_can_be_measured_but_not_certified(self):
        report = self.report(self.measure(*self.inputs(fresh=False)))
        self.assertGreaterEqual(report["actual_rgb_boundary_pixels"], 2)
        self.assertFalse(report["actual_rgb_boundary_verified"])

    def test_mismatched_temporal_shapes_are_rejected(self):
        teacher, seg, rgb = self.inputs()
        with self.assertRaisesRegex(ValueError, "temporal shapes differ"):
            self.measure(teacher, seg[:-1], rgb)


if __name__ == "__main__":
    unittest.main()
