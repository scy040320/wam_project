"""CPU adapter tests with explicit renderer/physics doubles, not real labels.

The fake environment exercises callback plumbing and absent observations. It
does not establish that OpenCV, MuJoCo replay, or physical recovery is valid.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import collect_recovery_observation_evidence as adapter


ATOM_NAMES = ("left_finger_target_contact", "right_finger_target_contact",
              "target_support_contact", "target_anchor_contact")


def gradient_image():
    ys, xs = np.indices((256, 256))
    return np.stack((xs, ys, (xs + ys) % 256), axis=-1).astype(np.uint8)


def exact_image_check(a, b):
    return {"passed": bool(np.array_equal(a, b))}


class FakeReplay:
    """Three-frame pure CPU replay/renderer double; no simulator is started."""

    def __init__(self, *, missing_repeat_target=False, cached_mismatch_step=None,
                 end_surface_depth=1., original_replay_failure=False,
                 frame_count=3, support_contact_step=None,
                 support_in_world_body=False, missing_repeat_support=False,
                 hide_left=False):
        self.missing_repeat_target = missing_repeat_target
        self.cached_mismatch_step = cached_mismatch_step
        self.end_surface_depth = end_surface_depth
        self.original_replay_failure = original_replay_failure
        self.frame_count = frame_count
        self.support_contact_step = support_contact_step
        self.missing_repeat_support = missing_repeat_support
        self.hide_left = hide_left
        self.calls = 0
        self.callback_results = []
        self.rgb = gradient_image()
        model = SimpleNamespace(
            ngeom=7, nbody=6,
            geom_bodyid=np.array([1, 2, 2, 3, 4, 4, 5]),
            body_parentid=np.array([0, 0, 0, 2, 0, 0]),
            body_jntnum=np.array([0, 0, 0, 0, 0, 1]),
            geom_group=np.array([1, 0, 1, 0, 0, 1, 1]),
            cam_fovy=np.array([90., 90.]),
            camera_name2id=lambda name: 0 if name == "agentview" else 1,
            geom_id2name=lambda gid: ("target_visual", "left_collision",
                "left_visual", "left_pad", "right_collision", "right_visual",
                "anchor_visual")[gid],
            body_id2name=lambda bid: ("world", "target", "leftfinger",
                "left_tip", "rightfinger", "anchor")[bid],
        )
        data = SimpleNamespace(ncon=0, cam_xpos=np.zeros((2, 3)),
            cam_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
            body_xpos=np.zeros((6, 3)),
            body_xmat=np.tile(np.eye(3).reshape(1, 9), (6, 1)))
        if support_in_world_body:model.geom_bodyid[6]=0
        self.env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
        self.env.sim.render = self.render
        self.base = SimpleNamespace(bind=self.bind, capture=self.capture,
            replay=self.replay, body_descendants=self.descendants,
            image_check=exact_image_check)

    @staticmethod
    def descendants(_model, root):
        return {2, 3} if root == 2 else {root}

    @staticmethod
    def bind(_env, _task):
        return {"left": {1, 3}, "right": {4}, "target_geoms": {0},
                "anchor_body": 5, "world_geoms": {0, 6}}

    def render(self, *, width, height, camera_name, depth):
        if (width, height) != (256, 256) or not depth:
            raise AssertionError("Unexpected fake renderer request")
        surface = np.full((256, 256), self.surface_depth)
        return np.flipud(self.rgb).copy(), np.flipud(surface).copy()

    def capture(self, _env, _raw, _actual, _binding, step, _convention):
        physics={name: False for name in ATOM_NAMES}
        physics['target_support_contact']=bool(_env.sim.data.ncon)
        return {"step": step, "physics": physics}, {
            "primary": self.geom.copy(), "wrist": self.geom.copy()}

    def replay(self, env, _fork, _collector, binding, _folder, _snapshot,
               _cause, *, save_segments=True):
        if self.original_replay_failure:
            raise RuntimeError("Original trajectory replay mismatch")
        repeat = self.calls > 0
        self.calls += 1
        frames = []
        for step in range(self.frame_count):
            self.geom = np.full((256, 256), -1, np.int32)
            if not (repeat and self.missing_repeat_target and step == 2):
                self.geom[100:112, 100 + step:112 + step] = 0
            if not self.hide_left:self.geom[120:128, 120:128] = 2
            self.geom[130:138, 130:138] = 5
            if not (repeat and self.missing_repeat_support and step==2):
                self.geom[30:38, 30:38] = 6
            contact_active=self.support_contact_step is not None and step>=self.support_contact_step
            env.sim.data.ncon=int(contact_active)
            env.sim.data.contact=[SimpleNamespace(geom1=0,geom2=6,pos=np.array([-.75,.75,-1.]),dist=0.)] if contact_active else []
            env.sim.data.body_xpos[1] = [step / 128., 0., 0.]
            self.surface_depth = self.end_surface_depth if step == 2 else 1.
            sensor = np.flipud(self.rgb).copy()
            if step == self.cached_mismatch_step:
                sensor[:] = 0
            raw = {"agentview_image": sensor.copy(),
                   "robot0_eye_in_hand_image": sensor.copy()}
            actual = {"primary": np.flipud(sensor).copy(),
                      "wrist": np.flipud(sensor).copy()}
            frame, _ = self.base.capture(env, raw, actual, binding, step, 1)
            frames.append(frame)
        arrays = np.zeros((self.frame_count, 2))
        return frames, [], {}, arrays.copy(), arrays.copy()

    def track_double(self, _start, _end, _sm, _em, *, projected_endpoints):
        # A known set of start role pixels tests the PRIVATE projection callback.
        # No claim about Lucas-Kanade or actual-image identity is made here.
        result = projected_endpoints(np.array([[104., 104.], [106., 104.],
                                               [104., 106.]], np.float32))
        self.callback_results.append(result)
        return {"verified": False, "metric_world_motion_certified": False,
                "reason": "explicit_test_double"}

    def run(self):
        camera = ModuleType("robosuite.utils.camera_utils")
        camera.get_real_depth_map = lambda _sim, value: value
        modules = {"robosuite": ModuleType("robosuite"),
                   "robosuite.utils": ModuleType("robosuite.utils"),
                   "robosuite.utils.camera_utils": camera}
        with patch.dict("sys.modules", modules), patch.object(
                adapter, "track_rgb_role", side_effect=self.track_double):
            adapter.attach(self.base, Path("unused_test_root"))
            binding = self.base.bind(self.env, 9)
            result = self.base.replay(self.env, None, None, binding,
                Path("candidate0"), {}, "normal")
        return binding, result


class ObservationAdapterSerializationTests(unittest.TestCase):
    def test_numpy_scalars_nested_arrays_and_nonfinite_values_are_json_safe(self):
        value = {"camera": np.array([[1., np.nan], [np.inf, -np.inf]]),
                 "nested": (np.int64(4), np.float32(.5), np.bool_(True)),
                 "missing": None}
        ready = adapter.json_ready(value)
        self.assertEqual(ready["camera"], [[1., None], [None, None]])
        self.assertEqual(ready["nested"], [4, .5, True])
        self.assertIsNone(ready["missing"])
        self.assertEqual(json.loads(json.dumps(ready, allow_nan=False)), ready)

    def test_json_conversion_does_not_mutate_input_measurements(self):
        array = np.array([np.nan, 2.])
        original = {"values": array, "noise": (None, np.float64(0.))}
        adapter.json_ready(original)
        self.assertIs(original["values"], array)
        self.assertTrue(np.isnan(array[0]))
        self.assertIsInstance(original["noise"], tuple)
        self.assertIsNone(original["noise"][0])

    def test_missing_noise_and_measured_zero_remain_distinct(self):
        ready = adapter.json_ready({"unknown_noise": np.array([np.nan]),
                                    "zero_noise": np.array([0.])})
        self.assertEqual(ready, {"unknown_noise": [None], "zero_noise": [0.]})


class ObservationAdapterDescriptorTests(unittest.TestCase):
    def test_descriptor_uses_only_selected_pixels_and_xy_coordinates(self):
        image = gradient_image()
        mask = np.zeros((256, 256), bool)
        mask[2, 3] = mask[4, 7] = True
        result = adapter.descriptor(mask, image, image.copy())
        self.assertEqual(result["rendered_pixels"], 2)
        self.assertEqual(result["centroid_xy"], [5., 3.])
        self.assertEqual(result["bbox_xyxy"], [3, 2, 7, 4])
        self.assertEqual(result["local_rgb"]["roi_pixels"], 2)
        self.assertTrue(result["local_rgb"]["passed"])

    def test_empty_role_is_unknown_geometry_not_origin_or_zero_noise(self):
        image = gradient_image()
        result = adapter.descriptor(np.zeros((256, 256), bool), image, image)
        self.assertEqual(result["rendered_pixels"], 0)
        self.assertIsNone(result["centroid_xy"])
        self.assertIsNone(result["bbox_xyxy"])
        self.assertFalse(result["local_rgb"]["measurement_valid"])
        self.assertFalse(result["local_rgb"]["passed"])

    def test_local_rgb_mismatch_is_not_hidden_by_unchanged_full_image(self):
        rendered = gradient_image()
        actual = rendered.copy()
        mask = np.zeros((256, 256), bool)
        mask[100:104, 100:104] = True
        actual[mask] = 255 - actual[mask]
        result = adapter.descriptor(mask, actual, rendered)
        self.assertEqual(result["local_rgb"]["roi_pixels"], 16)
        self.assertFalse(result["local_rgb"]["passed"])

    def test_background_texture_cannot_create_role_texture_in_descriptor(self):
        image = np.zeros((256, 256, 3), np.uint8)
        image[0, 0] = 255
        mask = np.zeros((256, 256), bool)
        mask[100:104, 100:104] = True
        result = adapter.descriptor(mask, image, image.copy())
        self.assertTrue(result["local_rgb"]["pixel_metrics_passed"])
        self.assertFalse(result["local_rgb"]["texture_sufficient"])
        self.assertFalse(result["local_rgb"]["passed"])


class ObservationAdapterFakeReplayTests(unittest.TestCase):
    def test_independent_finger_tracks_and_noise_are_saved_for_each_side(self):
        _,result=FakeReplay().run()
        evidence=result[0][2]['observation_evidence']
        for interval in evidence['actual_rgb_temporal_intervals']:
            self.assertEqual(set(interval['tracks']),{'target','eef','anchor','left','right'})
            self.assertEqual(interval['same_candidate_repeat_noise']['left'],0.)
            self.assertEqual(interval['same_candidate_repeat_noise']['right'],0.)
            self.assertTrue(interval['finger_identity_requires_independent_same_side_actual_RGB_tracks'])

    def test_hidden_finger_cannot_borrow_eef_union_or_other_finger_evidence(self):
        case=FakeReplay(hide_left=True)
        old_double=case.track_double
        real_tracker=adapter.track_rgb_role
        def observe_masks(start,end,sm,em,*,projected_endpoints):
            if not np.any(sm) or not np.any(em):
                return real_tracker(start,end,sm,em,projected_endpoints=projected_endpoints)
            return old_double(start,end,sm,em,projected_endpoints=projected_endpoints)
        case.track_double=observe_masks
        _,result=case.run()
        evidence=result[0][2]['observation_evidence']
        view=evidence['views']['primary']
        self.assertEqual(view['roles']['left']['rendered_pixels'],0)
        self.assertGreater(view['roles']['right']['rendered_pixels'],0)
        self.assertGreater(view['roles']['eef']['rendered_pixels'],0)
        for interval in evidence['actual_rgb_temporal_intervals']:
            self.assertFalse(interval['tracks']['left']['verified'])
            self.assertEqual(interval['tracks']['left']['reason'],'empty_role_mask')
            self.assertIsNone(interval['same_candidate_repeat_noise']['left'])
            self.assertEqual(interval['same_candidate_repeat_noise']['right'],0.)

    def test_motion_window_schedule_includes_current_cumulative_step2_without_frame0(self):
        self.assertEqual(adapter.causal_motion_start_steps(0), [])
        self.assertEqual(adapter.causal_motion_start_steps(1), [])
        self.assertEqual(adapter.causal_motion_start_steps(2), [1])
        self.assertEqual(adapter.causal_motion_start_steps(3), [1,2])
        self.assertEqual(adapter.causal_motion_start_steps(4), [1,2])
        self.assertEqual(adapter.causal_motion_start_steps(16), [1,2,14])
        for step in (-1,17,True):
            with self.assertRaises(ValueError):adapter.causal_motion_start_steps(step)

    def test_added_long_window_ends_at_current_and_does_not_annotate_start_frame(self):
        _,result=FakeReplay(frame_count=6).run()
        intervals=result[0][5]['observation_evidence']['actual_rgb_temporal_intervals']
        long=[row for row in intervals if row['start_step']==2]
        self.assertEqual({row['view'] for row in long},{'primary','wrist'})
        self.assertTrue(all(row['end_step']==5 and row['no_future_identity_backfill'] for row in long))
        self.assertTrue(all(row['interval_kind']=='cumulative_from_real_step2' for row in long))
        self.assertEqual(result[0][1]['observation_evidence']['actual_rgb_temporal_intervals'],[])

    def test_typed_support_is_exact_counterpart_body_and_not_all_world_union(self):
        binding,result=FakeReplay(support_contact_step=0).run()
        self.assertNotIn('support_render',binding)
        instance=binding['typed_support_instances']['body_5']
        self.assertEqual({g['geom_id'] for g in instance['visual_geoms']},{6})
        evidence=result[0][2]['observation_evidence']
        role=evidence['views']['primary']['support_instances']['body_5']
        self.assertEqual(role['rendered_pixels'],64)
        self.assertEqual(role['body_name'],'anchor')
        self.assertFalse(role['identity_ambiguous'])
        self.assertNotIn('support',evidence['views']['primary']['roles'])
        contact=evidence['views']['primary']['contact_interfaces'][0]['physics_contact']
        self.assertEqual(contact['support_instance_key'],'body_5')
        for row in evidence['actual_rgb_temporal_intervals']:
            typed=row['typed_support_tracks']['body_5']
            self.assertEqual(typed['body_id'],5)
            self.assertEqual(typed['same_candidate_repeat_noise_px'],0.)
            self.assertFalse(typed['track']['verified'])  # Synthetic backend never certifies RGB.

    def test_future_support_contact_does_not_backfill_frame2_role_identity(self):
        _,result=FakeReplay(frame_count=6,support_contact_step=3).run()
        self.assertEqual(result[0][2]['observation_evidence']['views']['primary']['support_instances'],{})
        for interval in result[0][5]['observation_evidence']['actual_rgb_temporal_intervals']:
            if interval['start_step']==2:
                typed=interval['typed_support_tracks']['body_5']
                self.assertFalse(typed['track']['verified'])
                self.assertEqual(typed['track']['reason'],'typed_support_reference_not_observed_in_causal_prefix')
                self.assertIsNone(typed['same_candidate_repeat_noise_px'])
        self.assertTrue(result[0][3]['observation_evidence']['views']['primary']['support_instances'])

    def test_world_body_shared_surfaces_are_ambiguous_and_not_track_certified(self):
        binding,result=FakeReplay(support_contact_step=0,support_in_world_body=True).run()
        instance=binding['typed_support_instances']['body_0']
        self.assertEqual({g['geom_id'] for g in instance['visual_geoms']},{6})
        self.assertTrue(instance['world_body_not_a_unique_surface_instance'])
        for row in result[0][2]['observation_evidence']['actual_rgb_temporal_intervals']:
            typed=row['typed_support_tracks']['body_0']
            self.assertTrue(typed['identity_ambiguous'])
            self.assertFalse(typed['track']['verified'])
            self.assertEqual(typed['track']['reason'],'world_body_is_not_a_unique_support_surface_instance')

    def test_missing_typed_support_repeat_pixels_keeps_none_not_zero_noise(self):
        _,result=FakeReplay(support_contact_step=0,missing_repeat_support=True).run()
        evidence=result[0][2]['observation_evidence']
        self.assertIsNone(evidence['views']['primary']['support_instances']['body_5']['repeat_noise']['centroid_noise_px'])
        for interval in evidence['actual_rgb_temporal_intervals']:
            self.assertIsNone(interval['typed_support_tracks']['body_5']['same_candidate_repeat_noise_px'])

    def test_visual_roles_include_finger_root_and_tip_without_other_finger(self):
        binding, _ = FakeReplay().run()
        self.assertEqual(binding["left_render"], {1, 2, 3})
        self.assertEqual(binding["right_render"], {4, 5})
        self.assertFalse(binding["left_render"] & binding["right_render"])
        self.assertEqual(binding["left"], {1, 3})  # Contact IDs stay frozen.

    def test_anchor_fixed_children_visible_but_all_joint_branches_are_pruned(self):
        case = FakeReplay()
        model = case.env.sim.model
        # Root body5 has a freejoint and no geom. Its fixed base child6 has
        # geom6, while drawers7/8/9 each add a joint. Even a jointless child10
        # under a moving drawer must be excluded; fixed detail11 is included.
        model.nbody, model.ngeom = 12, 11
        model.body_parentid = np.array([0, 0, 0, 2, 0, 0, 5, 6, 6, 6, 7, 6])
        model.body_jntnum = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
        model.geom_bodyid = np.array([7, 2, 2, 3, 4, 4, 6, 8, 9, 10, 11])
        model.geom_group = np.ones(11, np.int32)
        body_names = ("world", "unused", "leftfinger", "left_tip", "rightfinger",
                      "cabinet_root", "cabinet_base", "top_drawer", "middle_drawer",
                      "bottom_drawer", "top_handle", "fixed_trim")
        model.body_id2name = lambda bid: body_names[bid]
        model.geom_id2name = lambda gid: "geom_" + str(gid)

        def descendants(m, root):
            selected = set()
            for bid in range(m.nbody):
                node = bid
                while node > 0 and node != root:
                    node = int(m.body_parentid[node])
                if node == root:
                    selected.add(bid)
            return selected

        case.base.body_descendants = descendants
        adapter.attach(case.base, Path("unused_test_root"))
        binding = case.base.bind(case.env, 0)
        self.assertEqual(model.body_jntnum[5], 1)  # Root's own joint is allowed.
        self.assertFalse(np.any(model.geom_bodyid == 5))
        self.assertEqual(binding["anchor_render"], {6, 10})
        self.assertFalse(binding["anchor_render"] & {0, 7, 8, 9})
        self.assertEqual({item["body_name"] for item in
                          binding["visual_role_mapping"]["anchor"]},
                         {"cabinet_base", "fixed_trim"})

    def test_repeat_missing_role_noise_remains_none_not_zero(self):
        _, result = FakeReplay(missing_repeat_target=True).run()
        evidence = result[0][2]["observation_evidence"]
        self.assertIsNone(evidence["views"]["primary"]["repeat_noise"]["target"]["centroid_noise_px"])
        for interval in evidence["actual_rgb_temporal_intervals"]:
            self.assertIsNone(interval["same_candidate_repeat_noise"]["target"])
            self.assertEqual(interval["same_candidate_repeat_noise"]["anchor"], 0.)

    def test_repeat_qc_is_explicitly_physics_only(self):
        _, result = FakeReplay().run()
        qc = result[0][2]["observation_evidence"]["same_candidate_repeat_qc"]
        self.assertTrue(qc["physics_passed"])
        self.assertNotIn("passed", qc)

    def test_cached_frame_zero_mismatch_is_diagnostic_and_does_not_abort(self):
        _, result = FakeReplay(cached_mismatch_step=0).run()
        evidence = result[0][0]["observation_evidence"]
        self.assertTrue(evidence["cached_entry_is_diagnostic_only"])
        self.assertFalse(evidence["views"]["primary"]["fresh_render_vs_cached_observation"]["passed"])
        self.assertFalse(evidence["views"]["primary"]["fresh_geometry_registered_to_actual_input"])
        self.assertEqual(evidence["actual_rgb_temporal_intervals"], [])

    def test_post_action_fresh_geometry_mismatch_abstains_without_replacing_input(self):
        _, result = FakeReplay(cached_mismatch_step=1).run()
        evidence = result[0][1]["observation_evidence"]
        view = evidence["views"]["primary"]
        self.assertFalse(evidence["cached_entry_is_diagnostic_only"])
        self.assertFalse(view["fresh_render_vs_cached_observation"]["passed"])
        self.assertFalse(view["global_rgb"]["passed"])
        self.assertFalse(view["fresh_geometry_registered_to_actual_input"])
        self.assertTrue(evidence["physical_events_do_not_certify_input_observability"])
        self.assertEqual(len(result[0]), 3)

    def test_original_replay_hard_failure_is_not_swallowed_by_adapter(self):
        with self.assertRaisesRegex(RuntimeError, "Original trajectory replay mismatch"):
            FakeReplay(original_replay_failure=True).run()

    def test_real_post_action_intervals_do_not_use_cached_frame_zero(self):
        _, result = FakeReplay().run()
        for frame in result[0]:
            for interval in frame["observation_evidence"]["actual_rgb_temporal_intervals"]:
                self.assertGreaterEqual(interval["start_step"], 1)
                self.assertLess(interval["start_step"], interval["end_step"])
                self.assertLessEqual(interval["end_step"], frame["step"])
                self.assertTrue(interval["correspondence_camera_pose_accounted"])
                self.assertFalse(interval["camera_motion_compensated_by_per_frame_projection"])

    def test_projection_callback_accepts_registered_visible_material_points(self):
        case = FakeReplay()
        case.run()
        self.assertTrue(case.callback_results)
        for result in case.callback_results:
            self.assertTrue(np.asarray(result["valid"]).all())
            np.testing.assert_allclose(result["end_pixel_xy"] - result["start_pixel_xy"], [[1., 0.]] * 3)
            np.testing.assert_allclose(result["camera_only_end_pixel_xy"], result["start_pixel_xy"])

    def test_projection_callback_rejects_self_occluded_end_surface(self):
        case = FakeReplay(end_surface_depth=.5)
        case.run()
        self.assertTrue(case.callback_results)
        for result in case.callback_results:
            self.assertFalse(np.asarray(result["valid"]).any())


if __name__ == "__main__":
    unittest.main()
