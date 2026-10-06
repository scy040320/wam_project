"""Fake CPU runtime tests only; no MuJoCo model/environment construction."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
import numpy as np

from wam_reranking.hand_contact_supervision import capture_hand_contact_metadata


def fake_runtime():
    names = ["world", "robot0_base", "robot0_right_hand", "gripper0_right_gripper",
             "gripper0_leftfinger", "gripper0_lefttip", "gripper0_rightfinger",
             "gripper0_righttip", "target", "target_part", "other"]
    geom_names = ["mount_collision", "gripper0_hand_visual", "gripper0_hand_collision",
                  "gripper0_finger1_visual", "gripper0_finger1_collision", "gripper0_finger1_pad_collision",
                  "gripper0_finger2_visual", "gripper0_finger2_collision", "gripper0_finger2_pad_collision",
                  "target_g0", "target_part_g0", None]
    visual = [1, 3, 6]
    m = NS(nbody=len(names), ngeom=len(geom_names),
           body_parentid=np.array([0, 0, 1, 2, 3, 4, 3, 6, 0, 8, 0]),
           geom_bodyid=np.array([2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 9, 10]),
           geom_contype=np.array([int(i not in visual) for i in range(12)]),
           geom_conaffinity=np.ones(12, dtype=int),
           geom_group=np.array([int(i in visual) for i in range(12)]),
           body_mocapid=np.full(11, -1, dtype=int), neq=0,
           jnt_bodyid=np.array([1, 8]), jnt_type=np.array([3,0]),
           body_jntadr=np.array([-1,0,-1,-1,-1,-1,-1,-1,1,-1,-1]),
           body_jntnum=np.array([0,1,0,0,0,0,0,0,1,0,0]), nu=1,
           actuator_trntype=np.array([0]), actuator_trnid=np.array([[0, -1]]),
           nplugin=0, dof_bodyid=np.array([1, 8]),
           body_id2name=lambda i: names[i], geom_id2name=lambda i: geom_names[i],
           body_name2id=lambda name: names.index(name))
    for i in visual:
        m.geom_conaffinity[i] = 0
    d = NS(ncon=0, contact=[], eq_active=np.array([], dtype=bool),
           xfrc_applied=np.zeros((11, 6)), qfrc_applied=np.zeros(2))
    gripper = NS(root_body="gripper0_right_gripper",
        contact_geoms=[geom_names[i] for i in [2, 4, 5, 7, 8]],
        visual_geoms=[geom_names[i] for i in visual],
        important_geoms={"left": [geom_names[4], geom_names[5]], "right": [geom_names[7], geom_names[8]]})
    robot = NS(gripper=gripper, robot_model=NS(root_body="robot0_base", eef_name="robot0_right_hand"))
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata-only capture must not execute/reset/render")
    env = NS(sim=NS(model=m, data=d), robots=[robot], step=forbidden, reset=forbidden, render=forbidden)
    return env


def equality(env, typ, first, second, active=True):
    m, d = env.sim.model, env.sim.data
    m.neq=1
    m.eq_type=np.array([typ]);m.eq_obj1id=np.array([first]);m.eq_obj2id=np.array([second])
    m.eq_objtype=np.array([3 if typ == 2 else 1])
    m.eq_data=np.zeros((1, 11));m.eq_active0=np.array([True])
    d.eq_active=np.array([active], dtype=bool)


class HandMetadataTests(unittest.TestCase):
    def test_palm_and_mount_retained_not_only_fingers(self):
        r=capture_hand_contact_metadata(fake_runtime(), "target")
        self.assertEqual(r["full_hand"]["body_ids"], [2,3,4,5,6,7])
        self.assertIn(2,r["full_hand"]["collision_geom_ids"])
        self.assertIn(1,r["full_hand"]["visual_geom_ids"])
        self.assertIn(0,r["full_hand"]["collision_geom_ids"])
        self.assertEqual(r["target"]["geom_ids"], [9,10])

    def test_never_certifies_negative_even_complete(self):
        r=capture_hand_contact_metadata(fake_runtime(), 8)
        self.assertTrue(r["metadata_complete"])
        self.assertEqual(r["attachment"]["status"], "verified_absent")
        self.assertFalse(r["contacts"]["target_fullhand_contact"])
        for key in ("deployment_allowed", "supervision_mask", "held_negative_certificate_issued",
                    "release_certificate_issued", "whole_training_gate_pass"):
            self.assertIs(r[key],False)

    def test_palm_contact_prevents_fullhand_no_contact(self):
        e=fake_runtime();e.sim.data.ncon=1;e.sim.data.contact=[NS(geom1=9,geom2=2)]
        r=capture_hand_contact_metadata(e,8)
        self.assertTrue(r["contacts"]["target_fullhand_contact"])
        self.assertTrue(r["contacts"]["target_any_robot_contact"])

    def test_contact_id_survives_missing_name(self):
        e=fake_runtime();e.sim.data.ncon=1;e.sim.data.contact=[NS(geom1=9,geom2=11)]
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["contacts"]["pairs"][0]["geom_ids"],[9,11])
        self.assertIsNone(r["contacts"]["pairs"][0]["geom_names"][1])

    def test_invalid_contact_id_not_silently_dropped(self):
        e=fake_runtime();e.sim.data.ncon=1;e.sim.data.contact=[NS(geom1=9,geom2=-1)]
        with self.assertRaises(ValueError):capture_hand_contact_metadata(e,8)

    def test_short_contact_array_raises(self):
        e=fake_runtime();e.sim.data.ncon=1
        with self.assertRaises(ValueError):capture_hand_contact_metadata(e,8)

    def test_active_target_weld_is_present(self):
        e=fake_runtime();equality(e,1,8,3)
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["attachment"]["status"],"present")
        self.assertEqual(r["equality"]["active_target_constraint_ids"],[0])

    def test_missing_current_eq_does_not_fallback_initial(self):
        e=fake_runtime();equality(e,1,8,3);del e.sim.data.eq_active
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["attachment"]["status"],"unknown")
        self.assertFalse(r["metadata_complete"])

    def test_missing_eq_object_type_is_unknown_not_body_guess(self):
        e=fake_runtime();equality(e,1,8,3);del e.sim.model.eq_objtype
        self.assertEqual(capture_hand_contact_metadata(e,8)["attachment"]["status"],"unknown")

    def test_site_equality_resolves_site_body_not_site_id(self):
        e=fake_runtime();equality(e,0,0,1)
        e.sim.model.eq_objtype=np.array([6]);e.sim.model.site_bodyid=np.array([8,3])
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["equality"]["rows"][0]["body_ids"],[8,3])
        self.assertEqual(r["attachment"]["status"],"present")

    def test_unknown_active_constraint_masked(self):
        e=fake_runtime();equality(e,99,8,3)
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["attachment"]["status"],"unknown")
        self.assertEqual(r["equality"]["unsupported_active_constraint_ids"],[0])

    def test_tendon_and_flex_are_not_guessed(self):
        for typ in (3,4):
            e=fake_runtime();equality(e,typ,0,0)
            r=capture_hand_contact_metadata(e,8)
            self.assertEqual(r["attachment"]["status"],"unknown")

    def test_joint_constraint_target_to_constant_masks(self):
        e=fake_runtime();equality(e,2,1,-1)
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["equality"]["rows"][0]["body_ids"],[8])
        self.assertEqual(r["attachment"]["status"],"present")

    def test_target_mocap_present(self):
        e=fake_runtime();e.sim.model.body_mocapid[8]=0
        self.assertEqual(capture_hand_contact_metadata(e,8)["attachment"]["status"],"present")

    def test_missing_mocap_is_unknown(self):
        e=fake_runtime();del e.sim.model.body_mocapid
        self.assertEqual(capture_hand_contact_metadata(e,8)["attachment"]["status"],"unknown")

    def test_target_under_robot_is_not_free_object(self):
        e=fake_runtime();e.sim.model.body_parentid[8]=3
        r=capture_hand_contact_metadata(e,8)
        self.assertTrue(r["attachment"]["target_in_robot_subtree"])

    def test_target_actuator_and_external_forces_not_excluded(self):
        e=fake_runtime();e.sim.model.actuator_trnid[0,0]=1
        self.assertEqual(capture_hand_contact_metadata(e,8)["other_drives"]["status"],"present")
        e=fake_runtime();e.sim.data.xfrc_applied[8,0]=.001
        self.assertEqual(capture_hand_contact_metadata(e,8)["other_drives"]["status"],"present")

    def test_missing_force_and_plugin_fields_unknown(self):
        e=fake_runtime();del e.sim.data.qfrc_applied;del e.sim.model.nplugin
        r=capture_hand_contact_metadata(e,8)
        self.assertEqual(r["other_drives"]["status"],"unknown")
        self.assertFalse(r["metadata_complete"])

    def test_freejoint_kind_is_recorded_not_inferred_from_task(self):
        e=fake_runtime();r=capture_hand_contact_metadata(e,8)
        self.assertTrue(r["target"]["exactly_one_root_freejoint"])
        e.sim.model.jnt_type[1]=3
        self.assertFalse(capture_hand_contact_metadata(e,8)["target"]["exactly_one_root_freejoint"])
        del e.sim.model.jnt_type
        r=capture_hand_contact_metadata(e,8)
        self.assertIsNone(r["target"]["exactly_one_root_freejoint"])
        self.assertFalse(r["metadata_complete"])

    def test_nan_or_cyclic_runtime_rejected(self):
        e=fake_runtime();e.sim.model.body_parentid[8]=9
        with self.assertRaises(ValueError):capture_hand_contact_metadata(e,8)
        e=fake_runtime();e.sim.data.xfrc_applied[8,0]=float("nan")
        with self.assertRaises(ValueError):capture_hand_contact_metadata(e,8)

    def test_world_or_bool_not_unique_target(self):
        for target in (0,False):
            with self.assertRaises(ValueError):capture_hand_contact_metadata(fake_runtime(),target)

    def test_explicit_multiarm_selection_required(self):
        e=fake_runtime();g=e.robots[0].gripper;e.robots[0].gripper={"left":g,"right":g}
        e.robots[0].robot_model.eef_name={"left":"robot0_right_hand","right":"robot0_right_hand"}
        with self.assertRaises(ValueError):capture_hand_contact_metadata(e,8)
        self.assertFalse(capture_hand_contact_metadata(e,8,gripper_arm="right")["supervision_mask"])

    def test_wrapper_and_source_scope_without_mutation(self):
        e=fake_runtime();before=e.sim.model.body_parentid.copy()
        identity={"dataset":"aux", "suite":"libero_90", "task":46,"state":10,
                  "candidate_id":1,"observed_block_id":7,"source_step":3}
        r=capture_hand_contact_metadata(NS(env=e),8,source_identity=identity)
        self.assertEqual(r["source_identity"],identity)
        np.testing.assert_array_equal(before,e.sim.model.body_parentid)
        r["source_identity"]["task"]=0
        self.assertEqual(identity["task"],46)


if __name__ == "__main__":
    unittest.main()
