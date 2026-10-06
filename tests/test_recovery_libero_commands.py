"""New recovery schema follows recorded LIBERO Panda commands, not old parser."""
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.contracts import Stage
from wam_reranking.recovery_contract import (
    COMMAND_FEATURE_NAMES, FEATURE_SCHEMA, command_requirements,
    planned_command_timing,
)
from test_recovery_coupling_scores import candidate


class RecoveryLiberoCommandTests(unittest.TestCase):
    def feature_dict(self, actions):
        return dict(zip(COMMAND_FEATURE_NAMES, planned_command_timing(actions)))

    def test_negative_one_is_open_not_grasp_or_close(self):
        actions = np.zeros((16, 7)); actions[:, 6] = -1.
        _, order = command_requirements(candidate(0, .9, stage=Stage.GRASP), actions)
        features = self.feature_dict(actions)
        self.assertIsNone(order["first_close"])
        self.assertIsNone(order["first_release"])
        self.assertEqual(features["has_close"], 0.)
        self.assertEqual(features["has_release"], 0.)
        self.assertEqual(features["close_fraction"], 0.)

    def test_positive_one_is_close_not_release(self):
        actions = np.zeros((16, 7)); actions[:, 6] = 1.
        _, order = command_requirements(candidate(0, .9, stage=Stage.GRASP), actions)
        features = self.feature_dict(actions)
        self.assertEqual(order["first_close"], 0)
        self.assertIsNone(order["first_release"])
        self.assertEqual(features["has_close"], 1.)
        self.assertEqual(features["close_fraction"], 1.)
        self.assertEqual(features["has_release"], 0.)

    def test_close_then_open_assigns_grasp_use_and_release_use_to_actual_plan_times(self):
        actions = np.zeros((16, 7))
        actions[:4, 6] = -1.; actions[4:12, 6] = 1.; actions[12:, 6] = -1.
        actions[8:, 2] = .1
        grasp_uses, order = command_requirements(candidate(0, .9, stage=Stage.GRASP), actions)
        place = replace(candidate(0, .9, stage=Stage.PLACE), required_facts={"place_ready": .85})
        place_uses, _ = command_requirements(place, actions)
        self.assertEqual(grasp_uses[0].at_step, 4)
        self.assertEqual(place_uses[0].at_step, 12)
        self.assertEqual(order["first_release"], 12)
        self.assertTrue(order["closes_before_upward"])
        features = self.feature_dict(actions)
        self.assertEqual(features["first_close_time"], 4/16)
        self.assertEqual(features["first_release_time"], 12/16)
        self.assertEqual(features["release_after_close"], 1.)

    def test_neutral_zero_after_close_does_not_fabricate_opening_or_release(self):
        actions = np.zeros((16, 7)); actions[:4, 6] = 1.; actions[4:, 2] = .1
        _, order = command_requirements(candidate(0, .9, stage=Stage.PLACE), actions)
        self.assertIsNone(order["first_release"])
        features = self.feature_dict(actions)
        self.assertEqual(features["has_release"], 0.)
        self.assertEqual(features["open_upward_fraction"], 0.)

    def test_negative_open_during_upward_is_command_feature_not_physical_grasp_evidence(self):
        actions = np.zeros((16, 7)); actions[:, 6] = -1.; actions[:, 2] = .1
        _, order = command_requirements(candidate(0, .9, stage=Stage.LIFT), actions)
        features = self.feature_dict(actions)
        self.assertEqual(features["open_upward_fraction"], 1.)
        self.assertEqual(features["has_close"], 0.)
        self.assertTrue(order["commanded_close_is_not_grasp_evidence"])

    def test_new_semantics_are_an_explicit_schema_version_not_silent_old_model_reuse(self):
        self.assertEqual(FEATURE_SCHEMA, "cause_dependency_recovery_features_v3_libero_command")


if __name__ == "__main__":
    unittest.main()
