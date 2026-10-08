"""Actual-history source binding, not tests of a physical-state detector."""
from dataclasses import replace
import unittest

import numpy as np

from wam_reranking.current_observation_evidence import (
    ActualObservationInputs, array_sha256, make_actual_inputs,
)
from wam_reranking.current_observation_window import (
    ActualObservationWindow, make_actual_observation_window,
)


def frames():
    return tuple(make_actual_inputs(pool_id="runtime-pool", entity_key="butter",
        anchor_key="basket", block_index=2 if step < 48 else 3, policy_step=step,
        actual_primary=np.full((8, 8, 3), step, np.uint8),
        actual_wrist=np.full((6, 8, 3), step + 1, np.uint8),
        actual_proprio=np.arange(9, dtype=float) + step)
        for step in (46, 47, 48))


def window(observations=None, use_step=48):
    observations = observations or frames()
    return make_actual_observation_window(recipient_snapshot=observations[-1].snapshot,
        use_policy_step=use_step, actual_frames=observations,
        actual_requested=np.zeros((len(observations) - 1, 7), np.float32),
        actual_applied=np.ones((len(observations) - 1, 7), np.float32))


class ActualObservationWindowTests(unittest.TestCase):
    def test_literal_window_has_canonical_content_manifest_and_no_facts(self):
        value = window()
        value.validate_for(value.frames[-1].snapshot, 48)
        record = value.audit_record()
        self.assertEqual(record["command_policy_steps"], [46, 47])
        self.assertEqual(record["actual_frames"], 3)
        self.assertEqual(record["predicate_facts"], [])
        self.assertFalse(record["physical_state_certified"])
        self.assertFalse(record["binding_is_physical_certificate"])
        self.assertFalse(record["upstream_capture_provenance_independently_certified"])
        self.assertTrue(record["command_values_are_not_measured_robot_motion"])
        self.assertNotIn("grasped", record)
        self.assertNotIn("lifted", record)

    def test_manifest_is_deterministic_and_changes_when_actual_commands_change(self):
        a, b = window(), window()
        self.assertEqual(a.identity_sha256, b.identity_sha256)
        request = np.full((2, 7), .25, np.float32)
        changed = replace(b, requested=request, requested_sha256=array_sha256(request))
        self.assertNotEqual(a.identity_sha256, changed.identity_sha256)

    def test_commands_are_copied_read_only(self):
        observations = frames()
        requested = np.zeros((2, 7), np.float32)
        value = make_actual_observation_window(recipient_snapshot=observations[-1].snapshot,
            use_policy_step=48, actual_frames=observations,
            actual_requested=requested, actual_applied=requested)
        requested[0, 0] = 9
        self.assertEqual(value.requested[0, 0], 0)
        self.assertFalse(value.requested.flags.writeable)
        with self.assertRaises(ValueError):
            value.requested[0, 0] = 1

    def test_mutated_commands_are_rehashed_before_every_use(self):
        value = window()
        value.applied.setflags(write=True)
        value.applied[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "declared SHA256"):
            value.audit_record()

    def test_mutated_actual_frame_is_rehashed_before_use(self):
        value = window()
        value.frames[0].primary.setflags(write=True)
        value.frames[0].primary[0, 0, 0] = 99
        with self.assertRaisesRegex(ValueError, "content changed"):
            value.validate()

    def test_final_frame_must_equal_independently_pinned_recipient(self):
        value = window()
        bad = replace(value.recipient_snapshot, primary_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "Final actual frame"):
            replace(value, recipient_snapshot=bad)

    def test_other_pool_entity_anchor_or_time_cannot_borrow_window(self):
        value = window()
        for field, other in (("pool_id", "another-pool"), ("entity_key", "cream cheese"),
                             ("anchor_key", "plate"), ("policy_step", 49),
                             ("primary_sha256", "b" * 64)):
            with self.subTest(field=field):
                bad = replace(value.recipient_snapshot, **{field: other})
                with self.assertRaisesRegex(ValueError, "exact recipient"):
                    value.validate_for(bad, 48)

    def test_use_time_is_bound_and_cannot_be_changed_after_admission(self):
        value = window(use_step=49)
        value.validate_for(value.recipient_snapshot, 49)
        with self.assertRaisesRegex(ValueError, "exact recipient"):
            value.validate_for(value.recipient_snapshot, 50)

    def test_future_actual_frame_cannot_backfill_earlier_use(self):
        with self.assertRaisesRegex(ValueError, "after the prerequisite use time"):
            window(use_step=47)

    def test_use_step_rejects_bool_fraction_and_negative(self):
        for bad in (True, 48., -1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                window(use_step=bad)

    def test_teacher_prediction_and_future_roles_are_rejected(self):
        value = window()
        for role in ("offline_supervision_only", "candidate_prediction", "simulator_ground_truth"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "runtime actual PRE"):
                replace(value, source_role=role)
        with self.assertRaises(ValueError):
            replace(value, predecision=False)

    def test_observation_gaps_and_repeated_frames_are_rejected(self):
        observations = frames()
        for bad in ((observations[0], observations[2]),
                    (observations[0], observations[0], observations[2]),
                    tuple(reversed(observations))):
            with self.subTest(steps=[f.snapshot.policy_step for f in bad]), self.assertRaises(ValueError):
                window(observations=bad)

    def test_different_frame_entity_and_extractor_are_rejected(self):
        observations = frames()
        for field, other in (("pool_id", "teacher-pool"), ("entity_key", "other-instance"),
                             ("anchor_key", "other-anchor"), ("extractor_version", "other-version")):
            changed = ActualObservationInputs(replace(observations[0].snapshot, **{field: other}),
                observations[0].primary, observations[0].wrist, observations[0].proprio)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                window(observations=(changed,) + observations[1:])

    def test_command_steps_are_measured_transitions_not_candidate_times(self):
        value = window()
        for steps in ((47, 48), (46, 48), (46, True), (46,), (46., 47.)):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                replace(value, command_policy_steps=steps)

    def test_wrong_command_shape_dtype_and_nonfinite_data_are_rejected(self):
        value = window()
        for bad in (np.zeros((3, 7)), np.zeros((2, 6)), np.zeros((2, 7), bool),
                    np.full((2, 7), np.nan), np.full((2, 7), np.inf)):
            with self.subTest(shape=bad.shape, dtype=str(bad.dtype)), self.assertRaises(ValueError):
                replace(value, requested=bad)

    def test_declared_command_hashes_are_not_trusted(self):
        value = window()
        for bad in ("bad", "a" * 64, "A" * 64):
            with self.subTest(digest=bad), self.assertRaises(ValueError):
                replace(value, requested_sha256=bad)

    def test_actual_missing_proprio_cannot_be_filled_from_requests(self):
        observations = frames()
        item = observations[0]
        missing = ActualObservationInputs(replace(item.snapshot, proprio_sha256=None), item.primary, item.wrist)
        with self.assertRaisesRegex(ValueError, "proprio history required"):
            window(observations=(missing,) + observations[1:])

    def test_actual_missing_wrist_cannot_be_filled_from_forecast(self):
        observations = frames()
        item = observations[0]
        missing = ActualObservationInputs(replace(item.snapshot, wrist_sha256=None), item.primary, proprio=item.proprio)
        with self.assertRaisesRegex(ValueError, "dual RGB"):
            window(observations=(missing,) + observations[1:])

    def test_measured_proprio_dimension_must_be_literal_nine(self):
        observations = frames()
        bad = make_actual_inputs(pool_id="runtime-pool", entity_key="butter", anchor_key="basket",
            block_index=2, policy_step=46, actual_primary=observations[0].primary,
            actual_wrist=observations[0].wrist, actual_proprio=np.zeros(7))
        with self.assertRaisesRegex(ValueError, "nine-component"):
            window(observations=(bad,) + observations[1:])

    def test_single_snapshot_is_not_padded_into_temporal_evidence(self):
        with self.assertRaisesRegex(ValueError, "two literal actual frames"):
            window(observations=(frames()[-1],))

    def test_block_id_cannot_reverse_or_exceed_recipient(self):
        observations = frames()
        for index in (4,):
            first = observations[0]
            bad = ActualObservationInputs(replace(first.snapshot, block_index=index),
                first.primary, first.wrist, first.proprio)
            with self.assertRaises(ValueError):
                window(observations=(bad,) + observations[1:])
        middle = observations[1]
        bad = ActualObservationInputs(replace(middle.snapshot, block_index=1),
            middle.primary, middle.wrist, middle.proprio)
        with self.assertRaisesRegex(ValueError, "backwards"):
            window(observations=(observations[0], bad, observations[2]))

    def test_old_valid_teacher_window_is_not_recipient_PRE(self):
        value = window()
        # Even a structurally valid observed window for its own endpoint cannot
        # be donated to an earlier root-192 PRE recipient.
        recipient = replace(value.recipient_snapshot, pool_id="root-192-PRE", policy_step=46)
        with self.assertRaises(ValueError):
            value.validate_for(recipient, 46)

    def test_unknown_candidate_or_teacher_arguments_are_not_accepted(self):
        with self.assertRaises(TypeError):
            make_actual_observation_window(recipient_snapshot=frames()[-1].snapshot,
                use_policy_step=48, actual_frames=frames(), actual_requested=np.zeros((2, 7)),
                actual_applied=np.zeros((2, 7)), predicted_primary=np.zeros((8, 8, 3)))


if __name__ == "__main__":
    unittest.main()
