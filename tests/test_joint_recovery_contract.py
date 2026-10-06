"""Synthetic contract fixtures only: not evidence of trained V10 benefit."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

from wam_reranking import joint_recovery_contract as contract
from wam_reranking.recovery_contract import VISUAL_EVIDENCE_KEYS


def _key(row):
    return tuple(row["identity"][k] for k in contract.IDENTITY_KEYS)


class WholeRecoveryContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.base = Path(self.temporary.name)
        self.serial = 0
        self.reader = contract.ArtifactReader(self.base)
        self.plain = self.artifact({"synthetic_fixture": True})
        self.x = self.make_x()
        self.rows = [self.row(split, cid) for split in ("train", "val") for cid in (0, 1)]
        producer = dict(schema="whole_recovery_input_producer_audit_v1", passed=True,
                        source_content_binding_verified=True, before_attribution_belief_changes_bound=True,
                        deployment_payload_sha256=contract.digest(contract.deployment_payload(self.rows)),
                        need_factory_source_sha256="a" * 64, input_builder_source_sha256="b" * 64,
                        **dict.fromkeys(contract.INPUT_AUDIT_FLAGS, True))
        controls = dict(schema="whole_recovery_control_contract_audit_v1", passed=True,
                        **dict.fromkeys(contract.CONTRACT_TESTS, True))
        control_rows = {mode: contract.control_rows(self.rows, mode) for mode in ("full", "masked", "no_dag", "effect_only")}
        control_rows["shuffled"], substitutions, shuffle_info = contract.shuffled_recovery_need_weights(self.rows)
        controls.update(control_payload_sha256={mode: contract.digest(contract.deployment_payload(rows))
                                                for mode, rows in control_rows.items()},
            shuffled_before_source_audit=dict(passed=True, seed=0, donors_split_local=True,
                donors_cross_group=True, semantic_conflicts_masked_not_normalized_into_physical_needs=True,
                factory_source_sha256="c" * 64, substitution_table_sha256=contract.digest(substitutions)))
        source = {"rank_rows": [{"synthetic_id": i} for i in range(768)],
                  "auxiliary_rows": [{"synthetic_id": i} for i in range(957)]}
        source_ref = self.artifact(source)
        definitions = {}
        for target in contract.TARGETS:
            definitions[target] = self.artifact(dict(schema="observed_recovery_predicate_contract_v1",
                predicate=target, passed=True, observable_definition_passed=True,
                positive_definition_audited=True, negative_definition_independent=True,
                missing_is_masked_not_negative=True, prefix_causal=True,
                old_numeric_thresholds_unchanged=True, entry_verification_safe=True,
                empirical_admission_claimed=False))
        self.bundle = dict(schema=contract.BUNDLE_SCHEMA, input_schema=contract.INPUT_SCHEMA,
            input_producer_audit=self.artifact(producer), control_contract_audit=self.artifact(controls),
            preserved_inherited_rank_rows=768, preserved_inherited_auxiliary_rows=957,
            source_input_sha256_before=source_ref["sha256"], source_input_sha256_after=source_ref["sha256"],
            immutable_source_input=source_ref, preserved_inherited_rows=deepcopy(source),
            controls=control_rows, shuffled_substitution_table=substitutions,
            head_contracts=definitions, candidates=deepcopy(self.rows),
            labels=[self.label(row, row["identity"]["candidate_id"] == 0) for row in self.rows])

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, payload):
        self.serial += 1
        path = self.base / (str(self.serial) + ".json")
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        return {"path": path.name, "sha256": contract.file_sha(path)}

    def source(self, role, ref=None):
        return dict(ref or self.plain, role=role, available_at_boundary=True,
                    relative_step=0, private_teacher=False)

    def array_artifact(self, value):
        self.serial += 1
        path = self.base / (str(self.serial) + ".npy")
        np.save(path, value, allow_pickle=False)
        return dict(path=path.name, sha256=contract.file_sha(path))

    def seal_input_sources(self, x):
        """Create real test data, not a producer-time or training certificate."""
        before, predicted = x["actual_before"], x["candidate_predicted"]
        for name in ("primary_rgb", "wrist_rgb"):
            before[name] = self.source("actual_before", self.array_artifact(np.zeros((3, 3, 3), dtype=np.uint8)))
        for name in ("primary_endpoint_rgb", "wrist_endpoint_rgb"):
            predicted[name] = self.source("candidate_prediction", self.array_artifact(np.zeros((3, 3, 3), dtype=np.uint8)))
        before["proprio_source"] = self.source("actual_before", self.array_artifact(np.asarray(before["proprio"])))
        x["planned_actions_source"] = self.source("planned_command", self.array_artifact(np.asarray(x["planned_actions"])))
        before["quality_source"] = self.source("deployable_before_observer", self.artifact(before["quality"]))
        predicted["visual_source"] = self.source("candidate_prediction", self.artifact(predicted["visual"]))
        binding = {k: v for k, v in x["task_context"].items() if k != "entity_binding_source"}
        x["task_context"]["entity_binding_source"] = self.source("frozen_task_binding", self.artifact(binding))
        return x

    def make_x(self):
        before = {key: self.source("actual_before") for key in
                  ("primary_rgb", "wrist_rgb", "proprio_source")}
        before.update(proprio=[0.] * 9, quality=dict(primary_reliable=True, wrist_reliable=True,
            execution_reliable=True, cross_view_conflict=False, observer_evidence_available=True),
            quality_source=self.source("deployable_before_observer"))
        predicted = {key: self.source("candidate_prediction") for key in
                     ("primary_endpoint_rgb", "wrist_endpoint_rgb", "visual_source")}
        predicted.update(predicted_for_step=16, intermediate_frames_available=False,
            visual={name: dict(value=None, available=False) for name in VISUAL_EVIDENCE_KEYS})
        x = dict(schema=contract.INPUT_SCHEMA, actual_before=before, candidate_predicted=predicted,
            planned_actions=[[0.] * 7 for _ in range(16)], planned_actions_source=self.source("planned_command"),
            task_context=dict(language="Synthetic test only", target="test object", anchor="test anchor",
                              relation="on", entity_binding_source=self.source("frozen_task_binding")))
        return self.seal_input_sources(x)

    def need(self, predicate="grasped", *, deadline=16, kind="next_boundary_goal", cause="object_shift"):
        paths = {"target_pose_current": ["target_pose_current"],
                 "target_reachable": ["target_pose_current", "target_reachable"],
                 "grasped": ["target_pose_current", "target_reachable", "grasped"],
                 "lifted": ["target_pose_current", "target_reachable", "grasped", "lifted"],
                 "place_ready": ["target_pose_current", "target_reachable", "grasped", "lifted", "place_ready"]}
        path = paths.get(predicate, [predicate])
        if cause in ("unknown", "visual_occlusion"):
            path = [predicate]
        return dict(cause=cause, predicate=predicate, deadline=deadline, kind=kind,
            weight=.8, propagated=len(path) > 1, path=path,
            origin="current_attribution_invalidation", old_value="true", old_confidence=.9,
            new_value="unknown", source_block_id="test_source_block")

    def row(self, split, cid):
        return dict(identity=dict(dataset="synthetic_fixture", suite="test_suite", task=46,
            state=10 if split == "train" else 13, candidate_id=cid, observed_block_id="test_block"),
            split=split, pool_id=split + "_test_pool", role="constructed_recovery_auxiliary",
            stage="uncertain", purpose="test_recovery", x=deepcopy(self.x), requirements=[],
            before_lineage=dict(snapshot_source=dict(self.plain, metadata_only=True, private_teacher=True),
                                query_source=dict(self.plain, metadata_only=True, private_teacher=False)),
            goals=[dict(predicate="grasped", deadline=16, confidence=.9, kind="next_boundary_goal")],
            needs=[self.need()], before_verification={p: dict(value="true", deployable_observer_verified=True)
                for p in contract.PREDICATES}, entry_route="verified", backbone_score=.5)

    def label(self, row, value, objective=None):
        objective = objective or row["goals"][0]
        p, t, kind = objective["predicate"], objective["deadline"], objective["kind"]
        cert = dict(schema=contract.AT_USE_LABEL_SCHEMA if kind == "required_use" else contract.GOAL_LABEL_SCHEMA,
            passed=True, identity_bound=True, actual_rgb_observable=True, time_bound=True,
            repeat_noise_verified=True, negative_not_inferred_from_missing=True,
            predicate=p, deadline=t, kind=kind, value=value, identity=deepcopy(row["identity"]))
        return dict(identity=deepcopy(row["identity"]), split=row["split"], pool_id=row["pool_id"],
            predicate=p, deadline=t, kind=kind, mask=True, value=value,
            actual_evidence_role="offline_supervision_only", certificate=self.artifact(cert))

    def audit(self):
        return contract.audit_whole_training(self.bundle, base_path=self.base)

    def rejected(self, text):
        report = self.audit()
        self.assertFalse(report["training_ready"])
        self.assertTrue(any(text in failure for failure in report["failures"]), report)
        self.assertFalse(report["training_started"])

    def probabilities(self):
        return {0: {("grasped", 16, "next_boundary_goal"): .2},
                1: {("grasped", 16, "next_boundary_goal"): .9}}

    def test_complete_synthetic_admission_not_a_fit(self):
        report = self.audit()
        self.assertTrue(report["input_supervision_and_direct_channel_contracts_passed"], report)
        self.assertFalse(report["passed"])
        self.assertFalse(report["complete_pipeline_controls_ready"])
        self.assertIn("complete_attribution_and_belief_control_producer_not_available", report["failures"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["partial_training_allowed"])
        self.assertFalse(report["gate3_or_main_experiment_ready_claimed"])
        self.assertEqual(report["active_heads"], [["grasped", 16, "next_boundary_goal"]])

    def test_missing_second_active_head_blocks_whole_not_partial(self):
        for row in self.bundle["candidates"]:
            row["goals"].append(dict(predicate="lifted", deadline=16, confidence=.75, kind="next_boundary_goal"))
            row["needs"].append(self.need("lifted"))
        self.rejected("whole_run_head_supervision_gap:lifted")

    def test_inactive_semantic_contract_is_still_required(self):
        self.bundle["head_contracts"].pop("place_ready")
        self.rejected("all_seven_predicate")

    def test_semantic_contract_target_cannot_be_reused(self):
        self.bundle["head_contracts"]["lifted"] = self.bundle["head_contracts"]["grasped"]
        self.rejected("semantic_contract_target_mismatch:lifted")

    def test_same_pool_same_time_positive_negative_required(self):
        for label in self.bundle["labels"]:
            row = next(r for r in self.rows if r["identity"] == label["identity"])
            label.update(self.label(row, True))
        self.rejected("whole_run_head_supervision_gap")

    def test_same_time_not_ever_or_endpoint_proxy(self):
        self.bundle["labels"][1]["deadline"] = 5
        self.rejected("whole_run_head_supervision_gap")

    def test_selected_historical_is_not_counterfactual_peer(self):
        for row in self.bundle["candidates"]:
            row["role"] = "historical_selected_auxiliary"
        self.rejected("whole_run_head_supervision_gap")

    def test_validation_positive_negative_cannot_be_waived(self):
        self.bundle["labels"] = self.bundle["labels"][:2]
        self.rejected("whole_run_head_supervision_gap")

    def test_masked_is_not_negative(self):
        self.bundle["labels"][1].update(mask=False, value=False)
        self.rejected("whole_run_head_supervision_gap")

    def test_duplicate_label_cannot_inflate_counts(self):
        self.bundle["labels"].append(deepcopy(self.bundle["labels"][0]))
        self.rejected("duplicate six-field identity/target/time/kind Y")

    def test_contradictory_duplicate_label_also_rejected(self):
        row = self.bundle["candidates"][0]
        self.bundle["labels"].append(self.label(row, False))
        self.rejected("duplicate six-field identity/target/time/kind Y")

    def test_certificate_other_candidate_is_rejected(self):
        self.bundle["labels"][0]["certificate"] = self.bundle["labels"][1]["certificate"]
        self.rejected("certificate target/time/value/source mismatch")

    def test_future_actual_input_is_rejected(self):
        self.bundle["candidates"][0]["x"]["candidate_predicted"]["visual_source"]["role"] = "actual_execution"
        self.rejected("preexecution-prediction X")

    def test_private_teacher_input_is_rejected(self):
        self.bundle["candidates"][0]["x"]["actual_before"]["quality_source"]["private_teacher"] = True
        self.rejected("preexecution-prediction X")

    def test_future_input_step_is_rejected(self):
        self.bundle["candidates"][0]["x"]["actual_before"]["primary_rgb"]["relative_step"] = 16
        self.rejected("preexecution-prediction X")

    def test_terminal_success_inside_x_is_rejected(self):
        self.bundle["candidates"][0]["x"]["terminal_success"] = True
        self.rejected("candidate X exact fields")

    def test_old_604_cannot_be_renamed_raw_b(self):
        self.bundle["candidates"][0]["x"] = {"schema": contract.INPUT_SCHEMA, "features": [0.] * 604}
        self.rejected("candidate X exact fields")

    def test_unavailable_visual_is_not_zero_negative(self):
        first = next(iter(self.bundle["candidates"][0]["x"]["candidate_predicted"]["visual"].values()))
        first["value"] = 0.
        self.rejected("unavailable predicted field must remain None")

    def test_nonfinite_or_bool_proprio_rejected(self):
        for bad in (float("nan"), True):
            row = deepcopy(self.rows[0])
            row["x"]["actual_before"]["proprio"][0] = bad
            with self.assertRaisesRegex(ValueError, "finite literal number"):
                contract.validate_candidate(row, self.reader)

    def test_no_hash_only_provenance_self_certification(self):
        self.bundle.pop("input_producer_audit")
        self.rejected("artifact_audit")

    def test_required_control_audit_cannot_be_waived(self):
        self.bundle.pop("control_contract_audit")
        self.rejected("artifact_audit")

    def test_control_flags_without_actual_control_rows_are_insufficient(self):
        self.bundle.pop("controls")
        self.rejected("control_payload_binding")

    def test_control_cannot_delete_ordinary_predicted_visual_information(self):
        self.bundle["controls"]["masked"][0]["x"]["candidate_predicted"]["visual"] = {}
        self.rejected("control_payload_binding")

    def test_no_dag_cannot_keep_propagated_needs(self):
        self.bundle["controls"]["no_dag"][0]["needs"][0]["weight"] = .8
        self.rejected("unexpected need-channel change")

    def test_shuffled_control_requires_before_only_donor_lineage(self):
        self.bundle.pop("shuffled_substitution_table")
        self.rejected("actual split-local weight shuffle or donor table")

    def test_real_shuffle_is_group_deranged_split_local_and_never_reads_y(self):
        rows = deepcopy(self.rows)
        for row in self.rows:
            added = deepcopy(row)
            added["identity"]["state"] += 20
            added["pool_id"] += "_second_group"
            added["needs"][0]["weight"] = .25
            added["terminal_success"] = object()  # The shuffle must not inspect Y.
            rows.append(added)
        shuffled, substitutions, audit = contract.shuffled_recovery_need_weights(rows)
        self.assertTrue(contract.assert_equal_control_information(rows, shuffled))
        for source, changed in zip(rows, shuffled):
            self.assertEqual({k: v for k, v in source["needs"][0].items() if k != "weight"},
                             {k: v for k, v in changed["needs"][0].items() if k != "weight"})
            self.assertNotEqual(source["needs"][0]["weight"], changed["needs"][0]["weight"])
        for item in substitutions:
            self.assertNotEqual(item["receiver_group"], item["donor_group"])
            donor = next(r for r in rows if r["identity"] == item["donor_identity"])
            self.assertEqual(donor["split"], item["split"])
            self.assertFalse(item["label_or_terminal_outcome_read"])
        self.assertTrue(audit["does_not_claim_complete_attribution_removal_or_rerun"])

    def test_shuffle_lone_group_or_missing_same_deadline_is_zero_not_new_cause(self):
        shuffled, table, _ = contract.shuffled_recovery_need_weights(self.rows)
        self.assertTrue(all(n["weight"] == 0. for r in shuffled for n in r["needs"]))
        self.assertTrue(all(i["reason"] == "single_group_no_derangement" for i in table))
        rows = deepcopy(self.rows[:2])
        added = deepcopy(rows[0])
        added["identity"]["state"] += 20
        added["needs"][0]["deadline"] = 5
        rows.append(added)
        shuffled, table, _ = contract.shuffled_recovery_need_weights(rows)
        self.assertTrue(all(n["weight"] == 0. for r in shuffled for n in r["needs"]))
        self.assertTrue(all(i["reason"] == "no_same_time_objective_in_donor" for i in table))

    def test_shuffle_is_deterministic_even_with_different_row_order(self):
        rows = deepcopy(self.rows)
        for row in self.rows:
            added = deepcopy(row)
            added["identity"]["state"] += 20
            added["needs"][0]["weight"] = .25
            rows.append(added)
        a, _, _ = contract.shuffled_recovery_need_weights(rows)
        b, _, _ = contract.shuffled_recovery_need_weights(list(reversed(rows)))
        self.assertEqual({_key(r): r["needs"] for r in a}, {_key(r): r["needs"] for r in b})

    def test_shuffle_donor_total_is_not_repeated_for_each_cause(self):
        rows = deepcopy(self.rows[:2])
        for row in rows:
            row["needs"][0]["weight"] = .1
            execution = self.need(cause="execution_contact_deviation")
            execution.update(weight=.1, path=["execution_consistent", "grasped"], propagated=True)
            row["needs"].append(execution)
        donor = deepcopy(self.rows[0])
        donor["identity"]["state"] += 20
        donor["pool_id"] += "_donor"
        donor["needs"][0]["weight"] = .4
        rows.append(donor)
        result, table, _ = contract.shuffled_recovery_need_weights(rows)
        for row in result[:2]:
            self.assertAlmostEqual(sum(n["weight"] for n in row["needs"]), .4)
            self.assertEqual([n["weight"] for n in row["needs"]], [.2, .2])
        self.assertEqual(table[0]["receiver_cause_share"], .5)
        self.assertEqual(table[0]["donor_objective_total"], .4)

    def test_source_hash_tamper_rejected(self):
        (self.base / self.bundle["immutable_source_input"]["path"]).write_text("{}", encoding="utf-8")
        self.rejected("immutable_source_membership")

    def test_inherited_row_content_not_just_count(self):
        self.bundle["preserved_inherited_rows"]["rank_rows"][1]["synthetic_id"] = 999
        self.rejected("original row identity/order/content not preserved")

    def test_pool_requires_same_snapshot_and_query(self):
        self.bundle["candidates"][1]["before_lineage"]["snapshot_source"].update(self.artifact({"different": True}))
        self.rejected("candidate pool does not share actual before state")

    def test_private_snapshot_lineage_is_not_deployment_input(self):
        self.bundle["candidates"][0]["before_lineage"]["snapshot_source"]["private_teacher"] = False
        self.rejected("runtime/query provenance is metadata-only")

    def test_train_val_task_state_leak_rejected(self):
        self.bundle["candidates"][2]["identity"]["state"] = 10
        self.bundle["candidates"][2]["identity"]["observed_block_id"] = "different_test_block"
        self.rejected("task/state train/val leakage")

    def test_pool_string_cannot_join_different_source_tasks(self):
        self.bundle["candidates"][1]["identity"]["task"] = 57
        self.rejected("pool string aliases different")

    def test_old_producer_audit_cannot_certify_changed_inline_input(self):
        self.bundle["candidates"][0]["x"]["actual_before"]["proprio"][0] = .25
        self.rejected("producer_payload_binding")

    def test_six_field_duplicate_identity_rejected(self):
        self.bundle["candidates"].append(deepcopy(self.bundle["candidates"][0]))
        self.rejected("duplicate six-field source identity")

    def test_constructed_probe_cannot_be_native_rank_outcome(self):
        self.bundle["candidates"][0]["terminal_success"] = True
        self.rejected("invented native terminal outcomes")

    def test_initial_unknown_is_not_new_attribution_deficit(self):
        self.bundle["candidates"][0]["needs"][0]["old_value"] = "unknown"
        self.rejected("initial UNKNOWN/FALSE")

    def test_true_dag_path_is_required(self):
        self.bundle["candidates"][0]["needs"][0]["path"] = ["target_pose_current", "grasped"]
        self.rejected("every legal DAG edge")

    def test_unknown_cannot_guess_physical_missing_state(self):
        self.bundle["candidates"][0]["needs"][0]["cause"] = "unknown"
        self.rejected("cannot guess physical missing state")

    def test_normal_cannot_gain_recovery_score(self):
        self.bundle["candidates"][0]["needs"][0]["cause"] = "normal"
        self.rejected("normal cannot invent caused recovery needs")
        rows = deepcopy(self.rows[:2])
        for row in rows:
            row["needs"][0].update(cause="normal", weight=0.)
        self.assertEqual(contract.score_pool(rows, self.probabilities(), mode="full",
            reference_candidate_id=0)[1]["recovery_residual"], 0.)

    def test_goal_requires_before_owned_need(self):
        self.bundle["candidates"][0]["needs"] = []
        self.rejected("goals must be source-owned before needs")

    def test_deadline_zero_keeps_safe_fallback_despite_future_goal(self):
        row = deepcopy(self.rows[0])
        row["stage"] = "transport"
        row["requirements"] = contract.expected_requirements("transport", row["x"]["planned_actions"], "on")
        row["before_verification"]["grasped"]["value"] = "unknown"
        with self.assertRaisesRegex(ValueError, "uncertified entry facts"):
            contract.validate_candidate(row, self.reader)
        row["entry_route"] = "safe_fallback"
        contract.validate_candidate(row, self.reader)
        second = deepcopy(row)
        second["identity"]["candidate_id"] = 1
        result = contract.score_pool([row, second], self.probabilities(), mode="full", reference_candidate_id=0)[1]
        self.assertGreater(result["recovery_residual"], 0.)
        self.assertFalse(result["candidate_execution_allowed"])
        self.assertTrue(result["goals_never_override_safe_fallback"])
        self.assertEqual([t["credit"] for t in result["terms"][:2]], [0., 0.])

    def test_original_use_deadline_cannot_be_shifted(self):
        row = deepcopy(self.rows[0])
        row["stage"] = "grasp"
        row["x"]["planned_actions"][4][6] = 1.
        self.seal_input_sources(row["x"])
        row["requirements"] = contract.expected_requirements("grasp", row["x"]["planned_actions"], "on")
        self.assertEqual([r["deadline"] for r in row["requirements"]], [4, 4])
        row["requirements"][0]["deadline"] = 16
        with self.assertRaisesRegex(ValueError, "use-time table was changed"):
            contract.validate_candidate(row, self.reader)

    def test_positive_close_negative_open_are_not_physical_labels(self):
        actions = deepcopy(self.x["planned_actions"])
        actions[4][6], actions[12][6], actions[6][2] = 1., -1., .1
        self.assertEqual(contract.expected_requirements("grasp", actions, "on")[0]["deadline"], 4)
        self.assertEqual(contract.expected_requirements("lift", actions, "on")[0]["deadline"], 6)
        self.assertEqual(contract.expected_requirements("place", actions, "on")[-1]["deadline"], 12)

    def test_articulated_requirements_do_not_lift_whole_drawer(self):
        actions = deepcopy(self.x["planned_actions"])
        actions[6][2] = .1
        for relation in ("open", "closed", "articulated"):
            for stage, deadline in (("place", 0), ("lift", 6)):
                requirements = contract.expected_requirements(stage, actions, relation)
                self.assertEqual([r["predicate"] for r in requirements], ["target_visible", "target_pose_current"])
                self.assertEqual([r["deadline"] for r in requirements], [deadline, deadline])

    def test_observer_goal_requires_explicit_information_plan(self):
        row = deepcopy(self.rows[0])
        row.update(stage="observe", purpose="information_acquisition")
        row["x"]["actual_before"]["quality"]["observer_evidence_available"] = False
        self.seal_input_sources(row["x"])
        row["goals"] = [dict(predicate=contract.OBSERVER, deadline=16, confidence=.8, kind="epistemic_goal")]
        need = self.need(contract.OBSERVER, kind="epistemic_goal", cause="unknown")
        need["origin"] = "before_epistemic_insufficiency"
        row["needs"] = [need]
        contract.validate_candidate(row, self.reader)
        row["x"]["actual_before"]["quality"]["observer_evidence_available"] = True
        self.seal_input_sources(row["x"])
        with self.assertRaisesRegex(ValueError, "insufficient before evidence"):
            contract.validate_candidate(row, self.reader)

    def test_secondary_observer_goal_never_changes_lift_entry_gate(self):
        row = deepcopy(self.rows[0])
        row.update(stage="lift", purpose="task_with_observation_recovery")
        row["requirements"] = contract.expected_requirements("lift", row["x"]["planned_actions"], "on")
        row["before_verification"]["grasped"]["value"] = "unknown"
        row["x"]["actual_before"]["quality"]["observer_evidence_available"] = False
        self.seal_input_sources(row["x"])
        row["goals"] = [dict(predicate=contract.OBSERVER, deadline=16, confidence=.8, kind="epistemic_goal")]
        need = self.need(contract.OBSERVER, kind="epistemic_goal", cause="unknown")
        need["origin"] = "before_epistemic_insufficiency"
        row["needs"] = [need]
        with self.assertRaisesRegex(ValueError, "uncertified entry facts"):
            contract.validate_candidate(row, self.reader)
        row["entry_route"] = "safe_fallback"
        contract.validate_candidate(row, self.reader)
        self.assertEqual(row["requirements"][0]["deadline"], 0)
        self.assertEqual(row["goals"][0]["deadline"], 16)

    def test_task_action_cannot_silently_become_information_goal(self):
        row = deepcopy(self.rows[0])
        row.update(stage="lift", purpose="task_action")
        row["requirements"] = contract.expected_requirements("lift", row["x"]["planned_actions"], "on")
        row["x"]["actual_before"]["quality"]["observer_evidence_available"] = False
        self.seal_input_sources(row["x"])
        row["goals"] = [dict(predicate=contract.OBSERVER, deadline=16, confidence=.8, kind="epistemic_goal")]
        need = self.need(contract.OBSERVER, kind="epistemic_goal", cause="unknown")
        need["origin"] = "before_epistemic_insufficiency"
        row["needs"] = [need]
        with self.assertRaisesRegex(ValueError, "explicit primary/secondary information"):
            contract.validate_candidate(row, self.reader)

    def test_no_dag_only_removes_propagated_credit(self):
        rows = deepcopy(self.rows[:2])
        result = contract.score_pool(rows, self.probabilities(), mode="no_dag", reference_candidate_id=0)
        self.assertEqual(result[1]["recovery_residual"], 0.)
        for row in rows:
            row["goals"] = [dict(predicate="target_pose_current", deadline=16, confidence=.6, kind="next_boundary_goal")]
            row["needs"] = [self.need("target_pose_current")]
        probs = {0: {("target_pose_current", 16, "next_boundary_goal"): .2},
                 1: {("target_pose_current", 16, "next_boundary_goal"): .9}}
        self.assertGreater(contract.score_pool(rows, probs, mode="no_dag", reference_candidate_id=0)[1]["recovery_residual"], 0.)

    def test_all_controls_share_raw_input_goals_gates_and_backbone(self):
        variants = [contract.control_rows(self.rows, mode) for mode in ("full", "masked", "no_dag", "effect_only")]
        self.assertTrue(contract.assert_equal_control_information(self.rows, *variants))
        variants[1][0]["x"]["candidate_predicted"]["visual"] = {}
        with self.assertRaisesRegex(ValueError, "ordinary input/objectives"):
            contract.assert_equal_control_information(self.rows, *variants)

    def test_shuffling_cannot_silently_change_candidate_inputs(self):
        with self.assertRaisesRegex(ValueError, "independently audited"):
            contract.control_rows(self.rows, "shuffled")
        shuffled = deepcopy(self.rows)
        shuffled[0]["requirements"].append(dict(predicate="grasped", deadline=1, confidence=.8, kind="required_use"))
        with self.assertRaisesRegex(ValueError, "ordinary input/objectives"):
            contract.assert_equal_control_information(self.rows, shuffled)

    def test_synthetic_scores_directly_use_needs_without_training_claim(self):
        rows = self.rows[:2]
        full = contract.score_pool(rows, self.probabilities(), mode="full", reference_candidate_id=0)
        masked = contract.score_pool(rows, self.probabilities(), mode="masked", reference_candidate_id=0)
        effect = contract.score_pool(rows, self.probabilities(), mode="effect_only", reference_candidate_id=0)
        self.assertEqual(full[0]["recovery_residual"], 0.)
        self.assertGreater(full[1]["recovery_residual"], 0.)
        self.assertEqual(masked[1]["recovery_residual"], 0.)
        self.assertGreater(effect[1]["recovery_residual"], full[1]["recovery_residual"])
        self.assertLessEqual(abs(full[1]["recovery_residual"]), contract.RESIDUAL_CAP)
        self.assertTrue(full[1]["hard_gate_unchanged"])

    def test_reference_needs_all_model_keys_even_if_its_plan_differs(self):
        probabilities = self.probabilities()
        probabilities[0] = {}
        with self.assertRaises(KeyError):
            contract.score_pool(self.rows[:2], probabilities, mode="full", reference_candidate_id=0)

    def test_effect_only_cannot_activate_unsupervised_head(self):
        rows = deepcopy(self.rows[:2])
        for row in rows:
            row["goals"].append(dict(predicate="lifted", deadline=16, confidence=.9, kind="next_boundary_goal"))
        shared = {("grasped", 16, "next_boundary_goal")}
        for mode in ("full", "masked", "shuffled", "no_dag", "effect_only"):
            result = contract.score_pool(rows, self.probabilities(), mode=mode, reference_candidate_id=0,
                                         admitted_keys=shared)[1]
            self.assertEqual(result["terms"][1]["credit"], 0.)
            self.assertEqual(result["terms"][1]["inactive_reason"], "head_not_empirically_admitted_in_shared_scope")
            self.assertTrue(result["explicit_shared_admission_scope"])

    def test_shared_admission_scope_stays_same_when_need_masked(self):
        rows = contract.control_rows(self.rows[:2], "masked")
        key = ("grasped", 16, "next_boundary_goal")
        result = contract.score_pool(rows, self.probabilities(), mode="effect_only", reference_candidate_id=0,
                                     admitted_keys={key})[1]
        self.assertGreater(result["recovery_residual"], 0.)
        masked = contract.score_pool(rows, self.probabilities(), mode="masked", reference_candidate_id=0,
                                     admitted_keys={key})[1]
        self.assertEqual(masked["recovery_residual"], 0.)

    def test_zero_head_fit_is_forbidden(self):
        for row in self.bundle["candidates"]:
            row["goals"], row["needs"] = [], []
        self.bundle["labels"] = []
        self.rejected("zero_head_fit_forbidden")

    def test_zero_confidence_goal_is_not_genuinely_active(self):
        for row in self.bundle["candidates"]:
            row["goals"][0]["confidence"] = 0.
        self.rejected("zero_head_fit_forbidden")

    def test_malformed_label_container_fail_closed(self):
        self.bundle["labels"] = None
        self.rejected("explicit_label_list_required")

    def test_cli_scoped_loader_does_not_fall_back_to_old_package(self):
        from scripts.audit_joint_recovery_training import load_gate
        with self.assertRaisesRegex(ValueError, "Scoped whole-run contract module missing"):
            load_gate(self.base)
        module, actual_sha = load_gate(Path(__file__).resolve().parents[1])
        self.assertIs(module, contract)
        self.assertEqual(actual_sha, contract.file_sha(contract.__file__))
        with self.assertRaisesRegex(ValueError, "source hash differs"):
            load_gate(Path(__file__).resolve().parents[1], "a" * 64)

    def test_cli_writes_new_audit_only_and_does_not_mutate_input(self):
        from scripts.audit_joint_recovery_training import main
        source = self.base / "input.json"
        source.write_text(json.dumps(self.bundle), encoding="utf-8")
        before = contract.file_sha(source)
        output = self.base / "new_audit.json"
        argv = ["audit_joint_recovery_training.py", "--input", str(source), "--output", str(output),
                "--library-root", str(Path(__file__).resolve().parents[1]),
                "--expected-module-sha256", contract.file_sha(contract.__file__)]
        with patch("sys.argv", argv), patch("builtins.print"):
            main()
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(before, contract.file_sha(source))
        self.assertTrue(report["input_supervision_and_direct_channel_contracts_passed"])
        self.assertFalse(report["training_ready"])
        self.assertFalse(report["training_started"])
        self.assertEqual(report["source_bundle_file_sha256"], before)
        with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "no overwrite"):
            main()

    def observer_certificate(self, value):
        from wam_reranking import observed_recovery
        row = self.rows[0]
        code = dict(path=str(Path(observed_recovery.__file__).resolve()), sha256=contract.file_sha(observed_recovery.__file__))
        model = self.artifact({"synthetic_model_fixture_not_fitted": True})
        maps = np.zeros((2, 8, 8), dtype=np.float64)
        if value:
            maps[:, 2, 3] = 1.
        record = dict(schema="actual_after_frozen_deployment_observer_quality_v1", role="offline_supervision_only",
            identity=deepcopy(row["identity"]), step=16,
            quality=dict(primary_reliable=True, wrist_reliable=True, execution_reliable=True,
                         cross_view_conflict=False, observer_evidence_available=value),
            observer_code_source=code, observer_model_source=model,
            subject_maps_source=self.array_artifact(maps),
            actual_after_rgb_sources={view: {k: self.x["candidate_predicted"][view + "_endpoint_rgb"][k] for k in ("path", "sha256")}
                                      for view in ("primary", "wrist")})
        binding = {k: record[k] for k in ("identity", "step", "actual_after_rgb_sources", "subject_maps_source", "observer_code_source", "observer_model_source")}
        record["observer_input_lineage_sha256"] = contract.digest(binding)
        certificate = dict(actual_rgb_measurement_verified=True, no_physical_absence_claim=True,
            after_quality_source=self.artifact(record), actual_rgb_observable=value)
        label = dict(identity=deepcopy(row["identity"]), value=value)
        definition = dict(frozen_observer_code_sha256=code["sha256"], frozen_observer_model_sha256=model["sha256"])
        return certificate, label, definition, record

    def test_observer_algorithm_negative_is_not_physical_absence(self):
        cert, label, definition, _ = self.observer_certificate(False)
        self.assertFalse(cert["actual_rgb_observable"])
        self.assertTrue(contract.validate_observer_quality_certificate(cert, label, self.reader, definition))

    def test_observer_positive_requires_quality_and_independent_actual_rgb_cert(self):
        cert, label, definition, _ = self.observer_certificate(True)
        self.assertTrue(contract.validate_observer_quality_certificate(cert, label, self.reader, definition))
        cert["actual_rgb_observable"] = False
        with self.assertRaisesRegex(ValueError, "positive observer quality still requires"):
            contract.validate_observer_quality_certificate(cert, label, self.reader, definition)

    def test_observer_quality_flag_cannot_override_real_map_output(self):
        cert, label, definition, record = self.observer_certificate(False)
        record["quality"]["observer_evidence_available"] = True
        label["value"] = True
        cert["after_quality_source"] = self.artifact(record)
        with self.assertRaisesRegex(ValueError, "differs from actual frozen observer evaluation"):
            contract.validate_observer_quality_certificate(cert, label, self.reader, definition)

    def test_observer_after_cannot_be_future_deployment_input_or_other_candidate(self):
        cert, label, definition, record = self.observer_certificate(False)
        for change in (dict(role="deployment_input"), dict(step=15), dict(identity=self.rows[1]["identity"])):
            changed = dict(record, **change)
            cert["after_quality_source"] = self.artifact(changed)
            with self.assertRaisesRegex(ValueError, "own-candidate after step16 as Y"):
                contract.validate_observer_quality_certificate(cert, label, self.reader, definition)

    def test_observer_model_source_must_be_frozen_in_semantic_contract(self):
        cert, label, definition, _ = self.observer_certificate(False)
        definition["frozen_observer_model_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from independently frozen"):
            contract.validate_observer_quality_certificate(cert, label, self.reader, definition)

    def test_observer_after_quality_is_read_from_actual_hashed_file(self):
        cert, label, definition, _ = self.observer_certificate(False)
        (self.base / cert["after_quality_source"]["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact missing or changed"):
            contract.validate_observer_quality_certificate(cert, label, self.reader, definition)

    def test_physical_negative_cannot_borrow_observer_quality_false(self):
        label = self.bundle["labels"][1]
        cert = json.loads((self.base / label["certificate"]["path"]).read_text(encoding="utf-8"))
        cert.update(actual_rgb_observable=False, actual_rgb_measurement_verified=True, no_physical_absence_claim=True)
        label["certificate"] = self.artifact(cert)
        self.rejected("artifact_audit:independent audit flags/schema incomplete")


if __name__ == "__main__":
    unittest.main()
