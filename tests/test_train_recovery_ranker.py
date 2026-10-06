"""CPU synthetic pipeline checks; not evidence of real V9 task improvement."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wam_reranking.candidate_utility import CandidateUtilityModel
from wam_reranking.recovery_contract import FEATURE_NAMES, FEATURE_SCHEMA, SCHEMA

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_recovery_ranker.py"
SPEC = importlib.util.spec_from_file_location("recovery_training_pipeline_under_test", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def synthetic_dataset():
    backbone = CandidateUtilityModel(np.zeros(27), np.ones(27), np.zeros(27), 0.)
    names = pipeline.EXPECTED_TARGET_NAMES
    rank, auxiliary = [], []
    slot = FEATURE_NAMES.index("object_shift.grasped.terminal_proposal")
    observed_slot = names.index("target_evidence_available.after")
    transition_slot = names.index("target_evidence_available.transition")
    for split, states in (("train", (0, 1)), ("val", (6, 7))):
        for state in states:
            for candidate_id in (0, 1):
                features = np.zeros(len(FEATURE_NAMES)); features[slot] = float(candidate_id == 0)
                shuffled = features.copy(); shuffled[slot] = 1. - features[slot]
                value = .5 if candidate_id == 0 else .6
                backbone_features = np.zeros(27); backbone_features[0] = value
                rank.append(dict(
                    pool_id=f"synthetic:libero90:task0:state{state}", split=split,
                    candidate_id=candidate_id, task=0, state=state,
                    features=features.tolist(), no_dag_features=np.zeros(len(FEATURE_NAMES)).tolist(),
                    shuffled_features=shuffled.tolist(), backbone_features=backbone_features.tolist(),
                    backbone_score=value, value=value, success=candidate_id == 0,
                    outcome=dict(success=candidate_id == 0, executed_steps=100 + candidate_id,
                                 continuation_wam_calls=5 + candidate_id)))
                targets = np.zeros(len(names)); masks = np.zeros(len(names), dtype=bool)
                targets[observed_slot] = float(candidate_id == 0); masks[observed_slot] = True
                targets[transition_slot] = 1. if candidate_id == 0 else -1.; masks[transition_slot] = True
                auxiliary.append(dict(
                    row_id=f"synthetic:observed_aux:{split}:{state}:{candidate_id}", split=split,
                    identity=dict(dataset="synthetic_not_real_evidence", suite="libero_90", task=0,
                                  state=state, candidate_id=candidate_id, observed_block_id=f"block{state}"),
                    features=features.tolist(), no_dag_features=np.zeros(len(FEATURE_NAMES)).tolist(),
                    shuffled_features=shuffled.tolist(), targets=targets.tolist(), masks=masks.tolist()))
    dataset = dict(feature_schema=FEATURE_SCHEMA, feature_names=list(FEATURE_NAMES),
        target_names=list(names), suite="libero_90", rank_rows=rank, auxiliary_rows=auxiliary,
        training_audit=dict(schema=SCHEMA, recovery_training_ready=True,
            deployment_gt_leakage=False, group_leakage=False,
            expected_rank_rows=8, expected_auxiliary_rows=8,
            role="unit_test_admission_fixture_not_actual_data_audit"))
    return dataset, backbone


class RecoveryTrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.dataset, self.backbone = synthetic_dataset()

    def audit(self, dataset=None):
        return pipeline.audit_training_input(self.dataset if dataset is None else dataset, self.backbone)

    def test_valid_explicit_source_groups_and_schemas_pass_audit(self):
        audit = self.audit()
        self.assertTrue(audit["ready"], audit["problems"])
        self.assertFalse(audit["group_leakage"])
        self.assertEqual(audit["counts"], dict(rank_train=4, rank_val=4, auxiliary_train=4, auxiliary_val=4))

    def test_signed_transition_encoding_copies_labels_and_preserves_mask(self):
        row = dict(targets=[-1., 0., 1., .2], masks=[True] * 4)
        original = deepcopy(row)
        encoded, masks = pipeline.normalize_auxiliary_targets(row,
            ("a.transition", "b.transition", "c.transition", "d.after"))
        np.testing.assert_allclose(encoded, [0., .5, 1., .2])
        np.testing.assert_array_equal(masks, np.ones(4, dtype=bool))
        self.assertEqual(row, original)

    def test_masked_unknown_is_not_force_encoded_as_negative_transition(self):
        encoded, masks = pipeline.normalize_auxiliary_targets(
            dict(targets=[np.nan], masks=[False]), ("a.transition",))
        self.assertEqual(encoded[0], 0.)
        self.assertFalse(masks[0])

    def test_invalid_signed_delta_is_rejected_not_clipped(self):
        with self.assertRaises(ValueError):
            pipeline.normalize_auxiliary_targets(dict(targets=[-1.2], masks=[True]), ("a.transition",))

    def test_masked_control_retains_exactly_raw_command_timing(self):
        features = np.ones(len(FEATURE_NAMES))
        masked = pipeline.mask_attribution_features(features)
        self.assertEqual(masked.sum(), 14.)
        for name, value in zip(FEATURE_NAMES, masked):
            self.assertEqual(value, float(name.startswith("command.")))

    def test_auxiliary_cannot_leak_train_group_into_val(self):
        data = deepcopy(self.dataset)
        data["auxiliary_rows"][4]["identity"]["state"] = 0
        audit = self.audit(data)
        self.assertFalse(audit["ready"])
        self.assertTrue(audit["group_leakage"])

    def test_missing_auxiliary_identity_is_rejected_not_guessed_from_row_id(self):
        data = deepcopy(self.dataset); data["auxiliary_rows"][0].pop("identity")
        audit = self.audit(data)
        self.assertFalse(audit["ready"])
        self.assertTrue(any("task/state" in message for message in audit["problems"]))

    def test_suite_alias_cannot_hide_same_physical_group_leak(self):
        data = deepcopy(self.dataset)
        data["auxiliary_rows"][4]["identity"].update(suite="libero90", state=0)
        self.assertTrue(self.audit(data)["group_leakage"])

    def test_duplicate_candidate_or_auxiliary_identity_is_rejected(self):
        for branch in ("rank_rows", "auxiliary_rows"):
            data = deepcopy(self.dataset)
            if branch == "rank_rows":
                data[branch][1]["candidate_id"] = data[branch][0]["candidate_id"]
            else:
                data[branch][1]["row_id"] = data[branch][0]["row_id"]
            with self.subTest(branch=branch):
                self.assertFalse(self.audit(data)["ready"])

    def test_old_feature_schema_or_wrong_target_schema_is_rejected(self):
        data = deepcopy(self.dataset); data["feature_schema"] = "cause_dependency_recovery_features_v2"
        self.assertFalse(self.audit(data)["ready"])
        data = deepcopy(self.dataset); data["target_names"] = list(reversed(data["target_names"]))
        self.assertFalse(self.audit(data)["ready"])

    def test_nonfinite_or_unfrozen_backbone_feature_score_is_rejected(self):
        data = deepcopy(self.dataset); data["rank_rows"][0]["features"][0] = float("nan")
        self.assertFalse(self.audit(data)["ready"])
        data = deepcopy(self.dataset); data["rank_rows"][0]["backbone_score"] += .001
        self.assertFalse(self.audit(data)["ready"])

    def test_reserved_32_scene_states_are_excluded_before_fit(self):
        data = deepcopy(self.dataset); data["rank_rows"][0]["state"] = 35
        audit = self.audit(data)
        self.assertFalse(audit["ready"])
        self.assertTrue(any("reserved" in message for message in audit["problems"]))

    def test_qc_confirmation_and_false_ready_audit_are_rejected(self):
        data = deepcopy(self.dataset); data["rank_rows"][0]["role"] = "clean_b"
        self.assertFalse(self.audit(data)["ready"])
        data = deepcopy(self.dataset); data["auxiliary_rows"][0]["split"] = "confirmation"
        self.assertFalse(self.audit(data)["ready"])
        data = deepcopy(self.dataset); data["training_audit"]["recovery_training_ready"] = False
        self.assertFalse(self.audit(data)["ready"])

    def test_no_val_row_is_passed_to_either_fitting_branch(self):
        rank, auxiliary = pipeline._training_rows(self.dataset, masked=False)
        self.assertEqual(len(rank), 4); self.assertEqual(len(auxiliary), 4)
        self.assertTrue(all(row["split"] == "train" for row in rank + auxiliary))
        self.assertTrue(all("state6" not in row["pool_id"] for row in rank))
        self.assertEqual(auxiliary[1]["targets"][pipeline.EXPECTED_TARGET_NAMES.index(
            "target_evidence_available.transition")], 0.)

    def test_common_selector_preserves_value_tie_rule_not_argmax_scores(self):
        rows = self.dataset["rank_rows"][:2]
        selected = pipeline._select_precomputed(rows, {0: .7, 1: .695}, 0.)
        self.assertEqual(selected, 1)

    def test_completed_cpu_pipeline_saves_models_hashes_and_reports_without_auto_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, backbone_file, output = folder / "input.json", folder / "backbone.json", folder / "result"
            source.write_text(json.dumps(self.dataset), encoding="utf-8")
            backbone_file.write_text(json.dumps(self.backbone.to_record()), encoding="utf-8")
            before_data = source.read_bytes(); before_backbone = backbone_file.read_bytes()
            report = pipeline.run_pipeline(source, backbone_file, output)
            self.assertEqual(source.read_bytes(), before_data)
            self.assertEqual(backbone_file.read_bytes(), before_backbone)
            self.assertTrue(report["no_auto_start_128"])
            self.assertTrue(report["no_auto_start_main_experiment"])
            self.assertFalse(report["independent_generalization_established"])
            self.assertFalse(report["full_hard_gate_deployment_verified"])
            self.assertIn("not_real_closedloop", report["role"])
            self.assertEqual(report["train"]["pools"], 2)
            self.assertEqual(report["validation"]["pools"], 2)
            self.assertEqual(set(report["validation"]["methods"]), set(pipeline.METHODS))
            protocol = json.loads((output / "training_protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["epochs"], 300); self.assertEqual(protocol["seed"], 0)
            self.assertTrue(protocol["no_validation_tuning"])
            for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                checksum, name = line.split("  ", 1)
                self.assertEqual(checksum, sha256((output / name).read_bytes()).hexdigest())
            full_model = json.loads((output / "recovery_model_v9.json").read_text(encoding="utf-8"))
            self.assertEqual(full_model["training_metadata"]["rank_rows"], 4)
            self.assertEqual(full_model["training_metadata"]["auxiliary_rows"], 4)

    def test_all_failure_pools_remain_denominator_and_missing_costs_are_not_invented(self):
        data = deepcopy(self.dataset)
        for row in data["rank_rows"]:
            if row["state"] == 7:
                row["success"] = False; row["outcome"]["success"] = False
                row["outcome"].pop("executed_steps"); row["outcome"].pop("continuation_wam_calls")
        rank, auxiliary = pipeline._training_rows(data, masked=False)
        model = pipeline.fit_recovery_residual(rank_rows=rank, auxiliary_rows=auxiliary,
            feature_names=FEATURE_NAMES, target_names=pipeline.EXPECTED_TARGET_NAMES, seed=0, epochs=300)
        report = pipeline.evaluate_pools([row for row in data["rank_rows"] if row["split"] == "val"],
                                         self.backbone, model, model.zero_residual_control())
        self.assertEqual(report["pools"], 2)
        self.assertEqual(report["all_failure_pools"], 1)
        self.assertEqual(report["oracle_success_ceiling"], 1)
        for record in report["rows"]:
            if record["state"] == 7:
                self.assertIsNone(record["methods"]["full"]["executed_steps"])
                self.assertIsNone(record["methods"]["full"]["continuation_wam_calls"])

    def test_audit_failure_is_preserved_without_training_or_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); data = deepcopy(self.dataset)
            data["training_audit"]["recovery_training_ready"] = False
            source, backbone_file, output = folder / "input.json", folder / "backbone.json", folder / "result"
            source.write_text(json.dumps(data), encoding="utf-8")
            backbone_file.write_text(json.dumps(self.backbone.to_record()), encoding="utf-8")
            with self.assertRaises(ValueError):
                pipeline.run_pipeline(source, backbone_file, output)
            self.assertTrue((output / "failure.json").is_file())
            self.assertFalse((output / "recovery_model_v9.json").exists())
            with self.assertRaises(RuntimeError):
                pipeline.run_pipeline(source, backbone_file, output)


if __name__ == "__main__":
    unittest.main()
