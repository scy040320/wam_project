"""Synthetic immutable-overlay audits; no model fitting or physical claims.

Fixtures mirror the inspected v1/v2 sidecar formats. Nonzero certified atoms
are TEST DATA, not approval of a real predicate or evidence of task benefit.
"""
from __future__ import annotations

from collections import Counter
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wam_reranking.recovery_contract import FEATURE_NAMES, FEATURE_SCHEMA, ROUTES, VISUAL_EVIDENCE_KEYS
from wam_reranking.recovery_observability_contract import ATOM_NAMES

_SPEC = importlib.util.spec_from_file_location("prepare_observed_event_overlay_under_test",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_observed_event_overlay.py")
overlay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(overlay)


def packed(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def frozen_input():
    features = np.zeros(len(FEATURE_NAMES))
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for key in VISUAL_EVIDENCE_KEYS:
        features[index[f"{ROUTES[0]}.visual.{key}.value"]] = .3
        features[index[f"{ROUTES[0]}.visual.{key}.available"]] = 1.
    features = features.tolist()
    targets, masks = [0.] * len(overlay.TARGET_NAMES), [False] * len(overlay.TARGET_NAMES)
    arrays = dict(features=features, no_dag_features=features, shuffled_features=features)
    rank = []
    for task in (0, 9, 20, 44, 46, 57):
        for state in range(8):
            for condition in range(4):
                for cid in range(4):
                    rank.append(dict(pool_id=f"rank/{task}/{state}/{condition}", suite="libero90",
                        task=task, state=state, candidate_id=cid, split="train" if state < 6 else "val",
                        backbone_score=.5, value=.5, success=bool(cid % 2), outcome={}, **arrays))
    auxiliary = []
    for n in range(797):
        split = "train" if n < 638 else "val"
        identity = dict(dataset="frozen_history", suite="libero90", task=0,
            state=31 if split == "train" else 34, candidate_id=1, observed_block_id=f"history/block_{n}")
        auxiliary.append(dict(row_id=f"history_{n}", identity=identity, split=split,
            role="historical_selected_arm_auxiliary", targets=targets, masks=masks, **arrays))
    for task in (0, 9, 46, 57):
        for state in (31, 34):
            for condition in ("normal", "visual_occlusion", "object_shift", "execution_contact_deviation", "unknown"):
                pool = f"task{task}_state{state}_{condition}"
                for cid in range(4):
                    identity = dict(dataset="frozen_paired", suite="libero90", task=task,
                        state=state, candidate_id=cid, observed_block_id=pool + "/first_block")
                    auxiliary.append(dict(row_id=f"paired/{pool}/{cid}", identity=identity,
                        split="train" if state == 31 else "val", role="paired_recovery_supervision",
                        targets=targets, masks=masks, **arrays))
    return dict(schema="recovery_ranker_training_input_v1", feature_schema=FEATURE_SCHEMA,
        feature_names=list(FEATURE_NAMES), target_names=list(overlay.TARGET_NAMES), rank_rows=rank,
        auxiliary_rows=auxiliary, training_audit=dict(recovery_training_ready=True))


class Sources:
    def __init__(self, root, input_bytes, full=False, version=1):
        self.root, self.version = root, version
        self.input = root / "input" / "training_input.json"
        self.input.parent.mkdir()
        self.input.write_bytes(input_bytes)
        self.labels = root / "labels"
        self.labels.mkdir()
        self.output = root / "prepared"
        self.records, self.sidecars = [], {}
        rows = json.loads(input_bytes)["auxiliary_rows"]
        for row in rows:
            if row["role"] != "paired_recovery_supervision":
                continue
            if not full and not row["identity"]["observed_block_id"].endswith("_object_shift/first_block"):
                continue
            identity = row["identity"]
            pool = identity["observed_block_id"].removesuffix("/first_block")
            cid = identity["candidate_id"]
            sidecar_name = f"certificates/{pool}/candidate_{cid}/observability_certificates.json"
            record = dict(identity=identity, pool_id=pool, candidate_id=cid, split=row["split"], sidecar=sidecar_name)
            frames = []
            for step in range(17):
                atoms = {}
                for name in ATOM_NAMES:
                    mask = (name == "right_finger_target_contact" and step == 4 and cid in (0, 1)) or (
                        name == "carried_sufficient_evidence" and step == 6 and cid == 0)
                    value = bool(cid == 0) if mask else None
                    atoms[name] = dict(schema=f"recovery_observability_certificate_v{version}",
                        identity=identity, name=name, step=step, physical_value=value, measurement_valid=mask,
                        new_mask=mask, reasons=[] if mask else ["synthetic_no_evidence"],
                        role="offline_supervision_only", deployment_allowed=False,
                        future_actual_frames_used_only_as_labels=True)
                frames.append(dict(step=step, atoms=atoms))
            self.records.append(record)
            self.sidecars[sidecar_name] = dict(schema=f"observable_recovery_labels_sidecars_v{version}",
                identity=identity, split=row["split"], role="offline_supervision_only", labels=frames,
                future_information_labels_only=True, old_masks_and_values_unchanged=True,
                deployment_features_created=False, cached_frame0_diagnostic_only=True, training_ready=False)
        self.flush()

    def flush(self):
        counts = {name: Counter() for name in ATOM_NAMES}
        for record in self.records:
            sidecar = self.sidecars[record["sidecar"]]
            path = self.labels / record["sidecar"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(packed(sidecar))
            record["sidecar_sha256"] = overlay.digest(path)
            for frame in sidecar["labels"]:
                for name, atom in frame["atoms"].items():
                    if name not in counts:
                        continue
                    field = ("certified_positive_frames" if atom.get("physical_value") else
                             "certified_negative_frames") if atom.get("new_mask") else "masked_frames"
                    counts[name][field] += 1
        audit = dict(schema=f"observable_recovery_labels_sidecars_v{self.version}", passed=True,
            integrity_passed=True, no_original_masks_or_values_changed=True,
            train_val_group_leakage=0, new_queries=0, new_candidates=0,
            completed_candidates=len(self.records), pools=len({r["pool_id"] for r in self.records}),
            split_candidates=dict(Counter(r["split"] for r in self.records)), training_ready=False,
            physical_vs_certified_counts={name:{field: counts[name][field] for field in (
                "certified_positive_frames", "certified_negative_frames", "masked_frames")} for name in ATOM_NAMES})
        (self.labels / "observability_label_audit.json").write_bytes(packed(audit))
        (self.labels / "records.json").write_bytes(packed(self.records))
        self.rehash()

    def rehash(self):
        paths = sorted(self.labels.rglob("*.json"))
        (self.labels / "SHA256SUMS.txt").write_text("".join(
            f"{overlay.digest(p)}  {p.relative_to(self.labels).as_posix()}\n" for p in paths), encoding="utf-8")

    def input_update(self, mutate):
        data = overlay.read(self.input)
        mutate(data)
        self.input.write_bytes(packed(data))

    def atom(self, record=0, step=4, name="right_finger_target_contact"):
        return self.sidecars[self.records[record]["sidecar"]]["labels"][step]["atoms"][name]

    def prepare(self):
        return overlay.prepare(self.input, self.labels, self.output)


class ObservedEventOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_bytes = packed(frozen_input())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="observed_overlay_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, **kwargs):
        return Sources(self.root, self.input_bytes, **kwargs)

    def assert_rejected(self, source, message=None):
        with self.assertRaisesRegex(ValueError, message or ".+"):
            source.prepare()
        self.assertFalse((source.output / "input_audit.json").exists())

    def test_pilot_keeps_all_old_rows_exactly_and_does_not_promote_ready(self):
        source = self.source()
        before = overlay.digest(source.input)
        report = source.prepare()
        inherited = overlay.read(source.output / "inherited_training_rows.json")
        original = overlay.read(source.input)
        for name in ("rank_rows", "auxiliary_rows", "feature_names", "target_names"):
            self.assertEqual(original[name], inherited[name])
        self.assertEqual((768, 957, 32, 544, 128), (report["inherited_rank_rows"],
            report["inherited_auxiliary_rows"], report["overlay_candidates"],
            report["overlay_frames"], report["paired_candidates_without_overlay"]))
        self.assertEqual(before, overlay.digest(source.input))
        self.assertEqual(before, report["original_input_sha256_before"])
        self.assertEqual(before, report["original_input_sha256_after"])
        self.assertFalse(inherited["training_ready"])
        self.assertFalse(report["training_ready"])
        self.assertTrue(original["training_audit"]["recovery_training_ready"])

    def test_full_overlay_retains_160_40_pools_and_80_80_split(self):
        report = self.source(full=True).prepare()
        self.assertEqual(160, report["overlay_candidates"])
        self.assertEqual({"train":80, "val":80}, report["split"])
        self.assertTrue(report["full_160_overlay_complete"])
        self.assertEqual(0, report["paired_candidates_without_overlay"])

    def test_parent_gt_free_claim_is_inherited_not_independently_certified(self):
        source = self.source()
        source.input_update(lambda d: d["training_audit"].update(deployment_gt_leakage=False,
            source_hashes={"/immutable/source": "a" * 64}))
        report = source.prepare()
        parent = overlay.read(source.input)["training_audit"]
        expected = sha256(json.dumps(parent, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        self.assertIs(report["inherited_source_producer_gt_leakage_claim"], False)
        self.assertIs(report["new_overlay_execution_x_leakage"], False)
        self.assertIsNone(report["future_execution_x_leakage"])
        self.assertFalse(report["original_604_feature_gt_freedom_independently_confirmed"])
        self.assertEqual(expected, report["inherited_source_producer_audit_payload_sha256"])
        self.assertEqual(1, report["inherited_source_producer_source_hashes_count"])
        self.assertFalse(report["training_ready"])

    def test_inspected_v2_sidecar_and_atom_schema_supported(self):
        self.assertEqual(32, self.source(version=2).prepare()["overlay_candidates"])

    def test_input_cardinality_truncation_rejected(self):
        source = self.source()
        source.input_update(lambda d: d["rank_rows"].pop())
        self.assert_rejected(source, "768 rank and 957")

    def test_historical_auxiliary_role_cannot_be_promoted(self):
        source = self.source()
        source.input_update(lambda d: d["auxiliary_rows"][0].update(role="paired_recovery_supervision"))
        self.assert_rejected(source, "role cardinality")

    def test_target_names_and_signed_targets_are_preserved_but_validated(self):
        source = self.source()
        source.input_update(lambda d: d["auxiliary_rows"][0].update(
            targets=[0., -1.] + [0.] * 24, masks=[False, True] + [False] * 24))
        self.assertEqual(-1., overlay.read(source.input)["auxiliary_rows"][0]["targets"][1])
        source.prepare()
        self.assertEqual(-1., overlay.read(source.output / "inherited_training_rows.json")["auxiliary_rows"][0]["targets"][1])

    def test_malformed_legacy_targets_rejected(self):
        source = self.source()
        source.input_update(lambda d: d.update(target_names=list(reversed(d["target_names"]))))
        self.assert_rejected(source, "schema drift")

    def test_invalid_unmasked_legacy_transition_rejected(self):
        source = self.source()
        source.input_update(lambda d: d["auxiliary_rows"][0].update(
            targets=[0., 2.] + [0.] * 24, masks=[False, True] + [False] * 24))
        self.assert_rejected(source, "frozen domain")

    def test_exact_identity_requires_six_literal_fields(self):
        value = dict(dataset="a", suite="libero90", task=0, state=31, candidate_id=0, observed_block_id="a/first_block")
        for mutation in ({"task":True}, {"state":"31"}, {"candidate_id":False}, {"dataset":""}, {"extra":1}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                overlay.identity_key(dict(value, **mutation))
        del value["observed_block_id"]
        with self.assertRaises(ValueError):
            overlay.identity_key(value)

    def test_foreign_source_or_duplicate_identity_rejected(self):
        source = self.source()
        source.records[1]["identity"] = copy.deepcopy(source.records[0]["identity"])
        source.flush()
        self.assert_rejected(source, "Duplicate/unmatched")

    def test_pool_candidate_and_source_identity_must_agree(self):
        source = self.source()
        source.records[0]["candidate_id"] = 1
        source.flush()
        self.assert_rejected(source, "pool/candidate identity")

    def test_record_split_cannot_override_parent(self):
        source = self.source()
        source.records[0]["split"] = "val"
        source.flush()
        self.assert_rejected(source, "split/role")

    def test_split_leakage_rejected_even_when_global_counts_unchanged(self):
        source = self.source()
        def mutate(d):
            d["auxiliary_rows"][0]["split"] = "val"
            d["auxiliary_rows"][638]["split"] = "train"
        source.input_update(mutate)
        self.assert_rejected(source, "split leakage")

    def test_missing_metadata_hash_manifest_entry_rejected(self):
        source = self.source()
        path = source.labels / "SHA256SUMS.txt"
        lines = [line for line in path.read_text().splitlines() if not line.endswith("  records.json")]
        path.write_text("\n".join(lines) + "\n")
        self.assert_rejected(source, "records and audit")

    def test_record_and_manifest_sidecar_digest_must_both_match(self):
        source = self.source()
        source.records[0]["sidecar_sha256"] = "0" * 64
        (source.labels / "records.json").write_bytes(packed(source.records))
        source.rehash()
        self.assert_rejected(source, "fingerprint mismatch")

    def test_corrupt_sidecar_is_detected_before_join(self):
        source = self.source()
        path = source.labels / source.records[0]["sidecar"]
        path.write_bytes(path.read_bytes() + b" ")
        self.assert_rejected(source, "hash mismatch")

    def test_all_17_steps_kept_in_original_order(self):
        source = self.source()
        sidecar = source.sidecars[source.records[0]["sidecar"]]
        sidecar["labels"][4]["step"] = 5
        source.flush()
        self.assert_rejected(source, "original observation time")

    def test_all_seven_atoms_required_at_each_step(self):
        source = self.source()
        del source.sidecars[source.records[0]["sidecar"]]["labels"][4]["atoms"]["joint_state_changed"]
        source.flush()
        self.assert_rejected(source, "exactly seven")

    def test_unmasked_truth_needs_literal_boolean_and_valid_measurement(self):
        source = self.source()
        source.atom()["measurement_valid"] = False
        source.flush()
        self.assert_rejected(source, "measured boolean")

    def test_numeric_masks_are_not_certificates(self):
        source = self.source()
        source.atom()["new_mask"] = 1
        source.flush()
        self.assert_rejected(source, "literal booleans")

    def test_cached_frame_zero_cannot_be_label(self):
        source = self.source()
        source.atom(step=0).update(new_mask=True, physical_value=True, measurement_valid=True)
        source.flush()
        self.assert_rejected(source, "Cached frame0")

    def test_atom_schema_and_deployment_role_cannot_be_forged(self):
        source = self.source()
        source.atom()["deployment_allowed"] = True
        source.flush()
        self.assert_rejected(source, "masquerades")

    def test_source_version_mixing_rejected(self):
        source = self.source()
        source.sidecars[source.records[0]["sidecar"]]["schema"] = "observable_recovery_labels_sidecars_v2"
        source.flush()
        self.assert_rejected(source, "temporal-source contract")

    def test_masked_false_is_not_a_supervised_negative(self):
        source = self.source()
        source.atom(step=7, name="carried_sufficient_evidence")["physical_value"] = False
        source.flush()
        report = source.prepare()
        values = overlay.read(source.output / "event_supervision_rows.json")["rows"][0]
        atom = values["supervision_only"]["physical_event_rows"][7]["targets"]["carried_sufficient_evidence"]
        self.assertIs(atom["measured_value"], False)
        self.assertFalse(atom["supervision_mask"])
        self.assertNotIn("carried_sufficient_evidence.negative", report["certified_atoms_by_split"]["train"])

    def test_same_pool_same_step_pairs_and_positive_only_blockers_are_explicit(self):
        report = self.source().prepare()
        for split in ("train", "val"):
            cell = report["certified_atoms_by_split_and_step"][split]["right_finger_target_contact"]["4"]
            self.assertEqual({"positive":4, "negative":4, "masked":8,
                "same_pool_positive_negative_pairs":4, "pools_with_supervised_difference":4}, cell)
        self.assertIn("positive_only_or_no_certified_train_negative", report["atom_training_blockers"]["carried_sufficient_evidence"])
        self.assertIn("physical_atom_to_predicate_valid_at_use_time_mapping_not_audited",
            report["atom_training_blockers"]["right_finger_target_contact"])

    def test_early_positive_and_late_negative_are_not_same_time_pairs(self):
        source = self.source()
        for n, record in enumerate(source.records):
            if record["candidate_id"] == 1:
                source.atom(n)["new_mask"] = False
                source.atom(n, step=16).update(new_mask=True, physical_value=False, measurement_valid=True)
        source.flush()
        report = source.prepare()
        for split in ("train", "val"):
            cells = report["certified_atoms_by_split_and_step"][split]["right_finger_target_contact"]
            self.assertEqual(0, sum(c["same_pool_positive_negative_pairs"] for c in cells.values()))
            self.assertEqual(4, cells["4"]["positive"])
            self.assertEqual(4, cells["16"]["negative"])

    def test_actual_teacher_changes_labels_never_preexecution_x(self):
        source = self.source()
        before = overlay.extract_raw_candidate_features(overlay.read(source.input)["auxiliary_rows"][797]["features"])
        source.atom()["physical_value"] = False
        source.flush()
        source.prepare()
        actual = overlay.read(source.output / "event_supervision_rows.json")["rows"][0]
        self.assertEqual(before.fingerprint(), actual["preexecution_x"]["fingerprint"])
        self.assertIs(actual["supervision_only"]["physical_event_rows"][4]["targets"]["right_finger_target_contact"]["measured_value"], False)

    def test_source_input_mutation_mid_prepare_prevents_passed_audit(self):
        source = self.source()
        real_extract = overlay.extract_raw_candidate_features
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            if not changed:
                source.input.write_bytes(source.input.read_bytes() + b" ")
                changed = True
            return real_extract(*args, **kwargs)
        with patch.object(overlay, "extract_raw_candidate_features", side_effect=mutate):
            self.assert_rejected(source, "changed during preparation")

    def test_source_sidecar_mutation_mid_prepare_prevents_passed_audit(self):
        source = self.source()
        real_extract = overlay.extract_raw_candidate_features
        changed = False
        def mutate(*args, **kwargs):
            nonlocal changed
            if not changed:
                path = source.labels / source.records[0]["sidecar"]
                path.write_bytes(path.read_bytes() + b" ")
                changed = True
            return real_extract(*args, **kwargs)
        with patch.object(overlay, "extract_raw_candidate_features", side_effect=mutate):
            self.assert_rejected(source, "source changed during preparation")

    def test_source_audit_counts_cannot_disagree_with_atoms(self):
        source = self.source()
        path = source.labels / "observability_label_audit.json"
        audit = overlay.read(path)
        audit["physical_vs_certified_counts"]["right_finger_target_contact"]["certified_positive_frames"] += 1
        path.write_bytes(packed(audit))
        source.rehash()
        self.assert_rejected(source, "counts differ")

    def test_relative_paths_and_overwrite_are_rejected(self):
        source = self.source()
        for name in ("../outside.json", "C:/outside.json", "/outside.json", "sub\\file.json", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                overlay.relative(source.labels.resolve(), name)
        source.output.mkdir()
        self.assert_rejected(source, "overwrite")


if __name__ == "__main__":
    unittest.main()
