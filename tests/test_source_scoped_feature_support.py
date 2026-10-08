"""Synthetic source-pinned PRE bundles; no labels, fitting, or coefficient search."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_source_scoped_feature_support.py"
_SPEC = importlib.util.spec_from_file_location("feature_support_audit", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class FeatureSupportAuditTests(unittest.TestCase):
    def setUp(self):
        # Keep generated fixtures inside the explicitly writable workspace;
        # Windows app sandboxes can expose a non-writable default temp root.
        self.temp = tempfile.TemporaryDirectory(prefix="feature_support_test_", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.journal = self.root / "decision_journals.json"
        self.current = self.root / "current.dat"
        self.current.write_bytes(b"authenticated shared CURRENT and own FORECAST fixture")
        # Two TRAIN identities are enough to exercise nonzero TRAIN centering;
        # counts remain exact and test geometry target is predeclared, not chosen.
        self.keys = [[9, 2, 0, "clean"], [46, 3, 0, "object_shift"], [46, 6, 0, "clean"]]
        journals = []
        for row_index, key in enumerate(self.keys):
            variants = {}
            for mode in audit.MODES[:-1]:
                world = .2 + .2*row_index
                if mode == "cause_only":
                    world = .9
                if mode == "shuffled":
                    world = .8 - .1*row_index
                quality = dict(primary_reliable=True, wrist_reliable=True,
                    execution_reliable=True, cross_view_conflict=mode == "quality_only")
                attr = dict(factor_probs=dict(world_state_consistent=1-world,
                    execution_contact_consistent=.7-.1*row_index, observation_reliable=.9, task_stage_consistent=.8),
                    class_probs=dict(normal=.5, visual_occlusion=.1, object_shift=.15,
                        execution_contact_deviation=.1, unknown=.15), projected_cause="normal",
                    evidence_quality=quality, source_block_id=f"libero90_task{key[0]}_block0")
                fact = dict(value="unknown", source="attribution", confidence=.8,
                    updated_at_block=1, evidence_ids=["test_world_transition"])
                hist = dict(predicate="target_pose_current", reason="world_state_consistent",
                    new_value="unknown", new_confidence=.8, evidence_id="test_world_transition",
                    direct=True, propagation_path=["target_pose_current"])
                variants[mode] = dict(attribution=attr, live_belief=dict(
                    task_id=key[0], facts={"target_pose_current":fact}, history=[hist]))
            # No terminal labels or candidate results are present anywhere.
            journals.append(dict(key=key, variants=variants))
        self.journal.write_text(json.dumps(journals), encoding="utf-8")
        self.journal_sha = audit.sha256(self.journal)
        self.source_sha = audit.sha256(self.current)
        rows, calibration = [], []
        for index, key in enumerate(self.keys):
            identity = dict(dataset="immutable_original_dataset", suite="libero90", task=key[0],
                state=key[1], moment=key[2], condition=key[3], block_id=f"libero90_task{key[0]}_block0", block_index=1)
            split = "val" if index == 2 else "train"
            rows.append(dict(key=key, split=split, matrices=f"pool_{index}.npz", audits={}))
            if split == "train":
                calibration.append(dict(identity=identity, sources={"attribution":self.journal_sha},
                    channels=audit._raw_channels(journals[index]["variants"]["full"]["attribution"]).tolist()))
        calibration.sort(key=lambda r:audit._digest(r["identity"]))
        mean = np.mean([r["channels"] for r in calibration], axis=0)
        fingerprint = audit._digest(dict(schema=audit.FEATURE_SCHEMA, channels=list(audit.CHANNELS), rows=calibration))
        named = {name:self.source_sha for name in ("current_primary", "current_wrist", "localizer_model",
            "localizer_code", "shared_fusion_code", "candidate_parser_code")}
        named.update({f"candidate_{cid}_{field}":self.source_sha for cid in range(4)
                      for field in ("plan", "predicted_primary", "predicted_wrist")})
        named["attribution"] = self.journal_sha
        for index, row in enumerate(rows):
            matrices = {}
            main = np.arange(104, dtype=np.float64).reshape(4,26) / 104.
            identity = dict(dataset="immutable_original_dataset", suite="libero90", task=row["key"][0],
                state=row["key"][1], moment=row["key"][2], condition=row["key"][3],
                block_id=f"libero90_task{row['key'][0]}_block0", block_index=1)
            for mode in audit.MODES:
                base = "full" if mode == "masked_soft_only" else mode
                arm = journals[index]["variants"][base]
                raw = audit._raw_channels(arm["attribution"])
                centered = raw - mean
                filtered = centered.copy()
                usable = not arm["attribution"]["evidence_quality"]["cross_view_conflict"]
                if not usable:
                    filtered[[0,2,3]] = 0.
                scale = .3 if mode == "learned_no_dag" else .8
                exposures = [{"invalidation_evidence":[scale*(1-.1*cid)]*7,
                              "unverified_evidence":[1-scale]*7} for cid in range(4)]
                interaction = np.array([np.concatenate([np.outer(filtered*np.array(e[scope]),main[cid]).reshape(-1)
                    for scope in audit.SCOPES]) for cid,e in enumerate(exposures)])
                if mode in audit.MASKED_MODES:
                    interaction[:] = 0.
                matrices[f"{mode}_main"], matrices[f"{mode}_interaction"] = main.copy(), interaction
                metadata = dict(schema=audit.FEATURE_SCHEMA, identity=identity, split=row["split"], variant=mode,
                    main_names=list(audit.MAIN_NAMES), interaction_names=list(audit.INTERACTION_NAMES),
                    exposure_scopes=list(audit.SCOPES), no_dag=mode=="learned_no_dag", mask_learned=mode in audit.MASKED_MODES,
                    source_sha256=named, centering_fingerprint=fingerprint, centering_source_count=2,
                    raw_channels=raw.tolist(), centered_channels=centered.tolist(), dependency_exposure=exposures,
                    soft_required_facts=[{"target_pose_current":.6} for _ in range(4)],
                    learned_visual_channel_usable=usable, execution_channel_soft_usable=True,
                    current_physical_certificates_generated=0, prediction_writes=0,
                    belief_context={"target_pose_current":dict(value="unknown", source="attribution", confidence=.8,
                        updated_at_block=1, source_evidence_ids=["test_world_transition"],
                        latest_matching_reason="world_state_consistent", prior_is_current_certificate=False,
                        historical_invalidation_evidence=[dict(reason="world_state_consistent",
                            evidence_id="test_world_transition", root="target_pose_current", root_strength=.8,
                            propagation_path=["target_pose_current"], historical_block_time_authenticated=False,
                            physical_state_certified=False)])})
                for flag in ("invalidation_evidence_is_not_physical_FALSE", "unknown_evidence_is_not_physical_invalidation",
                    "retained_prior_cannot_pass_current_hard_gate", "own_variant_history_identity_bound",
                    "ordinary_forecast_source_contract_audited", "full_ordinary_main_capacity_preserved"):
                    metadata[flag] = True
                for flag in ("ordinary_main_gated_by_learned_quality", "live_belief_mutated", "actual_after_used",
                             "terminal_labels_used", "model_fitted", "deployed"):
                    metadata[flag] = False
                row["audits"][mode] = metadata
            np.savez_compressed(self.root / row["matrices"], **matrices)
        self.report = dict(passed=True, schema=audit.BUNDLE_SCHEMA, modes=list(audit.MODES), main_features=26,
            interaction_features=364, pools=3, candidates=12, train=2, val=1, rows=rows,
            source_sha256={str(self.journal):self.journal_sha, str(self.current):self.source_sha},
            output_sha256={row["matrices"]:audit.sha256(self.root / row["matrices"]) for row in rows},
            centering=dict(mean=mean.tolist(), fingerprint=fingerprint, source_count=2))
        for flag in ("train_only_channel_mean", "ordinary_main_identical_across_controls", "own_variant_history_used"):
            self.report[flag] = True
        for flag in ("terminal_outcomes_consumed", "terminal_labels_joined", "unknown_or_forecast_proxy_admitted_as_fact",
                     "model_deployed", "closedloop_run", "utility_improvement_claim"):
            self.report[flag] = False
        for field in ("physical_TRUE_facts_created", "policy_queries", "actions", "fits"):
            self.report[field] = 0
        self.report_path = self.root / "completion_audit.json"

    def run_audit(self, report=None):
        self.report_path.write_text(json.dumps(self.report if report is None else report), encoding="utf-8")
        return audit.audit_bundle(self.report_path, expected_report_sha256=audit.sha256(self.report_path),
                                  expected_journal_sha256=self.journal_sha)

    def change_matrix(self, mode, part, edit):
        path = self.root / self.report["rows"][0]["matrices"]
        with np.load(path, allow_pickle=False) as archive:
            data = {key:archive[key] for key in archive.files}
        edit(data[f"{mode}_{part}"])
        np.savez_compressed(path, **data)
        self.report["output_sha256"][path.name] = audit.sha256(path)

    def test_valid_bundle_without_terminal_labels(self):
        result = self.run_audit()
        self.assertTrue(result["passed"])
        self.assertTrue(result["ordinary_main_bitwise_equal_across_all_controls"])
        self.assertTrue(result["masked_and_ranker_interactions_exactly_zero"])
        self.assertEqual(result["own_variant_history_records_verified"], 21)
        self.assertEqual(result["fixed_train_target"]["key"], [9,2,0,"clean"])
        self.assertGreater(result["fixed_train_target"]["modes"]["full"]["complete_PRE"]["euclidean_distance"], 0.)
        self.assertGreater(result["full_vs_control"]["learned_no_dag"]["changed_pools"], 0)
        self.assertGreater(result["full_vs_control"]["cause_only"]["changed_pools"], 0)
        self.assertGreater(result["full_vs_control"]["quality_only"]["changed_pools"], 0)
        self.assertFalse(result["fixed_train_target"]["learnability_claim"])
        self.assertFalse(result["validation_labels_accessed"])
        self.assertEqual(result["fits"], 0)

    def test_completion_pin_rejects_tampering(self):
        self.run_audit()
        expected = audit.sha256(self.report_path)
        self.report_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(audit.AuditError, "Completion report SHA256"):
            audit.audit_bundle(self.report_path, expected_report_sha256=expected, expected_journal_sha256=self.journal_sha)

    def test_original_journal_must_have_independent_pin(self):
        self.run_audit()
        with self.assertRaisesRegex(audit.AuditError, "journal receipt"):
            audit.audit_bundle(self.report_path, expected_report_sha256=audit.sha256(self.report_path),
                               expected_journal_sha256="0"*64)

    def test_source_and_output_tampering(self):
        self.current.write_bytes(b"not frozen source")
        with self.assertRaisesRegex(audit.AuditError, "Pinned source SHA256"):
            self.run_audit()
        self.current.write_bytes(b"authenticated shared CURRENT and own FORECAST fixture")
        path = self.root / self.report["rows"][0]["matrices"]
        path.write_bytes(b"tampered output")
        with self.assertRaisesRegex(audit.AuditError, "Output matrix SHA256"):
            self.run_audit()

    def test_schema_scope_order_is_strict(self):
        self.report["rows"][0]["audits"]["full"]["exposure_scopes"].reverse()
        with self.assertRaisesRegex(audit.AuditError, "schema binding"):
            self.run_audit()

    def test_train_centering_cannot_be_replaced_by_val_statistics(self):
        self.report["centering"]["source_count"] = 3
        with self.assertRaisesRegex(audit.AuditError, "Centering source count"):
            self.run_audit()

    def test_visual_unreliability_does_not_erase_execution_interaction(self):
        result = self.run_audit()
        path = self.root / self.report["rows"][0]["matrices"]
        with np.load(path, allow_pickle=False) as archive:
            interaction = archive["quality_only_interaction"]
        # World channel is visual; reliable execution channel is independent.
        self.assertTrue(np.all(interaction[:, :26] == 0.))
        self.assertTrue(np.any(interaction[:, 26:52] != 0.))
        self.assertGreater(result["per_mode"]["quality_only"]["candidate_specific_pools"], 0)

    def test_history_provenance_cannot_be_a_physical_certificate(self):
        context = self.report["rows"][0]["audits"]["full"]["belief_context"]["target_pose_current"]
        context["historical_invalidation_evidence"][0]["physical_state_certified"] = True
        with self.assertRaisesRegex(audit.AuditError, "promoted to physical truth"):
            self.run_audit()

    def test_main_equality_is_bit_exact(self):
        self.change_matrix("shuffled", "main", lambda array: array.__setitem__((0,0), np.nextafter(array[0,0], 1.)))
        with self.assertRaisesRegex(audit.AuditError, "main features differ"):
            self.run_audit()

    def test_mask_must_zero_both_interaction_scopes(self):
        self.change_matrix("masked_soft_only", "interaction", lambda array: array.__setitem__((1,250), .01))
        with self.assertRaisesRegex(audit.AuditError, "recorded channel/scope product"):
            self.run_audit()

    def test_wrong_scope_matrix_or_nonfinite_rejected(self):
        self.change_matrix("full", "interaction", lambda array: array.__setitem__((0,0), array[0,0]+.01))
        with self.assertRaisesRegex(audit.AuditError, "recorded channel/scope product"):
            self.run_audit()
        self.change_matrix("full", "interaction", lambda array: array.__setitem__((0,0), np.nan))
        with self.assertRaisesRegex(audit.AuditError, "Finite float64 K4"):
            self.run_audit()

    def test_own_variant_history_metadata_not_borrowed(self):
        self.report["rows"][0]["audits"]["learned_no_dag"]["belief_context"]["target_pose_current"]["confidence"] = .3
        with self.assertRaisesRegex(audit.AuditError, "own journal fact"):
            self.run_audit()

    def test_output_escape_rejected(self):
        report = copy.deepcopy(self.report)
        report["rows"][0]["matrices"] = "../outside.npz"
        report["output_sha256"]["../outside.npz"] = report["output_sha256"].pop("pool_0.npz")
        with self.assertRaisesRegex(audit.AuditError, "escapes bundle"):
            self.run_audit(report)

    def test_zero_vector_geometry_does_not_claim_similarity(self):
        result = audit.geometry([0.,0.], [1.,0.], ("x", "y"))
        self.assertIsNone(result["cosine_similarity"])
        self.assertEqual(result["cosine_undefined_reason"], "zero_vector")
        self.assertEqual(result["euclidean_distance"], 1.)

    def test_task_state_split_leakage_is_not_ignored(self):
        self.report["rows"][2]["key"] = [46,3,1,"clean"]
        # Membership fails first: cannot disguise a new split as original frozen sources.
        with self.assertRaises(audit.AuditError):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
