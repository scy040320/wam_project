"""Read-only, source-pinned PRE feature support audit; no fitting or selection.

This checks bytes, metadata, the two recorded interaction scopes, and geometry.
It does not join terminal labels, search coefficients, predict learnability, or
replay a policy. The fixed TRAIN target was declared before this audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "source_scoped_feature_support_audit_v1"
BUNDLE_SCHEMA = "source_scoped_shared_current_PRE_bundle_audit_v2"
FEATURE_SCHEMA = "source_scoped_shared_current_forecast_features_v2"
MODES = ("full", "ranker_command", "learned_no_dag", "cause_only", "quality_only",
         "shuffled", "masked_soft_only")
CHANNELS = ("world_anomaly", "execution_anomaly", "observation_corruption",
            "cross_view_conflict", "coarse_unknown", "direct_route_disagreement", "stage_unresolved")
MAIN_NAMES = ("relation_after", "relation_change_shared_current", "contact_after",
    "contact_change_shared_current", "grasp_after", "grasp_change_shared_current",
    "ordered_grasp_forecast", "ordered_release_forecast", "forecast_visibility",
    "relation_quality", "contact_quality", "cross_view_agreement", "forecast_uncertainty",
    "trajectory_risk", "path_efficiency", "net_upward", "net_lateral", "closed_at_entry",
    "first_close_step_fraction", "release_step_fraction", "ordered_release_present",
    "closed_before_release", "close_strength", "open_strength", "reset_stage", "parser_confidence")
SCOPES = ("invalidation_evidence", "unverified_evidence")
INTERACTION_NAMES = tuple(f"{scope}__{channel}__{main}" for scope in SCOPES
                          for channel in CHANNELS for main in MAIN_NAMES)
IDENTITY_FIELDS = {"dataset", "suite", "task", "state", "moment", "condition", "block_id", "block_index"}
TARGET = (9, 2, 0, "clean")
MASKED_MODES = {"ranker_command", "masked_soft_only"}
SCOPE_WIDTH = len(CHANNELS) * len(MAIN_NAMES)


class AuditError(ValueError):
    """Source or representation contract failed, not a training verdict."""


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _sha(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "Literal lowercase SHA256 receipt required")
    return value


def _read(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _key(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 4
             and all(type(v) is int and v >= 0 for v in value[:3])
             and type(value[3]) is str and bool(value[3]), "Exact four-part pool key required")
    return tuple(value)


def _vector(value, width, *, unit=False):
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (width,) and np.isfinite(array).all(), "Finite exact-width record required")
    if unit:
        _require(bool(((0. <= array) & (array <= 1.)).all()), "Probability/exposure outside [0,1]")
    return array


def _inside(root, relative):
    _require(type(relative) is str and bool(relative), "Literal relative matrix path required")
    path = Path(relative)
    _require(not path.is_absolute(), "Absolute output path forbidden")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root), "Matrix path escapes bundle directory")
    return resolved


def _raw_channels(attribution):
    factors, cp, quality = (attribution["factor_probs"], attribution["class_probs"],
                            attribution["evidence_quality"])
    _require(set(cp) == {"normal", "visual_occlusion", "object_shift",
             "execution_contact_deviation", "unknown"}, "Five original class probabilities required")
    probs = _vector(list(cp.values()), 5, unit=True)
    _require(np.isclose(probs.sum(), 1., atol=1e-5), "Class probabilities do not sum to one")
    _require(all(type(quality[k]) is bool for k in
        ("primary_reliable", "wrist_reliable", "execution_reliable", "cross_view_conflict")),
        "Literal original source-quality flags required")
    raw = [1. - factors["world_state_consistent"],
           1. - factors["execution_contact_consistent"],
           1. - factors["observation_reliable"], float(quality["cross_view_conflict"]),
           cp["unknown"], float(max(cp, key=cp.get) != attribution["projected_cause"]),
           1. - factors["task_stage_consistent"]]
    return _vector(raw, 7, unit=True)


def _history_metadata(audit, arm, identity):
    """Validate journal binding only; do not invent or recompute physical facts."""
    snapshot = arm["live_belief"]
    _require(isinstance(snapshot.get("facts"), dict) and isinstance(snapshot.get("history"), list),
             "Explicit own-variant facts and history required")
    if "task_id" in snapshot:
        _require(snapshot["task_id"] == identity["task"], "Journal task identity differs")
    context = audit["belief_context"]
    _require(isinstance(context, dict), "Explicit recorded belief context required")
    for predicate, fact in snapshot["facts"].items():
        _require(predicate in context, "Live fact omitted from feature metadata")
        record = context[predicate]
        for field in ("value", "source", "confidence", "updated_at_block"):
            _require(record[field] == fact[field], "Feature context differs from own journal fact")
        _require(record["source_evidence_ids"] == fact["evidence_ids"], "Fact evidence IDs differ")
        latest = next((h for h in reversed(snapshot["history"]) if
            h.get("predicate") == predicate and h.get("evidence_id") in fact["evidence_ids"]), None)
        _require(record["latest_matching_reason"] == (None if latest is None else latest.get("reason")),
                 "Feature context borrowed another history transition")
    for predicate, record in context.items():
        _require(record.get("prior_is_current_certificate") is False,
                 "Declared soft prior promoted to a current certificate")
        if predicate not in snapshot["facts"]:
            _require(record.get("source") == "absent_PRE_fact" and record.get("value") == "unknown",
                     "Absent fact presented as a sourced live fact")
        for historical in record.get("historical_invalidation_evidence", []):
            _require(historical.get("historical_block_time_authenticated") is False
                     and historical.get("physical_state_certified") is False,
                     "Untimed historical provenance promoted to physical truth")
            matches = [h for h in snapshot["history"] if h.get("predicate") == predicate
                and h.get("reason") == historical["reason"]
                and h.get("evidence_id") == historical["evidence_id"]
                and h.get("propagation_path") == historical["propagation_path"]
                and h.get("new_value") in {"unknown", "false"}]
            roots = [h for h in snapshot["history"] if h.get("predicate") == historical["root"]
                and h.get("reason") == historical["reason"]
                and h.get("evidence_id") == historical["evidence_id"] and h.get("direct") is True
                and h.get("propagation_path") == [historical["root"]]
                and h.get("new_value") in {"unknown", "false"}]
            _require(bool(matches) and bool(roots)
                     and historical["root_strength"] == roots[-1]["new_confidence"],
                     "Historical interaction provenance absent from own journal")


def geometry(left, right, names):
    """Raw PRE vector geometry, not a learned metric or coefficient search."""
    left, right = _vector(left, len(names)), _vector(right, len(names))
    difference = left - right
    norms = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    cosine = None if 0. in norms else float(np.clip(np.dot(left, right) / (norms[0] * norms[1]), -1., 1.))
    return dict(euclidean_distance=float(np.linalg.norm(difference)), cosine_similarity=cosine,
        cosine_undefined_reason="zero_vector" if cosine is None else None,
        left_norm=norms[0], right_norm=norms[1], exact_equal=bool(np.array_equal(left, right)),
        changed_features=[dict(name=name, left=float(a), right=float(b), difference=float(a-b))
                          for name, a, b in zip(names, left, right) if a != b])


def audit_bundle(completion_report, *, expected_report_sha256, expected_journal_sha256,
                 decision_journal=None):
    """Authenticate original source journal and matrices, then report support."""
    report_path = Path(completion_report).resolve()
    _require(sha256(report_path) == _sha(expected_report_sha256), "Completion report SHA256 differs")
    report = _read(report_path)
    _require(report.get("passed") is True and report.get("schema") == BUNDLE_SCHEMA,
             "Passed v2 PRE builder audit required")
    _require(report.get("modes") == list(MODES) and report.get("main_features") == 26
             and report.get("interaction_features") == 364, "Exact main26 + two182 scope schema required")
    for flag in ("train_only_channel_mean", "ordinary_main_identical_across_controls", "own_variant_history_used"):
        _require(report.get(flag) is True, "Required PRE contract missing: " + flag)
    for flag in ("terminal_outcomes_consumed", "terminal_labels_joined", "unknown_or_forecast_proxy_admitted_as_fact",
                 "model_deployed", "closedloop_run", "utility_improvement_claim"):
        _require(report.get(flag) is False, "Read-only PRE boundary violated: " + flag)
    for field in ("physical_TRUE_facts_created", "policy_queries", "actions", "fits"):
        _require(type(report.get(field)) is int and report[field] == 0, "PRE side effect recorded: " + field)
    sources, outputs = report["source_sha256"], report["output_sha256"]
    _require(isinstance(sources, dict) and bool(sources) and isinstance(outputs, dict) and bool(outputs),
             "Source and output pin manifests required")
    for path, expected in sources.items():
        _require(Path(path).is_absolute(), "Source pin must be absolute")
        _require(sha256(path) == _sha(expected), "Pinned source SHA256 differs: " + path)
    journal_paths = [Path(path).resolve() for path in sources if Path(path).name == "decision_journals.json"]
    _require(len(journal_paths) == 1, "Exactly one pinned original decision journal required")
    journal_path = journal_paths[0] if decision_journal is None else Path(decision_journal).resolve()
    _require(journal_path == journal_paths[0]
             and sources[str(journal_path)] == _sha(expected_journal_sha256)
             and sha256(journal_path) == expected_journal_sha256, "Independent original journal receipt differs")
    journals = _read(journal_path)
    _require(isinstance(journals, list), "Original journal must be a list of pool records")
    journal_map = {_key(row["key"]): row for row in journals}
    _require(len(journal_map) == len(journals), "Duplicate original journal pool key")
    rows = report["rows"]
    _require(isinstance(rows, list) and bool(rows), "Nonempty exact pool metadata required")
    keys = [_key(row["key"]) for row in rows]
    _require(len(set(keys)) == len(keys) and set(keys) == set(journal_map), "Bundle/journal membership differs")
    _require(set(outputs) == {row["matrices"] for row in rows} and len(outputs) == len(rows),
             "Exact one-matrix-archive-per-pool pin set required")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val")}
    _require(sum(counts.values()) == len(rows) and counts["train"] > 0,
             "Only TRAIN/VAL rows, with nonempty TRAIN, are allowed")
    _require((report["pools"], report["candidates"], report["train"], report["val"]) ==
             (len(rows), 4*len(rows), counts["train"], counts["val"]), "Declared pool/split counts differ")
    groups = {}
    for row, key in zip(rows, keys):
        group = key[:2]
        _require(group not in groups or groups[group] == row["split"], "Task/state group split leakage")
        groups[group] = row["split"]
    center = report["centering"]
    mean = _vector(center["mean"], 7, unit=True)
    _require(center["source_count"] == counts["train"], "Centering source count must be TRAIN only")
    _sha(center["fingerprint"])
    per_mode = {mode: dict(nonzero_interaction_pools=0, candidate_specific_pools=0,
        main_candidate_specific_pools=0, nonzero_entries=0,
        scope_nonzero_pools={scope: 0 for scope in SCOPES},
        scope_candidate_specific_pools={scope: 0 for scope in SCOPES}) for mode in MODES}
    comparisons = {mode: dict(changed_pools=0, changed_candidate_rows=0, changed_entries=0,
        maximum_candidate_l2=0., scope_changed_pools={scope: 0 for scope in SCOPES}) for mode in MODES[1:]}
    target, calibration = None, []
    own_history_checks = 0
    for row, key in zip(rows, keys):
        path = _inside(report_path.parent, row["matrices"])
        _require(sha256(path) == _sha(outputs[row["matrices"]]), "Output matrix SHA256 differs: " + path.name)
        with np.load(path, allow_pickle=False) as archive:
            expected_names = {f"{mode}_{part}" for mode in MODES for part in ("main", "interaction")}
            _require(set(archive.files) == expected_names, "Exact seven-control matrix membership required")
            matrices = {name: archive[name] for name in expected_names}
        for mode in MODES:
            main, interaction = matrices[f"{mode}_main"], matrices[f"{mode}_interaction"]
            _require(main.dtype == np.dtype("float64") and interaction.dtype == np.dtype("float64")
                and main.shape == (4, 26) and interaction.shape == (4, 364)
                and np.isfinite(main).all() and np.isfinite(interaction).all(), "Finite float64 K4 exact shape required")
            _require(main.tobytes() == matrices["full_main"].tobytes(), "Ordinary main features differ across controls")
            audit = row["audits"][mode]
            identity = audit["identity"]
            _require(set(identity) == IDENTITY_FIELDS and _key([identity[k] for k in
                     ("task", "state", "moment", "condition")]) == key and identity["suite"] == "libero90"
                     and type(identity["block_index"]) is int and identity["block_index"] >= 0
                     and identity["dataset"] and identity["block_id"], "Source-scoped feature identity differs")
            _require(audit["schema"] == FEATURE_SCHEMA and audit["split"] == row["split"]
                and audit["variant"] == mode and audit["main_names"] == list(MAIN_NAMES)
                and audit["interaction_names"] == list(INTERACTION_NAMES)
                and audit["exposure_scopes"] == list(SCOPES), "Feature record/schema binding differs")
            _require(audit["no_dag"] is (mode == "learned_no_dag")
                     and audit["mask_learned"] is (mode in MASKED_MODES), "Control identity changed")
            for flag in ("invalidation_evidence_is_not_physical_FALSE", "unknown_evidence_is_not_physical_invalidation",
                "retained_prior_cannot_pass_current_hard_gate", "own_variant_history_identity_bound",
                "ordinary_forecast_source_contract_audited", "full_ordinary_main_capacity_preserved"):
                _require(audit.get(flag) is True, "Feature boundary missing: " + flag)
            for flag in ("ordinary_main_gated_by_learned_quality", "live_belief_mutated", "actual_after_used",
                         "terminal_labels_used", "model_fitted", "deployed"):
                _require(audit.get(flag) is False, "Feature boundary violated: " + flag)
            _require(audit["current_physical_certificates_generated"] == 0 and audit["prediction_writes"] == 0,
                     "Feature construction fabricated current physical facts")
            named_sources = audit["source_sha256"]
            required_sources = {"current_primary", "current_wrist", "localizer_model", "localizer_code",
                "shared_fusion_code", "candidate_parser_code", "attribution"}
            required_sources.update(f"candidate_{cid}_{field}" for cid in range(4)
                                    for field in ("plan", "predicted_primary", "predicted_wrist"))
            _require(set(named_sources) == required_sources and all(_sha(v) in sources.values()
                for v in named_sources.values()) and named_sources["attribution"] == expected_journal_sha256,
                "Named input references are not authenticated by pinned source bytes")
            base = "full" if mode == "masked_soft_only" else mode
            arm = journal_map[key]["variants"][base]
            _require(arm["attribution"]["source_block_id"] == identity["block_id"], "Journal block differs")
            _history_metadata(audit, arm, identity)
            own_history_checks += 1
            raw = _raw_channels(arm["attribution"])
            centered = raw - mean
            _require(np.array_equal(raw, _vector(audit["raw_channels"], 7, unit=True))
                and np.array_equal(centered, _vector(audit["centered_channels"], 7))
                and audit["centering_fingerprint"] == center["fingerprint"]
                and audit["centering_source_count"] == counts["train"], "Channel content/TRAIN centering differs")
            quality = arm["attribution"]["evidence_quality"]
            visual_usable = (quality["primary_reliable"] or quality["wrist_reliable"]) and not quality["cross_view_conflict"]
            _require(audit["learned_visual_channel_usable"] is visual_usable
                     and audit["execution_channel_soft_usable"] is quality["execution_reliable"], "Quality source routing differs")
            filtered = centered.copy()
            if not visual_usable:
                filtered[[0, 2, 3]] = 0.
            if not quality["execution_reliable"]:
                filtered[1] = 0.
            exposure = audit["dependency_exposure"]
            _require(isinstance(exposure, list) and len(exposure) == 4
                     and isinstance(audit["soft_required_facts"], list)
                     and len(audit["soft_required_facts"]) == 4, "K4 exposure/required-fact metadata missing")
            reconstructed = []
            for cid, item in enumerate(exposure):
                _require(set(item) == set(SCOPES), "Invalidation and unverified scopes conflated")
                parts = [np.outer(filtered * _vector(item[scope], 7, unit=True), main[cid]).reshape(-1)
                         for scope in SCOPES]
                reconstructed.append(np.concatenate(parts))
            reconstructed = np.asarray(reconstructed)
            if mode in MASKED_MODES:
                reconstructed[:] = 0.
            _require(np.array_equal(interaction, reconstructed), "Interaction matrix differs from recorded channel/scope product")
            stat = per_mode[mode]
            stat["nonzero_interaction_pools"] += int(np.any(interaction != 0.))
            stat["candidate_specific_pools"] += int(np.any(np.ptp(interaction, axis=0) != 0.))
            stat["main_candidate_specific_pools"] += int(np.any(np.ptp(main, axis=0) != 0.))
            stat["nonzero_entries"] += int(np.count_nonzero(interaction))
            for scope_index, scope in enumerate(SCOPES):
                scope_matrix = interaction[:, scope_index*SCOPE_WIDTH:(scope_index+1)*SCOPE_WIDTH]
                stat["scope_nonzero_pools"][scope] += int(np.any(scope_matrix != 0.))
                stat["scope_candidate_specific_pools"][scope] += int(np.any(np.ptp(scope_matrix, axis=0) != 0.))
            if mode == "full" and row["split"] == "train":
                calibration.append(dict(identity=identity, sources={"attribution":expected_journal_sha256}, channels=raw.tolist()))
        full = matrices["full_interaction"]
        for mode, stat in comparisons.items():
            other = matrices[f"{mode}_interaction"]
            diff = full - other
            stat["changed_pools"] += int(np.any(diff != 0.))
            stat["changed_candidate_rows"] += int(np.any(diff != 0., axis=1).sum())
            stat["changed_entries"] += int(np.count_nonzero(diff))
            stat["maximum_candidate_l2"] = max(stat["maximum_candidate_l2"], float(np.linalg.norm(diff, axis=1).max()))
            for index, scope in enumerate(SCOPES):
                stat["scope_changed_pools"][scope] += int(np.any(diff[:, index*SCOPE_WIDTH:(index+1)*SCOPE_WIDTH] != 0.))
        if key == TARGET:
            _require(row["split"] == "train", "Predeclared geometry target must remain original TRAIN")
            target = dict(key=list(TARGET), split="train", candidate_ids=[0, 3], label_joined=False,
                modes={mode: dict(main=geometry(matrices[f"{mode}_main"][0], matrices[f"{mode}_main"][3], MAIN_NAMES),
                    **{scope:geometry(matrices[f"{mode}_interaction"][0, index*SCOPE_WIDTH:(index+1)*SCOPE_WIDTH],
                         matrices[f"{mode}_interaction"][3, index*SCOPE_WIDTH:(index+1)*SCOPE_WIDTH],
                         INTERACTION_NAMES[index*SCOPE_WIDTH:(index+1)*SCOPE_WIDTH]) for index,scope in enumerate(SCOPES)},
                    complete_PRE=geometry(np.concatenate((matrices[f"{mode}_main"][0], matrices[f"{mode}_interaction"][0])),
                        np.concatenate((matrices[f"{mode}_main"][3], matrices[f"{mode}_interaction"][3])), MAIN_NAMES+INTERACTION_NAMES))
                       for mode in MODES}, normalization="none: raw audited PRE units", learnability_claim=False)
    calibration.sort(key=lambda item: _digest(item["identity"]))
    fingerprint = _digest(dict(schema=FEATURE_SCHEMA, channels=list(CHANNELS), rows=calibration))
    _require(fingerprint == center["fingerprint"] and np.array_equal(
        np.mean([item["channels"] for item in calibration], axis=0), mean), "Centering fingerprint/mean differs from original TRAIN")
    return dict(passed=True, schema=SCHEMA, completion_report=dict(path=str(report_path), sha256=expected_report_sha256),
        original_journal=dict(path=str(journal_path), sha256=expected_journal_sha256), source_files_verified=len(sources),
        matrix_files_verified=len(outputs), pools=len(rows), candidates=4*len(rows), **counts,
        main_features=26, interaction_features=364, interaction_scope_width=182,
        ordinary_main_bitwise_equal_across_all_controls=True, masked_and_ranker_interactions_exactly_zero=True,
        own_variant_history_records_verified=own_history_checks, train_centering_fingerprint_verified=True,
        per_mode=per_mode, full_vs_control=comparisons, fixed_train_target=target,
        target_missing=target is None, geometry_only=True, terminal_labels_accessed=False, validation_labels_accessed=False,
        fits=0, coefficient_searches=0, model_parameters_changed=False, actions=0, policy_queries=0,
        selection_run=False, utility_improvement_claim=False, physical_fact_or_certificate_claim=False,
        interpretation="Nonzero differences demonstrate represented input distinctions, not learnability or policy benefit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-report", type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--decision-journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "Fresh audit output required; prior results cannot be overwritten")
    result = audit_bundle(args.completion_report, expected_report_sha256=args.completion_sha256,
                          expected_journal_sha256=args.journal_sha256, decision_journal=args.decision_journal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("passed", "pools", "train", "val", "geometry_only", "fits")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
