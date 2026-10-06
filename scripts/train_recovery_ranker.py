"""Frozen-protocol V9 recovery-ranker fitting and offline same-pool controls.

No collection, validation tuning, deployment mutation or automatic 128-scene
run is performed.  Saved candidate outcomes are supervision/evaluation only;
the common selection wrapper receives scores and official values, never GT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from wam_reranking.candidate_utility import CandidateUtilityModel, select_with_utility
from wam_reranking.contracts import CandidateDecision
from wam_reranking.recovery_contract import (
    COMMAND_FEATURE_NAMES, FEATURE_NAMES, FEATURE_SCHEMA, require_recovery_training_ready,
)
from wam_reranking.recovery_label_contract import (
    OBSERVABILITY_TARGETS, SEALED_32_GROUPS, TARGETS,
)
from wam_reranking.recovery_residual import (
    AUXILIARY_LOSS_WEIGHT, L2_WEIGHT, RESIDUAL_CAP, RANK_LOSS_WEIGHT,
    fit_recovery_residual,
)

EPOCHS = 300
SEED = 0
METHODS = ("value_only", "candidate_backbone", "full",
           "masked_attribution_same_model", "shuffled_attribution_same_model",
           "no_dependency_same_model", "retrained_masked_model")
EXPECTED_TARGET_NAMES = tuple(f"{name}.{endpoint}"
    for name in (*TARGETS, *OBSERVABILITY_TARGETS, "relation_support_2d")
    for endpoint in ("after", "transition"))


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return sha256(path.read_bytes()).hexdigest()


def _suite(value):
    if not isinstance(value, str) or not value:
        raise ValueError("explicit task suite required; naked task IDs cannot identify groups")
    return value.lower().replace("_", "").replace("-", "")


def _group(row, dataset):
    identity = row.get("identity", {})
    if not isinstance(identity, dict):
        raise ValueError("malformed source-scoped row identity")
    suite = identity.get("suite", row.get("suite", dataset.get("suite")))
    task = identity.get("task", row.get("task"))
    state = identity.get("state", row.get("state"))
    if type(task) is not int or type(state) is not int or min(task, state) < 0:
        raise ValueError("explicit nonnegative task/state required for rank AND auxiliary rows")
    if "task" in row and task != row["task"] or "state" in row and state != row["state"]:
        raise ValueError("row/identity task-state conflict")
    if "suite" in row and _suite(suite) != _suite(row["suite"]):
        raise ValueError("row/identity task-suite conflict")
    return (_suite(suite), task, state)


def _finite_vector(values, width, name):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid finite {name} vector")
    return vector


def _success(value):
    if not isinstance(value, (bool, int)) or value not in (0, 1):
        raise ValueError("success supervision must be literal boolean/0/1")
    return bool(value)


def mask_attribution_features(values):
    """Retain only the 14 generic planned-command fields, not cause channels."""
    values = _finite_vector(values, len(FEATURE_NAMES), "recovery")
    keep = np.asarray([name.startswith("command.") for name in FEATURE_NAMES])
    if int(keep.sum()) != len(COMMAND_FEATURE_NAMES):
        raise ValueError("global command feature contract drift")
    return values * keep


def normalize_auxiliary_targets(row, names):
    """Copy signed delta labels into BCE coordinates; never rewrite silver."""
    targets = np.asarray(row["targets"], dtype=np.float64)
    masks = np.asarray(row["masks"])
    if targets.shape != (len(names),) or masks.shape != targets.shape or not np.isin(masks, (0, 1)).all():
        raise ValueError("auxiliary target/mask schema mismatch")
    mask = masks.astype(bool)
    result = np.zeros_like(targets)
    for index, name in enumerate(names):
        if not mask[index]:
            continue
        value = targets[index]
        if not np.isfinite(value):
            raise ValueError("unmasked recovery target is nonfinite")
        if name.endswith(".transition"):
            if not -1. <= value <= 1.:
                raise ValueError("signed transition target outside [-1,1]; do not silently clip")
            result[index] = .5 + .5 * value
        else:
            if not 0. <= value <= 1.:
                raise ValueError("after/availability target outside [0,1]")
            result[index] = value
    return result, mask


def audit_training_input(dataset, backbone):
    problems = []
    try:
        require_recovery_training_ready(dataset.get("training_audit", {}))
    except (RuntimeError, TypeError) as error:
        problems.append(str(error))
    if dataset.get("feature_schema") != FEATURE_SCHEMA:
        problems.append("feature_version_mismatch")
    if tuple(dataset.get("feature_names", ())) != FEATURE_NAMES:
        problems.append("feature_names_order_mismatch")
    target_names = tuple(dataset.get("target_names", ()))
    if not target_names or len(target_names) != len(set(target_names)) or any(
            not isinstance(name, str) or not name for name in target_names):
        problems.append("target_names_malformed")
    if target_names != EXPECTED_TARGET_NAMES:
        problems.append("target_names_26_endpoint_transition_contract_mismatch")
    rank = dataset.get("rank_rows", ())
    auxiliary = dataset.get("auxiliary_rows", ())
    if not isinstance(rank, list) or not rank:
        problems.append("missing_rank_rows"); rank = []
    if not isinstance(auxiliary, list):
        problems.append("malformed_auxiliary_rows"); auxiliary = []
    groups = {"train": set(), "val": set()}
    ids, auxiliary_ids, pool_metadata = set(), set(), {}
    counts = Counter()
    forbidden = set(SEALED_32_GROUPS)
    for group in dataset.get("reserved_groups", ()):
        try:
            forbidden.add((_suite(group[0]), int(group[1]), int(group[2])))
        except (ValueError, TypeError, IndexError):
            problems.append("malformed_reserved_group")
    for branch, rows in (("rank", rank), ("auxiliary", auxiliary)):
        for index, row in enumerate(rows):
            prefix = f"{branch}{index}:"
            try:
                if not isinstance(row, dict):
                    raise ValueError("row must be a mapping")
                split = row.get("split")
                if split not in ("train", "val"):
                    raise ValueError("non-development split")
                for key in ("role", "dataset_role"):
                    if str(row.get(key, "")).lower() in (
                            "qc", "clean_b", "confirmation", "qualification", "test"):
                        raise ValueError("QC/assessment role cannot enter development fit")
                if row.get("is_qc", False):
                    raise ValueError("QC sample cannot enter fit")
                group = _group(row, dataset)
                if group in forbidden:
                    raise ValueError("reserved validation group must not enter V9 development")
                groups[split].add(group)
                counts[branch + "_" + split] += 1
                for field in ("features", "no_dag_features", "shuffled_features"):
                    _finite_vector(row[field], len(FEATURE_NAMES), field)
                if branch == "rank":
                    pool_id, candidate_id = row.get("pool_id"), row.get("candidate_id")
                    if not isinstance(pool_id, str) or not pool_id or type(candidate_id) is not int or candidate_id < 0:
                        raise ValueError("ranking source pool/candidate identity malformed")
                    identity = (pool_id, candidate_id)
                    if identity in ids:
                        raise ValueError("duplicate ranking candidate identity")
                    ids.add(identity)
                    metadata = pool_metadata.setdefault(pool_id, dict(
                        group=group, split=split, candidate_ids=set()))
                    if metadata["group"] != group or metadata["split"] != split:
                        raise ValueError("pool identity crosses physical group/split")
                    metadata["candidate_ids"].add(candidate_id)
                    success = _success(row["success"])
                    values = _finite_vector(row["backbone_features"], len(backbone.feature_names), "backbone")
                    value, score = float(row["value"]), float(row["backbone_score"])
                    if not np.isfinite(value) or not 0 <= value <= 1 or not np.isfinite(score):
                        raise ValueError("nonfinite official/backbone score")
                    if not np.isclose(values[0], value, atol=1e-10, rtol=1e-10):
                        raise ValueError("official value and backbone feature mismatch")
                    if not np.isclose(backbone.score(values), score, atol=1e-10, rtol=1e-10):
                        raise ValueError("recorded score differs from frozen backbone")
                    outcome = row.get("outcome")
                    if not isinstance(outcome, dict):
                        raise ValueError("explicit supervised outcome record required")
                    if "success" in outcome and _success(outcome["success"]) != success:
                        raise ValueError("outcome/success supervision mismatch")
                    for key in ("executed_steps", "continuation_wam_calls"):
                        if key in outcome and outcome[key] is not None:
                            if not np.isfinite(outcome[key]) or outcome[key] < 0:
                                raise ValueError("invalid observed outcome cost")
                else:
                    row_id = row.get("row_id")
                    if not isinstance(row_id, str) or not row_id or row_id in auxiliary_ids:
                        raise ValueError("duplicate/missing auxiliary source identity")
                    auxiliary_ids.add(row_id)
                    normalize_auxiliary_targets(row, target_names)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                problems.append(prefix + str(error))
    overlap = groups["train"] & groups["val"]
    if overlap:
        problems.append("physical_task_state_group_leakage_rank_plus_auxiliary")
    for pool, metadata in pool_metadata.items():
        candidate_ids = metadata["candidate_ids"]
        if len(candidate_ids) < 2 or candidate_ids != set(range(len(candidate_ids))):
            problems.append("noncontiguous_incomplete_candidate_pool:" + pool)
    expected = dataset.get("training_audit", {})
    for field, actual in (("expected_rank_rows", len(rank)), ("expected_auxiliary_rows", len(auxiliary))):
        if field in expected and expected[field] != actual:
            problems.append(field + "_count_drift")
    if counts["rank_train"] == 0 or counts["rank_val"] == 0:
        problems.append("train_and_val_candidate_pools_both_required")
    return dict(ready=not problems, problems=problems,
        feature_schema=FEATURE_SCHEMA, feature_count=len(FEATURE_NAMES),
        target_names=list(target_names), rank_rows=len(rank), auxiliary_rows=len(auxiliary),
        counts=dict(counts), pool_count=len(pool_metadata),
        group_leakage=bool(overlap), train_groups=sorted(groups["train"]), val_groups=sorted(groups["val"]),
        deployment_gt_leakage=dataset.get("training_audit", {}).get("deployment_gt_leakage", True),
        outcome_fields_are_supervision_only=True)


def _training_rows(dataset, *, masked):
    rank, auxiliary = [], []
    for row in dataset["rank_rows"]:
        if row["split"] != "train":
            continue
        features = mask_attribution_features(row["features"]) if masked else np.asarray(row["features"])
        rank.append(dict(pool_id=row["pool_id"], candidate_id=row["candidate_id"],
            split="train", features=features, backbone_score=row["backbone_score"], success=row["success"]))
    for row in dataset["auxiliary_rows"]:
        if row["split"] != "train":
            continue
        targets, masks = normalize_auxiliary_targets(row, dataset["target_names"])
        features = mask_attribution_features(row["features"]) if masked else np.asarray(row["features"])
        auxiliary.append(dict(row_id=row["row_id"], split="train", features=features,
                              targets=targets, masks=masks))
    return rank, auxiliary


def _select_precomputed(rows, scores, margin):
    # Identical unfiltered wrapper in every arm: no GT success, cost or labels
    # are passed to this decision function. Hard gates are NOT evaluated here.
    decisions = [CandidateDecision(row["candidate_id"], True, float(row["value"]),
        float(scores[row["candidate_id"]]), (), {"offline_ungated_replay": 1.}) for row in rows]
    features = {candidate_id: np.asarray([score]) for candidate_id, score in scores.items()}
    class ScoreAdapter:
        switch_margin = margin
        @staticmethod
        def score(values):
            return float(values[0])
    selected, _ = select_with_utility(decisions, features, ScoreAdapter())
    if selected is None:
        raise RuntimeError("unfiltered audited pool unexpectedly had no candidate")
    return selected.candidate_id


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _comparison(records, method, reference):
    common = [record for record in records if record["methods"][method]["success"]
              and record["methods"][reference]["success"]]
    costs = {}
    for key in ("executed_steps", "continuation_wam_calls"):
        paired = [(record["methods"][method][key], record["methods"][reference][key])
                  for record in common if record["methods"][method][key] is not None
                  and record["methods"][reference][key] is not None]
        costs[key] = dict(paired_observed_count=len(paired), missing_count=len(common) - len(paired),
            method_mean=_mean_or_none([pair[0] for pair in paired]),
            reference_mean=_mean_or_none([pair[1] for pair in paired]),
            mean_delta=_mean_or_none([pair[0] - pair[1] for pair in paired]))
    return dict(
        improvements=sum(record["methods"][method]["success"] and not record["methods"][reference]["success"]
                         for record in records),
        success_harms=sum(not record["methods"][method]["success"] and record["methods"][reference]["success"]
                         for record in records),
        selection_switches=sum(record["methods"][method]["candidate_id"] != record["methods"][reference]["candidate_id"]
                               for record in records),
        common_success_count=len(common), common_success_costs=costs)


def evaluate_pools(rows, backbone, full_model, masked_model):
    pools = defaultdict(list)
    for row in rows:
        pools[row["pool_id"]].append(row)
    records = []
    residual_stats = {method: dict(outside_train_support=0, nonzero_corrections=0,
                                  corrections=[]) for method in METHODS[2:]}
    contrasts = {method: [] for method in METHODS[3:6]}
    for pool_id, candidates in sorted(pools.items()):
        candidates = sorted(candidates, key=lambda row: row["candidate_id"])
        scored = {"value_only": {}, "candidate_backbone": {}}
        features = {}
        for row in candidates:
            candidate_id = row["candidate_id"]
            scored["value_only"][candidate_id] = float(row["value"])
            scored["candidate_backbone"][candidate_id] = float(row["backbone_score"])
            features[candidate_id] = dict(
                full=row["features"], masked_attribution_same_model=mask_attribution_features(row["features"]),
                shuffled_attribution_same_model=row["shuffled_features"],
                no_dependency_same_model=row["no_dag_features"],
                retrained_masked_model=mask_attribution_features(row["features"]))
        for method in METHODS[2:]:
            model = masked_model if method == "retrained_masked_model" else full_model
            scored[method] = {}
            for row in candidates:
                candidate_id = row["candidate_id"]
                parts = model.score_decomposition(features[candidate_id][method], row["backbone_score"])
                correction = parts["cause_recovery_residual"]
                scored[method][candidate_id] = parts["total_score"]
                residual_stats[method]["outside_train_support"] += parts["outside_train_support"]
                residual_stats[method]["nonzero_corrections"] += abs(correction) > 1e-12
                residual_stats[method]["corrections"].append(correction)
        for method in contrasts:
            contrasts[method].extend(scored["full"][row["candidate_id"]] - scored[method][row["candidate_id"]]
                                     for row in candidates)
        outcomes = {}
        for method in METHODS:
            selected = _select_precomputed(candidates, scored[method], backbone.switch_margin)
            row = next(row for row in candidates if row["candidate_id"] == selected)
            outcome = row["outcome"]
            outcomes[method] = dict(candidate_id=selected, success=bool(row["success"]),
                executed_steps=outcome.get("executed_steps"),
                continuation_wam_calls=outcome.get("continuation_wam_calls"))
        successes = [bool(row["success"]) for row in candidates]
        record = dict(pool_id=pool_id,
            suite=candidates[0].get("suite", candidates[0].get("identity", {}).get("suite", "unspecified")),
            task=candidates[0]["task"], state=candidates[0]["state"],
            candidate_count=len(candidates), covered=any(successes), mixed=any(successes) and not all(successes),
            all_failure=not any(successes), all_success=all(successes), methods=outcomes, scores=scored)
        records.append(record)
    summary = dict(pools=len(records), candidates=len(rows),
        covered_pools=sum(record["covered"] for record in records),
        mixed_pools=sum(record["mixed"] for record in records),
        all_failure_pools=sum(record["all_failure"] for record in records),
        all_success_pools=sum(record["all_success"] for record in records),
        oracle_success_ceiling=sum(record["covered"] for record in records),
        methods={}, residual_statistics={}, attribution_score_contrasts={}, rows=records)
    for method in METHODS:
        successful = [record["methods"][method] for record in records if record["methods"][method]["success"]]
        summary["methods"][method] = dict(
            successes=len(successful), success_rate=len(successful) / len(records) if records else None,
            success_at_selection_given_covered=len(successful) / summary["covered_pools"] if summary["covered_pools"] else None,
            observed_success_steps=_mean_or_none([record["executed_steps"] for record in successful
                                                 if record["executed_steps"] is not None]),
            observed_success_continuation_wam_calls=_mean_or_none([record["continuation_wam_calls"] for record in successful
                                                                  if record["continuation_wam_calls"] is not None]),
            vs_value_only=_comparison(records, method, "value_only"),
            vs_candidate_backbone=_comparison(records, method, "candidate_backbone"),
            vs_retrained_masked=_comparison(records, method, "retrained_masked_model"))
    for method, stats in residual_stats.items():
        corrections = stats.pop("corrections")
        summary["residual_statistics"][method] = {**stats,
            "mean_abs_correction": _mean_or_none([abs(value) for value in corrections]),
            "max_abs_correction": max((abs(value) for value in corrections), default=0.)}
    for method, delta in contrasts.items():
        summary["attribution_score_contrasts"][method] = dict(
            nonzero_candidate_deltas=sum(abs(value) > 1e-12 for value in delta),
            mean_abs_delta=_mean_or_none([abs(value) for value in delta]),
            max_abs_delta=max((abs(value) for value in delta), default=0.))
    summary["by_task"] = {}
    for suite, task in sorted({(record["suite"], record["task"]) for record in records}):
        subset = [record for record in records if (record["suite"], record["task"]) == (suite, task)]
        summary["by_task"][f"{suite}:task{task}"] = dict(
            pools=len(subset), covered_pools=sum(record["covered"] for record in subset),
            mixed_pools=sum(record["mixed"] for record in subset),
            oracle_success_ceiling=sum(record["covered"] for record in subset),
            successes={method: sum(record["methods"][method]["success"] for record in subset)
                       for method in METHODS},
            full_vs_value_only=_comparison(subset, "full", "value_only"),
            full_vs_retrained_masked=_comparison(subset, "full", "retrained_masked_model"))
    return summary


def run_pipeline(input_path: Path, backbone_path: Path, output: Path):
    if output.exists():
        raise RuntimeError("refuse overwriting/resuming a historical recovery training result")
    output.mkdir(parents=True, exist_ok=False)
    _dump(output / "status.json", dict(phase="input_audit", model_alias="V9"))
    try:
        dataset = json.loads(input_path.read_text(encoding="utf-8"))
        backbone = CandidateUtilityModel.from_record(json.loads(backbone_path.read_text(encoding="utf-8")))
        audit = audit_training_input(dataset, backbone)
        _dump(output / "input_audit.json", audit)
        if not audit["ready"]:
            raise ValueError("recovery training input failed provenance/schema audit")
        protocol = dict(
            feature_schema=FEATURE_SCHEMA, feature_names=list(FEATURE_NAMES),
            target_names=dataset["target_names"], model_alias="V9", seed=SEED, epochs=EPOCHS,
            loss_weights=dict(success_first_pairwise=RANK_LOSS_WEIGHT,
                              masked_auxiliary=AUXILIARY_LOSS_WEIGHT, l2=L2_WEIGHT),
            residual_cap=RESIDUAL_CAP, no_validation_tuning=True,
            auxiliary_transition_encoding="unit_interval = 0.5 + 0.5 * signed_delta",
            auxiliary_transition_decoding="signed_delta = 2 * predicted_unit_interval - 1",
            after_target_encoding="identity in [0,1]",
            unobserved_labels_are_masked_not_negative=True,
            identical_selection_wrapper="select_with_utility",
            inherited_switch_margin=backbone.switch_margin,
            hard_gate_not_evaluated_in_this_offline_training_report=True,
            outcome_steps_field="executed_steps",
            outcome_wam_cost_field="continuation_wam_calls",
            costs_describe_saved_candidate_continuations_not_new_full_method_closedloop=True,
            data_sha256=_sha(input_path), backbone_sha256=_sha(backbone_path),
            auto_start_128=False, auto_start_main_experiment=False)
        _dump(output / "training_protocol.json", protocol)
        _dump(output / "status.json", dict(phase="training", model_alias="V9"))
        full_rank, full_aux = _training_rows(dataset, masked=False)
        masked_rank, masked_aux = _training_rows(dataset, masked=True)
        full = fit_recovery_residual(rank_rows=full_rank, auxiliary_rows=full_aux,
            feature_names=FEATURE_NAMES, target_names=dataset["target_names"], seed=SEED, epochs=EPOCHS)
        masked = fit_recovery_residual(rank_rows=masked_rank, auxiliary_rows=masked_aux,
            feature_names=FEATURE_NAMES, target_names=dataset["target_names"], seed=SEED, epochs=EPOCHS)
        _dump(output / "recovery_model_v9.json", full.to_record())
        _dump(output / "retrained_masked_model_v9.json", masked.to_record())
        _dump(output / "training_curves.json", dict(full=list(full.training_curve),
                                                    retrained_masked=list(masked.training_curve)))
        evaluation_rows = [{**row, "suite": _group(row, dataset)[0]} for row in dataset["rank_rows"]]
        train = evaluate_pools([row for row in evaluation_rows if row["split"] == "train"],
                               backbone, full, masked)
        validation = evaluate_pools([row for row in evaluation_rows if row["split"] == "val"],
                                    backbone, full, masked)
        checks = dict(
            train_strictly_better_than_value_only=train["methods"]["full"]["successes"] > train["methods"]["value_only"]["successes"],
            val_noninferior_to_value_only=validation["methods"]["full"]["successes"] >= validation["methods"]["value_only"]["successes"],
            train_zero_success_harm_vs_value_only=train["methods"]["full"]["vs_value_only"]["success_harms"] == 0,
            train_zero_success_harm_vs_candidate_backbone=train["methods"]["full"]["vs_candidate_backbone"]["success_harms"] == 0,
            val_zero_success_harm_vs_candidate_backbone=validation["methods"]["full"]["vs_candidate_backbone"]["success_harms"] == 0,
            val_zero_success_harm_vs_value_only=validation["methods"]["full"]["vs_value_only"]["success_harms"] == 0)
        gate = dict(**checks, passed=all(checks.values()))
        extra = {}
        for label, report in (("train", train), ("val", validation)):
            comparison = report["methods"]["full"]["vs_retrained_masked"]
            extra[label] = dict(**comparison,
                success_delta=report["methods"]["full"]["successes"] - report["methods"]["retrained_masked_model"]["successes"],
                positive_zero_harm_increment_observed=comparison["improvements"] > 0 and comparison["success_harms"] == 0)
        report = dict(
            role="offline_development_same_candidate_pool_replay_not_real_closedloop",
            train=train, validation=validation, development_gate=gate,
            attribution_increment_vs_matched_retrained_masked=extra,
            ordinary_backbone_gain_is_not_attribution_gain=True,
            zero_residual_is_legal_not_proof_of_coupling_benefit=True,
            no_failure_pool_removed=True, independent_generalization_established=False,
            full_hard_gate_deployment_verified=False, no_new_confirmation=True,
            no_auto_start_128=True, no_auto_start_main_experiment=True)
        _dump(output / "development_report.json", report)
        _dump(output / "status.json", dict(phase="training_complete", model_alias="V9",
            development_gate_passed=gate["passed"], independent_coupling_benefit_established=False))
        hashes = [f"{_sha(path)}  {path.name}" for path in sorted(output.glob("*.json"))]
        (output / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
        return report
    except Exception as error:
        _dump(output / "failure.json", dict(error_type=type(error).__name__, message=str(error),
                                             no_protocol_or_gate_modified=True))
        _dump(output / "status.json", dict(phase="failed", model_alias="V9", training_completed=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_pipeline(args.input, args.backbone, args.output)
    print(json.dumps(dict(development_gate=report["development_gate"],
                         attribution_increment=report["attribution_increment_vs_matched_retrained_masked"]),
                     ensure_ascii=False))
    if not report["development_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
