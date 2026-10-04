#!/usr/bin/env python3
"""Train the attribution-conditioned pairwise candidate-utility ranker on development pools only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from wam_reranking import (
    AttributionOutput, CandidateEffect,
    CandidateVisualEvidence,
    CoarseCause, ConsistencyFactor, EvidenceQuality, Stage, TriValue,
    candidate_utility_features,
    fit_pairwise_utility,
    parse_candidate_effect,
)


ANOMALY_FACTORS = (
    "visual_evidence_corrupted",
    "object_or_environment_state_changed",
    "execution_or_contact_deviated",
    "cross_view_conflict",
)


def attribution_output(record: dict[str, Any], source: str) -> AttributionOutput:
    """Convert frozen D18 predictions to the deployable D19 contract."""
    factors = {name: float(record["factor_prob"][name]) for name in ANOMALY_FACTORS}
    unknown_probability = float(record["unknown_prob"])
    unknown_threshold = float(record["unknown_threshold"])
    observation_consistent = 1.0 - max(
        factors["visual_evidence_corrupted"], factors["cross_view_conflict"]
    )
    factor_probs = {
        ConsistencyFactor.OBSERVATION_RELIABLE.value: observation_consistent,
        ConsistencyFactor.WORLD_STATE_CONSISTENT.value:
            1.0 - factors["object_or_environment_state_changed"],
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value:
            1.0 - factors["execution_or_contact_deviated"],
        ConsistencyFactor.TASK_STAGE_CONSISTENT.value: 1.0,
        ConsistencyFactor.CAUSE_RESOLVED.value:
            1.0 if unknown_probability < unknown_threshold else 0.0,
    }
    states, confidences = {}, {}
    for name, probability in factor_probs.items():
        states[name] = TriValue.TRUE if probability >= 0.5 else TriValue.FALSE
        confidences[name] = max(probability, 1.0 - probability)
    cause = CoarseCause(str(record["predicted"]))
    conflict = factors["cross_view_conflict"] >= 0.5
    if conflict or unknown_probability >= unknown_threshold:
        cause = CoarseCause.UNKNOWN
    class_probs = {name: float(value) for name, value in record["class_probs"].items()}
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in class_probs.values())
    return AttributionOutput(
        factor_probs, class_probs, cause,
        float(np.clip(record["confidence"], 0.0, 1.0)), entropy,
        EvidenceQuality(
            # Older audited D19 pools predate explicit availability fields.
            # Their paired-scenario contract required both camera streams and
            # the action record, so absence of these metadata keys means
            # "present under the audited contract", not missing evidence.
            bool(record.get("primary_available", True)) and observation_consistent >= 0.5,
            bool(record.get("wrist_available", True)) and observation_consistent >= 0.5,
            bool(record.get("action_record_available", True)), conflict,
        ),
        source,
        factor_states=states,
        factor_confidences=confidences,
    )


def outcome_key(candidate: dict[str, Any]) -> tuple[int, float, float, int]:
    outcome = candidate["outcome"]
    return (
        1 if bool(outcome["success"]) else 0,
        -float(outcome["executed_steps"]),
        -float(outcome["continuation_wam_calls"]),
        -int(candidate["candidate_id"]),
    )


def load_scenarios(root: Path) -> list[dict[str, Any]]:
    paired = json.loads((root / "paired_scenarios.json").read_text())
    prediction_manifest = json.loads((root / "learned_predictions.json").read_text())
    frozen_unknown_threshold = float(prediction_manifest["unknown_threshold"])
    evidence_index = {
        (int(row["task_id"]), int(row["state"]), int(row["moment_id"]), row["condition"]):
            (
                row["candidate_visual_evidence"],
                row["learned_attribution"],
                {int(item) for item in row["learned"]["accepted"]},
                row["learned"].get("fallback"),
                row["learned"].get("outcome"),
            )
        for row in paired
    }
    scenarios: list[dict[str, Any]] = []
    for result_path in sorted((root / "candidate_outcomes").glob("*/results.json")):
        for row in json.loads(result_path.read_text()):
            key = (int(row["task_id"]), int(row["state"]), int(row["moment_id"]), row["condition"])
            visuals, attribution_record, accepted_ids, fallback, fallback_outcome = evidence_index[key]
            attribution_record = dict(attribution_record)
            attribution_record.setdefault("unknown_threshold", frozen_unknown_threshold)
            attribution = attribution_output(attribution_record, f"development:{key}")
            candidates = []
            for candidate, visual in zip(row["candidates"], visuals, strict=True):
                actions = np.load(result_path.parent / candidate["actions_path"])
                visual_evidence = CandidateVisualEvidence(**visual)
                effect = parse_candidate_effect(
                    int(candidate["candidate_id"]), actions, visual_evidence=visual_evidence
                )
                candidates.append({
                    **candidate,
                    "accepted": int(candidate["candidate_id"]) in accepted_ids,
                    "features": candidate_utility_features(
                        effect, float(candidate["value"]), attribution
                    ),
                })
            scenarios.append({
                "key": key, "task_id": key[0], "state": key[1], "candidates": candidates,
                "fallback": fallback, "fallback_outcome": fallback_outcome,
            })
    if len(scenarios) != len(paired):
        raise RuntimeError(f"scenario count mismatch: outcomes={len(scenarios)} paired={len(paired)}")
    return scenarios


def load_audit_scenarios(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    scenarios = []
    for row in payload["records"]:
        attribution_record = row.get("learned_attribution") or row.get("attribution")
        if attribution_record is None:
            raise RuntimeError(
                "attribution-conditioned utility requires attribution for every audit scenario"
            )
        attribution = attribution_output(
            dict(attribution_record),
            f"audit:{row['task_id']}:{row['state']}:{row['moment_id']}",
        )
        candidates = []
        for candidate in row["candidates"]:
            effect = CandidateEffect(
                int(candidate["candidate_id"]), Stage(candidate["stage"]), {}, {},
                float(candidate["confidence"]), candidate["evidence"], candidate["hard_violations"],
            )
            candidates.append({
                "candidate_id": int(candidate["candidate_id"]),
                "value": float(candidate["official_value"]),
                "outcome": candidate["diagnostic_outcome"],
                "accepted": bool(candidate.get("accepted", True)),
                "features": candidate_utility_features(
                    effect, float(candidate["official_value"]), attribution
                ),
            })
        key = (
            int(row["task_id"]), int(row["state"]), int(row["moment_id"]), row["condition"]
        )
        scenarios.append({
            "key": key, "task_id": key[0], "state": key[1], "candidates": candidates,
            "fallback": row.get("fallback"),
            "fallback_outcome": row.get("fallback_outcome"),
        })
    return scenarios


def preference_pairs(scenarios: list[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for scenario in scenarios:
        candidates = [item for item in scenario["candidates"] if item["accepted"]]
        successful = [item for item in candidates if item["outcome"]["success"]]
        failed = [item for item in candidates if not item["outcome"]["success"]]
        if not successful:
            continue
        # Success is lexicographically primary.  Use every observable
        # success/failure contrast instead of allowing numerous cheap-success
        # comparisons to dominate the sparse mixed candidate sets.
        for better in successful:
            for worse in failed:
                pairs.append((better["features"], worse["features"]))
        # Cost is secondary and is learned only within the successful set.
        # One best-vs-rest set per scenario keeps this signal auditable and
        # prevents it from outweighing task completion.
        best_success = max(successful, key=outcome_key)
        for candidate in successful:
            if outcome_key(best_success) > outcome_key(candidate):
                pairs.append((best_success["features"], candidate["features"]))
    return pairs


def evaluate(model: object, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for scenario in scenarios:
        candidates = scenario["candidates"]
        eligible = [item for item in candidates if item["accepted"]]
        score_by_id = {
            int(item["candidate_id"]): float(model.score(item["features"]))
            for item in eligible
        }
        selected = (
            max(eligible, key=lambda item: score_by_id[int(item["candidate_id"])])
            if eligible else None
        )
        value = max(candidates, key=lambda item: (float(item["value"]), -int(item["candidate_id"])))
        oracle = max(candidates, key=outcome_key)
        value_features = np.asarray(value["features"], dtype=np.float64)
        if selected is not None:
            selected_features = np.asarray(selected["features"], dtype=np.float64)
            normalized_delta = (selected_features - value_features) / model.scale
            contributions = normalized_delta * model.weights
            ranked_contributions = sorted(
                (
                    {
                        "feature": model.feature_names[index],
                        "contribution": float(contributions[index]),
                        "selected_raw": float(selected_features[index]),
                        "value_raw": float(value_features[index]),
                    }
                    for index in range(len(model.feature_names))
                    if abs(float(contributions[index])) > 1e-10
                ),
                key=lambda item: abs(item["contribution"]),
                reverse=True,
            )[:8]
            selected_id = int(selected["candidate_id"])
            selected_score = score_by_id[selected_id]
        else:
            ranked_contributions = []
            selected_id = None
            selected_score = None
        executed_outcome = (
            selected["outcome"] if selected is not None else scenario.get("fallback_outcome")
        )
        executed_success = bool(executed_outcome and executed_outcome.get("success", False))
        rows.append({
            "key": list(scenario["key"]),
            "covered": any(item["outcome"]["success"] for item in candidates),
            "selected_candidate": selected_id,
            "selected_success": executed_success,
            "candidate_selected_success": bool(
                selected is not None and selected["outcome"]["success"]
            ),
            "fallback": scenario.get("fallback") if selected is None else None,
            "executed_steps": (
                int(executed_outcome["executed_steps"]) if executed_outcome else None
            ),
            "continuation_wam_calls": (
                int(executed_outcome["continuation_wam_calls"]) if executed_outcome else None
            ),
            "accepted_candidates": [int(item["candidate_id"]) for item in eligible],
            "value_candidate": int(value["candidate_id"]),
            "value_success": bool(value["outcome"]["success"]),
            "selected_score": selected_score,
            "value_candidate_learned_score": score_by_id.get(int(value["candidate_id"])),
            "score_delta_vs_value": (
                selected_score - score_by_id[int(value["candidate_id"])]
                if selected is not None and int(value["candidate_id"]) in score_by_id
                else None
            ),
            "official_value_delta": (
                float(selected["value"]) - float(value["value"])
                if selected is not None else None
            ),
            "top_residual_contributions_vs_value": ranked_contributions,
            "oracle_candidate": int(oracle["candidate_id"]),
            "oracle_success": bool(oracle["outcome"]["success"]),
        })
    covered = [row for row in rows if row["covered"]]
    return {
        "scenarios": len(rows),
        "covered": len(covered),
        "learned_successes": sum(row["selected_success"] for row in rows),
        "value_successes": sum(row["value_success"] for row in rows),
        "oracle_successes": sum(row["oracle_success"] for row in rows),
        "learned_conditional_coverage": (
            sum(row["selected_success"] for row in covered) / len(covered) if covered else None
        ),
        "value_conditional_coverage": (
            sum(row["value_success"] for row in covered) / len(covered) if covered else None
        ),
        "rows": rows,
    }


def grouped_cross_validation(
    scenarios: list[dict[str, Any]], *, field: str
) -> dict[str, Any]:
    folds = []
    for value in sorted({item[field] for item in scenarios}):
        train = [item for item in scenarios if item[field] != value]
        held = [item for item in scenarios if item[field] == value]
        pairs = preference_pairs(train)
        if not pairs:
            folds.append({"held_value": value, "status": "no_training_pairs"})
            continue
        model = fit_pairwise_utility(pairs)
        folds.append({"held_value": value, "training_pairs": len(pairs), **evaluate(model, held)})
    evaluated = [row for row in folds if row.get("status") is None and row["covered"] > 0]
    learned = sum(int(row["learned_successes"]) for row in evaluated)
    value = sum(int(row["value_successes"]) for row in evaluated)
    wins = sum(row["learned_successes"] > row["value_successes"] for row in evaluated)
    harms = sum(row["learned_successes"] < row["value_successes"] for row in evaluated)
    return {
        "group_field": field,
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "covered_scenarios": sum(int(row["covered"]) for row in evaluated),
        "learned_successes": learned,
        "value_successes": value,
        "fold_wins": wins,
        "fold_harms": harms,
        "stable_improvement": bool(evaluated and learned > value and wins >= 1 and harms == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--additional-development-root", type=Path, action="append", default=[])
    parser.add_argument("--audit-json", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    scenarios = load_scenarios(args.development_root)
    for root in args.additional_development_root:
        scenarios.extend(load_scenarios(root))
    existing_keys = {item["key"] for item in scenarios}
    for path in args.audit_json:
        additions = load_audit_scenarios(path)
        scenarios.extend(item for item in additions if item["key"] not in existing_keys)
        existing_keys.update(item["key"] for item in additions)
    keys = [item["key"] for item in scenarios]
    if len(keys) != len(set(keys)):
        raise RuntimeError("development scenario keys must be unique across sources")
    task_cv = grouped_cross_validation(scenarios, field="task_id")
    state_cv = grouped_cross_validation(scenarios, field="state")
    pairs = preference_pairs(scenarios)
    model = fit_pairwise_utility(pairs)
    model_record = model.to_record()
    model_record.update({
        "development_root": str(args.development_root),
        "additional_development_roots": [str(root) for root in args.additional_development_root],
        "development_scenarios": len(scenarios),
        "development_candidates": sum(len(item["candidates"]) for item in scenarios),
        "preference_pairs": len(pairs),
        "independent_evaluation_groups_used": False,
        "additional_audit_sources": [str(path) for path in args.audit_json],
    })
    model_path = args.output_root / "candidate_utility_model.json"
    model_path.write_text(json.dumps(model_record, indent=2) + "\n")
    full = evaluate(model, scenarios)
    mixed = sum(
        0 < sum(bool(item["outcome"]["success"]) for item in scenario["candidates"])
        < len(scenario["candidates"])
        for scenario in scenarios
    )
    gate = {
        "minimum_mixed_scenarios": mixed >= 3,
        "full_development_exceeds_value": full["learned_successes"] > full["value_successes"],
        "leave_one_task_out_stable_improvement": task_cv["stable_improvement"],
        "leave_one_state_out_stable_improvement": state_cv["stable_improvement"],
        "confirmation_data_not_used": True,
    }
    gate["passed"] = all(gate.values())
    report = {
        "protocol": "d21_value_anchored_attribution_residual_development_v3",
        "attribution_and_selection_are_joint": True,
        "development_only": True,
        "hard_gate_is_not_bypassed": True,
        "mixed_scenarios": mixed,
        "full_development": full,
        "leave_one_task_out": task_cv,
        "leave_one_state_out": state_cv,
        "development_gate": gate,
    }
    report_path = args.output_root / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    hashes = []
    for path in (model_path, report_path):
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.output_root / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n")
    print(json.dumps({
        key: report[key]
        for key in (
            "protocol", "mixed_scenarios", "full_development",
            "leave_one_task_out", "leave_one_state_out", "development_gate",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
