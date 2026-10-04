#!/usr/bin/env python3
"""Train the safe-residual ranker with a train-only calibrated, value-preserving residual policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from wam_reranking import CandidateDecision, fit_pairwise_utility, select_with_utility


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_utility_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def enrich_with_gate_audit(scenarios, audit_path: Path):
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or not audit.get("reconstruction_exact"):
        raise RuntimeError("hard-gate audit is not an exact passing reconstruction")
    index = {tuple(row["key"]): row for row in audit["rows"]}
    overrides = 0
    for scenario in scenarios:
        row = index[tuple(scenario["key"])]
        decisions = {int(item["candidate_id"]): item for item in row["decisions"]}
        for candidate in scenario["candidates"]:
            raw = decisions[int(candidate["candidate_id"])]
            decision = CandidateDecision(
                int(raw["candidate_id"]), bool(raw["accepted"]),
                float(raw["official_value"]),
                None if raw["total_score"] is None else float(raw["total_score"]),
                tuple(str(item) for item in raw["rejection_reasons"]),
                {str(k): float(v) for k, v in raw["components"].items()},
            )
            is_value = int(candidate["candidate_id"]) == int(row["value_candidate"])
            if is_value and row["value_rejection_is_epistemic_only"]:
                decision = replace(
                    decision,
                    accepted=True,
                    total_score=decision.official_value,
                    rejection_reasons=(),
                    components={
                        **decision.components,
                        "baseline_preserving_epistemic_override": 1.0,
                        "overridden_epistemic_rejection_count": float(
                            len(raw["rejection_reasons"])
                        ),
                    },
                )
                overrides += 1
            candidate["decision"] = decision
            candidate["accepted"] = decision.accepted
    return overrides


def balanced_pairs(module, scenarios, seed: int = 2108):
    by_task = defaultdict(list)
    for scenario in scenarios:
        by_task[int(scenario["task_id"])].extend(module.preference_pairs([scenario]))
    nonempty = {task: pairs for task, pairs in by_task.items() if pairs}
    if not nonempty:
        raise RuntimeError("no observable preference pairs")
    target = max(len(pairs) for pairs in nonempty.values())
    rng = np.random.default_rng(seed)
    pairs, sampling = [], {}
    for task in sorted(nonempty):
        raw = nonempty[task]
        indices = np.arange(len(raw))
        if len(raw) < target:
            indices = rng.choice(indices, size=target, replace=True)
        pairs.extend(raw[int(index)] for index in indices)
        sampling[str(task)] = {"raw_pairs": len(raw), "balanced_pairs": len(indices)}
    return pairs, sampling


def outcome_key(candidate):
    outcome = candidate["outcome"]
    return (
        int(bool(outcome["success"])),
        -int(outcome["executed_steps"]),
        -int(outcome["continuation_wam_calls"]),
        -int(candidate["candidate_id"]),
    )


def evaluate(model, scenarios):
    rows = []
    for scenario in scenarios:
        candidates = scenario["candidates"]
        decisions = [item["decision"] for item in candidates]
        features = {int(item["candidate_id"]): item["features"] for item in candidates}
        selected_decision, scores = select_with_utility(decisions, features, model)
        by_id = {int(item["candidate_id"]): item for item in candidates}
        selected = None if selected_decision is None else by_id[selected_decision.candidate_id]
        value = max(candidates, key=lambda item: (float(item["value"]), -int(item["candidate_id"])))
        oracle = max(candidates, key=outcome_key)
        executed = selected["outcome"] if selected is not None else scenario.get("fallback_outcome")
        rows.append({
            "key": list(scenario["key"]),
            "covered": any(bool(item["outcome"]["success"]) for item in candidates),
            "selected_candidate": None if selected is None else int(selected["candidate_id"]),
            "selected_success": None if executed is None else bool(executed.get("success", False)),
            "outcome_observed": executed is not None,
            "fallback": scenario.get("fallback") if selected is None else None,
            "accepted_candidates": [item.candidate_id for item in decisions if item.accepted],
            "value_candidate": int(value["candidate_id"]),
            "value_success": bool(value["outcome"]["success"]),
            "oracle_candidate": int(oracle["candidate_id"]),
            "oracle_success": bool(oracle["outcome"]["success"]),
            "selected_score": None if selected is None else float(scores[int(selected["candidate_id"])]),
            "value_score": scores.get(int(value["candidate_id"])),
            "score_delta_vs_value": (
                None if selected is None or int(value["candidate_id"]) not in scores
                else float(scores[int(selected["candidate_id"])] - scores[int(value["candidate_id"])])
            ),
            "baseline_preserving_epistemic_override": bool(
                selected_decision and selected_decision.components.get(
                    "baseline_preserving_epistemic_override", 0.0
                ) >= 0.5
            ),
        })
    covered = [row for row in rows if row["covered"]]
    return {
        "scenarios": len(rows),
        "covered": len(covered),
        "learned_successes": sum(row["selected_success"] is True for row in rows),
        "unobserved_fallbacks": sum(not row["outcome_observed"] for row in rows),
        "value_successes": sum(row["value_success"] for row in rows),
        "oracle_successes": sum(row["oracle_success"] for row in rows),
        "improvements": sum(row["selected_success"] is True and not row["value_success"] for row in rows),
        "harms": sum(row["value_success"] and row["selected_success"] is False for row in rows),
        "rows": rows,
    }


def margin_grid(model, scenarios):
    raw = replace(model, switch_margin=0.0)
    deltas = {0.01}
    for scenario in scenarios:
        accepted = [item for item in scenario["candidates"] if item["decision"].accepted]
        if not accepted:
            continue
        scores = {int(x["candidate_id"]): raw.score(x["features"]) for x in accepted}
        best = max(accepted, key=lambda x: scores[int(x["candidate_id"])])
        anchor = max(accepted, key=lambda x: (float(x["value"]), -int(x["candidate_id"])))
        if int(best["candidate_id"]) != int(anchor["candidate_id"]):
            deltas.add(max(0.01, float(scores[int(best["candidate_id"])] - scores[int(anchor["candidate_id"])])))
    return sorted(deltas)


def calibrate_margin(model, train):
    records = []
    for margin in margin_grid(model, train):
        candidate = replace(model, switch_margin=float(margin))
        report = evaluate(candidate, train)
        records.append({
            "margin": float(margin),
            "successes": report["learned_successes"],
            "improvements": report["improvements"],
            "harms": report["harms"],
            "unobserved_fallbacks": report["unobserved_fallbacks"],
        })
    feasible = [row for row in records if row["harms"] == 0 and row["unobserved_fallbacks"] == 0]
    if not feasible:
        raise RuntimeError("no train-only switch margin achieves zero baseline harm")
    best = sorted(feasible, key=lambda row: (-row["successes"], row["margin"]))[0]
    return replace(
        model,
        schema="d21_value_anchored_safe_residual_v8",
        switch_margin=float(best["margin"]),
    ), records, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--hard-gate-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    module = load_module(args.contract)
    scenarios = module.load_scenarios(args.bundle)
    overrides = enrich_with_gate_audit(scenarios, args.hard_gate_audit)
    train = [item for item in scenarios if int(item["state"]) <= 5]
    val = [item for item in scenarios if int(item["state"]) >= 6]
    if {(x["task_id"], x["state"]) for x in train} & {(x["task_id"], x["state"]) for x in val}:
        raise RuntimeError("task-state leakage")
    pairs, sampling = balanced_pairs(module, train)
    initial_model = fit_pairwise_utility(pairs)
    model, calibration, selected_calibration = calibrate_margin(initial_model, train)
    train_report = evaluate(model, train)
    # Validation is evaluated exactly once after the train-only policy is frozen.
    val_report = evaluate(model, val)
    gate = {
        "train_strictly_exceeds_value_only": train_report["learned_successes"] > train_report["value_successes"],
        "train_zero_success_harm": train_report["harms"] == 0,
        "validation_strictly_exceeds_value_only": val_report["learned_successes"] > val_report["value_successes"],
        "validation_zero_success_harm": val_report["harms"] == 0,
        "train_all_selected_outcomes_observed": train_report["unobserved_fallbacks"] == 0,
        "validation_all_selected_outcomes_observed": val_report["unobserved_fallbacks"] == 0,
        "task_state_group_leakage_absent": True,
        "validation_used_once_after_freeze": True,
        "explicit_false_and_candidate_hard_violations_not_overridden": True,
    }
    gate["passed"] = all(gate.values())
    model_record = model.to_record()
    model_record.update({
        "protocol": "d21_v8_train_only_calibrated_safe_residual_v1",
        "formal_train_pools": len(train),
        "formal_validation_pools": len(val),
        "balanced_training_pairs": len(pairs),
        "task_pair_sampling": sampling,
        "epistemic_value_overrides": overrides,
    })
    model_path = args.output_root / "candidate_utility_model.json"
    model_path.write_text(json.dumps(model_record, indent=2) + "\n")
    report = {
        "protocol": "d21_v8_train_only_calibrated_safe_residual_v1",
        "epistemic_value_overrides": overrides,
        "task_pair_sampling": sampling,
        "margin_calibration_source": "train_only",
        "margin_calibration": calibration,
        "selected_margin": selected_calibration,
        "train": train_report,
        "validation": val_report,
        "development_gate": gate,
        "interpretation_guards": {
            "validation_not_used_to_choose_margin": True,
            "development_validation_is_not_independent_confirmation": True,
            "risk_proxy_is_not_physical_safety_proof": True,
            "all_failed_pools_remain_in_denominator": True,
        },
    }
    report_path = args.output_root / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    hashes = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in (model_path, report_path)
    ]
    (args.output_root / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n")
    print(json.dumps({
        "selected_margin": selected_calibration,
        "train": {k: train_report[k] for k in ("scenarios", "covered", "learned_successes", "value_successes", "oracle_successes", "improvements", "harms")},
        "validation": {k: val_report[k] for k in ("scenarios", "covered", "learned_successes", "value_successes", "oracle_successes", "improvements", "harms")},
        "development_gate": gate,
    }, indent=2))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
