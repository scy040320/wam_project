#!/usr/bin/env python3
"""Sealed D20--D21 comparison on an already executed candidate pool.

``value_only`` reproduces official Cosmos argmax(value). ``outcome_oracle``
is a post-outcome ceiling: success, then fewer steps, then fewer WAM calls.
``learned`` uses frozen D18-v18i attribution, V6 candidate-effect parsing,
belief update and hard gating, with value only as the feasible-set tie-break.

This analyzer never generates candidates and never tunes thresholds or rules.
Ground-truth labels and candidate outcomes are invisible to the learned path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from wam_reranking import (
    AttributionOutput,
    CoarseCause,
    ConsistencyFactor,
    EvidenceQuality,
    SelectorMode,
    TaskBinding,
    TriValue,
    initial_belief,
    localize_candidate_visual_evidence,
    select_hard_gate_value_tiebreak,
    select_value_only,
)
from wam_reranking.relation_evidence import RELATION_BINDINGS
from wam_reranking.target_localization import CLIPSegTargetLocalizer


TASK_BINDINGS = {
    50: TaskBinding(
        50,
        "put both the alphabet soup and the tomato sauce in the basket",
        "alphabet_soup_1",
        "basket",
    )
}
ANOMALY_FACTORS = (
    "visual_evidence_corrupted",
    "object_or_environment_state_changed",
    "execution_or_contact_deviated",
    "cross_view_conflict",
)


def _entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probabilities.values())


def _state(probability: float) -> tuple[TriValue, float]:
    probability = float(np.clip(probability, 0.0, 1.0))
    return (TriValue.TRUE if probability >= 0.5 else TriValue.FALSE,
            max(probability, 1.0 - probability))


def learned_attribution(record: Mapping[str, Any], source: str) -> AttributionOutput:
    """Convert frozen D18 anomaly heads into the D19 consistency contract."""
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
        # D18-v18i has no stage-consistency head; do not invent an anomaly.
        ConsistencyFactor.TASK_STAGE_CONSISTENT.value: 1.0,
        # Preserve the frozen unknown routing threshold in this binary field.
        ConsistencyFactor.CAUSE_RESOLVED.value:
            1.0 if unknown_probability < unknown_threshold else 0.0,
    }
    states, confidences = {}, {}
    for name, probability in factor_probs.items():
        states[name], confidences[name] = _state(probability)

    predicted = CoarseCause(str(record["predicted"]))
    conflict = factors["cross_view_conflict"] >= 0.5
    if conflict or unknown_probability >= unknown_threshold:
        predicted = CoarseCause.UNKNOWN
    class_probs = {name: float(value) for name, value in record["class_probs"].items()}
    return AttributionOutput(
        factor_probs=factor_probs,
        class_probs=class_probs,
        projected_cause=predicted,
        confidence=float(np.clip(record["confidence"], 0.0, 1.0)),
        entropy=_entropy(class_probs),
        evidence_quality=EvidenceQuality(
            primary_reliable=bool(record["primary_available"]) and observation_consistent >= 0.5,
            wrist_reliable=bool(record["wrist_available"]) and observation_consistent >= 0.5,
            execution_reliable=bool(record["action_record_available"]),
            cross_view_conflict=conflict,
        ),
        source_block_id=source,
        factor_states=states,
        factor_confidences=confidences,
    )


def outcome_key(candidate: Mapping[str, Any]) -> tuple[int, float, float, float]:
    outcome = candidate["outcome"]
    return (int(bool(outcome["success"])), -float(outcome["executed_steps"]),
            -float(outcome["continuation_wam_calls"]), -float(candidate["candidate_id"]))


def evaluated_key(outcome: Mapping[str, Any] | None) -> tuple[int, float, float]:
    if outcome is None:
        return (-1, 0.0, 0.0)
    return (int(bool(outcome["success"])), -float(outcome["executed_steps"]),
            -float(outcome["continuation_wam_calls"]))


def load_rgb(path: Path) -> Image.Image:
    array = np.load(path)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid RGB array: {path}")
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def serialize_selection(selection: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected = selection.selected_candidate_id
    return {
        "selected_candidate_id": selected,
        "fallback": selection.fallback,
        "accepted": [d.candidate_id for d in selection.decisions if d.accepted],
        "rejected": {str(d.candidate_id): list(d.rejection_reasons)
                     for d in selection.decisions if not d.accepted},
        "decision_components": {str(d.candidate_id): dict(d.components)
                                for d in selection.decisions},
        "outcome": None if selected is None else candidates[selected]["outcome"],
    }


def aggregate(scenarios: list[dict[str, Any]], track: str) -> dict[str, Any]:
    outcomes = [row[track]["outcome"] for row in scenarios]
    evaluated = [o for o in outcomes if o is not None]
    successes = [o for o in outcomes if o is not None and o["success"]]
    covered = [row for row in scenarios if row["candidate_pool_has_success"]]
    covered_successes = sum(
        row[track]["outcome"] is not None and row[track]["outcome"]["success"]
        for row in covered
    )
    return {
        "selected_candidates": sum(row[track]["selected_candidate_id"] is not None for row in scenarios),
        "fallbacks": sum(row[track]["fallback"] is not None for row in scenarios),
        "evaluated_outcomes": len(evaluated),
        "observed_successes": len(successes),
        "observed_failures": len(evaluated) - len(successes),
        "selected_candidate_success_rate": len(successes) / len(evaluated) if evaluated else None,
        "overall_success_lower_bound": len(successes) / len(scenarios) if scenarios else 0.0,
        "successes_when_pool_covered": covered_successes,
        "covered_scenarios": len(covered),
        "conditional_success_lower_bound": covered_successes / len(covered) if covered else None,
        "mean_steps_when_success": float(np.mean([o["executed_steps"] for o in successes])) if successes else None,
        "mean_calls_when_success": float(np.mean([o["continuation_wam_calls"] for o in successes])) if successes else None,
    }


def paired_delta(scenarios: list[dict[str, Any]], track: str) -> dict[str, Any]:
    improvements = harms = ties = unobserved = baseline_success_to_fallback = 0
    step_deltas, call_deltas = [], []
    for row in scenarios:
        base, other = row["value_only"]["outcome"], row[track]["outcome"]
        if other is None:
            unobserved += 1
            baseline_success_to_fallback += int(base is not None and bool(base["success"]))
            continue
        a, b = evaluated_key(base), evaluated_key(other)
        improvements += b > a
        harms += b < a
        ties += b == a
        if base is not None and other is not None and base["success"] and other["success"]:
            step_deltas.append(float(other["executed_steps"]) - float(base["executed_steps"]))
            call_deltas.append(float(other["continuation_wam_calls"]) - float(base["continuation_wam_calls"]))
    return {
        "strict_improvements": improvements,
        "strict_harms": harms,
        "ties": ties,
        "unobserved_fallback_outcomes": unobserved,
        "baseline_success_replaced_by_fallback": baseline_success_to_fallback,
        "mean_step_delta_when_both_succeed": float(np.mean(step_deltas)) if step_deltas else None,
        "mean_call_delta_when_both_succeed": float(np.mean(call_deltas)) if call_deltas else None,
        "both_success_pairs": len(step_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    learned_payload = json.loads((args.root / "learned_predictions.json").read_text())
    for record in learned_payload["records"]:
        record["unknown_threshold"] = learned_payload["unknown_threshold"]
    prediction_index = {(int(r["task_id"]), int(r["state"]), int(r["moment_id"])): r
                        for r in learned_payload["records"]}
    expected = {(50, state, 0) for state in args.states}
    if set(prediction_index) != expected:
        raise RuntimeError(f"learned evidence mismatch: got={sorted(prediction_index)} expected={sorted(expected)}")

    localizer = CLIPSegTargetLocalizer(
        device="cuda", model_path=str(Path.cwd().parent / "model_cache/clipseg-rd64-refined"),
        local_files_only=True,
    )
    scenarios: list[dict[str, Any]] = []
    for state in args.states:
        result_path = args.source_root / "stress_outcomes" / f"task50_state{state}_moment0" / "results.json"
        rows = json.loads(result_path.read_text())
        if len(rows) != 1:
            raise RuntimeError(f"expected one scenario in {result_path}")
        row, candidates = rows[0], rows[0]["candidates"]
        if len(candidates) != 8 or [c["candidate_id"] for c in candidates] != list(range(8)):
            raise RuntimeError(f"non-frozen K=8 pool in {result_path}")
        outcome_root = result_path.parent
        actions = [np.load(outcome_root / c["actions_path"]) for c in candidates]
        values = [float(c["value"]) for c in candidates]
        current_primary = load_rgb(outcome_root / row["current_primary_path"])
        current_wrist = load_rgb(outcome_root / row["current_wrist_path"])
        binding = RELATION_BINDINGS[(50, 0)]
        visual_evidence = [localize_candidate_visual_evidence(
            localizer=localizer, current_primary=current_primary, current_wrist=current_wrist,
            predicted_primary=load_rgb(outcome_root / c["predicted_primary_path"]),
            predicted_wrist=load_rgb(outcome_root / c["predicted_wrist_path"]),
            target_prompt=binding.subject, anchor_prompt=binding.anchor, relation=binding.relation,
        ) for c in candidates]

        value = serialize_selection(select_value_only(values), candidates)
        record = prediction_index[(50, state, 0)]
        attr = learned_attribution(record, f"task50_state{state}:learned")
        learned = serialize_selection(select_hard_gate_value_tiebreak(
            mode=SelectorMode.LEARNED_HARD_GATE,
            belief=initial_belief(50, bindings=TASK_BINDINGS), attribution=attr, block_index=3,
            candidate_actions=actions, official_values=values, visual_evidence=visual_evidence,
            score_weights=None,
        ), candidates)
        best = max(candidates, key=outcome_key)
        oracle = {"selected_candidate_id": int(best["candidate_id"]), "fallback": None,
                  "accepted": list(range(8)), "rejected": {}, "decision_components": {},
                  "outcome": best["outcome"]}
        scenarios.append({
            "episode": row["episode"], "task_id": 50, "state": state, "moment_id": 0,
            "condition": row["condition"],
            "candidate_pool_has_success": any(c["outcome"]["success"] for c in candidates),
            "successful_candidate_ids": [c["candidate_id"] for c in candidates if c["outcome"]["success"]],
            "candidate_visual_evidence": [vars(item) for item in visual_evidence],
            "learned_attribution": record,
            "learned_belief_input": {
                "projected_cause": attr.projected_cause.value,
                "factor_probs": dict(attr.factor_probs),
                "factor_states": {k: v.value for k, v in attr.factor_states.items()},
                "evidence_quality": vars(attr.evidence_quality),
            },
            "value_only": value, "outcome_oracle": oracle, "learned": learned,
        })

    covered = sum(row["candidate_pool_has_success"] for row in scenarios)
    summary = {
        "protocol": "d20_d21_task50_frozen_k8_threeway_v1",
        "source_root": str(args.source_root), "states": args.states,
        "scenarios": len(scenarios), "candidate_count": len(scenarios) * 8,
        "candidate_pool_coverage": {"covered": covered, "total": len(scenarios),
                                    "rate": covered / len(scenarios)},
        "tracks": {name: aggregate(scenarios, name)
                   for name in ("value_only", "outcome_oracle", "learned")},
        "paired_vs_value_only": {name: paired_delta(scenarios, name)
                                 for name in ("outcome_oracle", "learned")},
        "interpretation_contract": {
            "outcome_oracle_is_post_outcome_ceiling": True,
            "learned_uses_outcomes_or_ground_truth_labels": False,
            "learned_weights_calibrated_in_this_pool": False,
            "candidate_generation_or_rerun": False,
            "d18_v18i_is_deployment_ready": False,
        },
    }
    (args.root / "paired_scenarios.json").write_text(json.dumps(scenarios, indent=2) + "\n")
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
