#!/usr/bin/env python3
"""Frozen D25--D27 same-pool comparison and ablation report.

All deployable arms see the same K=4 Cosmos pool. Candidate outcomes are read
only after selection and are used solely for evaluation. An outcome oracle is
reported as a ceiling, never as a deployable method.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from wam_reranking import (
    AttributionOutput, CandidateUtilityModel, CoarseCause, ConsistencyFactor,
    DependencyGraph, EvidenceQuality, SelectorMode, TaskBinding, TriValue,
    initial_belief, localize_candidate_visual_evidence,
    select_hard_gate_value_tiebreak, select_value_only,
)
from wam_reranking.relation_evidence import RELATION_BINDINGS
from wam_reranking.target_localization import CLIPSegTargetLocalizer

TASK_BINDINGS = {
    46: TaskBinding(46, "pick up the alphabet soup and put it in the basket", "alphabet_soup_1", "basket_1"),
    57: TaskBinding(57, "pick up the cream cheese and put it in the tray", "cream_cheese_1", "wooden_tray_1"),
}
ANOMALY_FACTORS = (
    "visual_evidence_corrupted", "object_or_environment_state_changed",
    "execution_or_contact_deviated", "cross_view_conflict",
)
EMPTY_GRAPH = DependencyGraph(())


def _entropy(values: Mapping[str, float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in values.values())


def _state(probability: float) -> tuple[TriValue, float]:
    p = float(np.clip(probability, 0.0, 1.0))
    return (TriValue.TRUE if p >= 0.5 else TriValue.FALSE, max(p, 1.0 - p))


def learned_attribution(record: Mapping[str, Any], source: str) -> AttributionOutput:
    factors = {name: float(record["factor_prob"][name]) for name in ANOMALY_FACTORS}
    unknown = float(record["unknown_prob"])
    threshold = float(record["unknown_threshold"])
    observation = 1.0 - max(factors["visual_evidence_corrupted"], factors["cross_view_conflict"])
    probabilities = {
        ConsistencyFactor.OBSERVATION_RELIABLE.value: observation,
        ConsistencyFactor.WORLD_STATE_CONSISTENT.value: 1.0 - factors["object_or_environment_state_changed"],
        ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value: 1.0 - factors["execution_or_contact_deviated"],
        ConsistencyFactor.TASK_STAGE_CONSISTENT.value: 1.0,
        ConsistencyFactor.CAUSE_RESOLVED.value: float(unknown < threshold),
    }
    states, confidences = {}, {}
    for name, value in probabilities.items():
        states[name], confidences[name] = _state(value)
    cause = CoarseCause(str(record["predicted"]))
    conflict = factors["cross_view_conflict"] >= 0.5
    if conflict or unknown >= threshold:
        cause = CoarseCause.UNKNOWN
    class_probs = {str(k): float(v) for k, v in record["class_probs"].items()}
    return AttributionOutput(
        probabilities, class_probs, cause, float(record["confidence"]), _entropy(class_probs),
        EvidenceQuality(
            bool(record["primary_available"]) and observation >= 0.5,
            bool(record["wrist_available"]) and observation >= 0.5,
            bool(record["action_record_available"]), conflict,
        ), source, factor_states=states, factor_confidences=confidences,
    )


def flat_attribution(record: Mapping[str, Any], source: str) -> AttributionOutput:
    """Direct coarse head projected to one cause, without multi-factor structure."""
    class_probs = {str(k): float(v) for k, v in record["class_probs"].items()}
    label = max(class_probs, key=class_probs.get)
    confidence = class_probs[label]
    probabilities = {factor.value: 0.9 for factor in ConsistencyFactor}
    if label == CoarseCause.VISUAL_OCCLUSION.value:
        probabilities[ConsistencyFactor.OBSERVATION_RELIABLE.value] = 0.1
    elif label == CoarseCause.OBJECT_SHIFT.value:
        probabilities[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = 0.1
    elif label == CoarseCause.EXECUTION_CONTACT_DEVIATION.value:
        probabilities[ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value] = 0.1
    elif label == CoarseCause.UNKNOWN.value:
        probabilities[ConsistencyFactor.CAUSE_RESOLVED.value] = 0.1
    return AttributionOutput(
        probabilities, class_probs, CoarseCause(label), confidence, _entropy(class_probs),
        EvidenceQuality(label != CoarseCause.VISUAL_OCCLUSION.value, True, True, False), source,
    )


def ground_truth_attribution(label: str, source: str) -> AttributionOutput:
    """Diagnostic-only perfect cause/factor attribution for the frozen selector.

    This track may use the offline intervention label, but candidate outcomes remain
    hidden until evaluation.  It is an attribution ceiling, not a deployable arm.
    """
    normalized = CoarseCause.NORMAL.value if label == "clean" else str(label)
    cause = CoarseCause(normalized)
    classes = [item.value for item in CoarseCause]
    residual = 0.01 / max(len(classes) - 1, 1)
    class_probs = {name: (0.99 if name == cause.value else residual) for name in classes}
    probabilities = {factor.value: 0.99 for factor in ConsistencyFactor}
    if cause is CoarseCause.VISUAL_OCCLUSION:
        probabilities[ConsistencyFactor.OBSERVATION_RELIABLE.value] = 0.01
    elif cause is CoarseCause.OBJECT_SHIFT:
        probabilities[ConsistencyFactor.WORLD_STATE_CONSISTENT.value] = 0.01
    elif cause is CoarseCause.EXECUTION_CONTACT_DEVIATION:
        probabilities[ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value] = 0.01
    elif cause is CoarseCause.UNKNOWN:
        probabilities[ConsistencyFactor.CAUSE_RESOLVED.value] = 0.01
    states, confidences = {}, {}
    for name, value in probabilities.items():
        states[name], confidences[name] = _state(value)
    return AttributionOutput(
        probabilities, class_probs, cause, 0.99, _entropy(class_probs),
        EvidenceQuality(
            cause is not CoarseCause.VISUAL_OCCLUSION,
            True,
            True,
            False,
        ),
        source,
        factor_states=states,
        factor_confidences=confidences,
    )


def image(path: Path) -> Image.Image:
    array = np.load(path)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid RGB: {path}")
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def selection_record(selection: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    selected = selection.selected_candidate_id
    return {
        "selected_candidate_id": selected,
        "fallback": selection.fallback,
        "accepted": [d.candidate_id for d in selection.decisions if d.accepted],
        "rejected": {str(d.candidate_id): list(d.rejection_reasons) for d in selection.decisions if not d.accepted},
        "components": {str(d.candidate_id): dict(d.components) for d in selection.decisions},
        # Fallback was not executed in D25; never score it as success or failure.
        "outcome": None if selected is None else candidates[selected]["outcome"],
    }


def oracle_record(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(candidates, key=lambda c: (
        int(bool(c["outcome"]["success"])), -float(c["outcome"]["executed_steps"]),
        -float(c["outcome"]["continuation_wam_calls"]), -int(c["candidate_id"]),
    ))
    return {"selected_candidate_id": int(best["candidate_id"]), "fallback": None,
            "accepted": list(range(len(candidates))), "rejected": {}, "components": {},
            "outcome": best["outcome"]}


def aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    outcomes = [row[arm]["outcome"] for row in rows]
    evaluated = [item for item in outcomes if item is not None]
    successes = [item for item in evaluated if item["success"]]
    covered = [row for row in rows if row["candidate_pool_has_success"]]
    covered_success = sum(row[arm]["outcome"] is not None and row[arm]["outcome"]["success"] for row in covered)
    return {
        "scenarios": len(rows), "selected": len(evaluated),
        "fallback_unobserved": len(rows) - len(evaluated), "successes": len(successes),
        "overall_success_lower_bound": len(successes) / len(rows) if rows else 0.0,
        "covered_scenarios": len(covered),
        "conditional_success_lower_bound": covered_success / len(covered) if covered else None,
        "mean_steps_when_success": float(np.mean([x["executed_steps"] for x in successes])) if successes else None,
        "mean_calls_when_success": float(np.mean([x["continuation_wam_calls"] for x in successes])) if successes else None,
    }


def paired(rows: list[dict[str, Any]], arm: str) -> dict[str, int]:
    result = {"improvements": 0, "harms": 0, "ties": 0, "unobserved": 0}
    for row in rows:
        base, other = row["value_only"]["outcome"], row[arm]["outcome"]
        if other is None:
            result["unobserved"] += 1
            continue
        key = lambda x: (-1, 0.0, 0.0) if x is None else (
            int(bool(x["success"])), -float(x["executed_steps"]), -float(x["continuation_wam_calls"])
        )
        if key(other) > key(base): result["improvements"] += 1
        elif key(other) < key(base): result["harms"] += 1
        else: result["ties"] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--utility-model", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.predictions.read_text())
    for item in payload["records"]:
        item["unknown_threshold"] = payload["unknown_threshold"]
    predictions = {}
    for item in payload["records"]:
        condition = str(item["sample_id"]).split("_m0_", 1)[1]
        predictions[(int(item["task_id"]), int(item["state"]), condition)] = item
    utility = CandidateUtilityModel.from_record(json.loads(args.utility_model.read_text()))
    localizer = CLIPSegTargetLocalizer(
        device="cuda", model_path=str(Path.cwd().parent / "model_cache/clipseg-rd64-refined"),
        local_files_only=True,
    )
    rows: list[dict[str, Any]] = []
    for result_path in sorted((args.source_root / "outcomes").glob("*/results.json")):
        data = json.loads(result_path.read_text())
        if len(data) != 1:
            raise RuntimeError(f"expected one scenario: {result_path}")
        source = data[0]; candidates = source["candidates"]
        if len(candidates) != 4 or [int(x["candidate_id"]) for x in candidates] != list(range(4)):
            raise RuntimeError(f"non-frozen K=4 pool: {result_path}")
        root = result_path.parent
        actions = [np.load(root / x["actions_path"]) for x in candidates]
        values = [float(x["value"]) for x in candidates]
        binding = RELATION_BINDINGS[(int(source["task_id"]), int(source["moment_id"]))]
        current_primary, current_wrist = image(root / source["current_primary_path"]), image(root / source["current_wrist_path"])
        visual = [localize_candidate_visual_evidence(
            localizer=localizer, current_primary=current_primary, current_wrist=current_wrist,
            predicted_primary=image(root / x["predicted_primary_path"]),
            predicted_wrist=image(root / x["predicted_wrist_path"]), target_prompt=binding.subject,
            anchor_prompt=binding.anchor, relation=binding.relation,
        ) for x in candidates]
        key = (int(source["task_id"]), int(source["state"]), str(source["condition"]))
        record = predictions[key]
        flat = flat_attribution(record, f"{source['episode']}:flat")
        hierarchical = learned_attribution(record, f"{source['episode']}:hierarchical")
        ground_truth = ground_truth_attribution(str(source["label"]), f"{source['episode']}:ground_truth")
        def choose(attr: AttributionOutput, graph: DependencyGraph):
            return selection_record(select_hard_gate_value_tiebreak(
                mode=SelectorMode.LEARNED_HARD_GATE,
                belief=initial_belief(int(source["task_id"]), bindings=TASK_BINDINGS),
                attribution=attr, block_index=3, candidate_actions=actions,
                official_values=values, visual_evidence=visual, utility_model=utility,
                dependency_graph=graph,
            ), candidates)
        value = selection_record(select_value_only(values), candidates)
        rows.append({
            **{name: source[name] for name in ("episode", "task_id", "state", "moment_id", "condition", "label")},
            "candidate_pool_has_success": any(x["outcome"]["success"] for x in candidates),
            "k1_candidate0": {"outcome": candidates[0]["outcome"]},
            "value_only": value,
            # At a native chunk boundary and under equal query budget, binary replan
            # asks the same query and therefore selects the same value-max candidate.
            "binary_global_replan": {**value, "boundary_equivalence": True},
            "flat_attribution_same_budget": choose(flat, EMPTY_GRAPH),
            "hierarchical_no_dependency": choose(hierarchical, EMPTY_GRAPH),
            "full_hierarchical_belief_dag": choose(hierarchical, __import__("wam_reranking").DEFAULT_GRAPH),
            "ground_truth_attribution_current_selector": choose(
                ground_truth, __import__("wam_reranking").DEFAULT_GRAPH
            ),
            "outcome_oracle": oracle_record(candidates),
            "learned_attribution": record,
        })
    expected = {(task, state, condition) for task in (46, 57) for state in range(10)
                for condition in ("clean", "visual_occlusion", "object_shift", "execution_contact_deviation")}
    actual = {(int(x["task_id"]), int(x["state"]), str(x["condition"])) for x in rows}
    if actual != expected:
        raise RuntimeError(f"D25 matrix incomplete: {len(actual)}/80")
    arms = ("value_only", "binary_global_replan", "flat_attribution_same_budget",
            "hierarchical_no_dependency", "full_hierarchical_belief_dag",
            "ground_truth_attribution_current_selector", "outcome_oracle")
    summary = {
        "protocol": "d25_d27_preregistered_same_pool_v1", "scenarios": len(rows), "k": 4,
        "tasks": [46, 57], "conditions": sorted({x["condition"] for x in rows}),
        "candidate_pool_coverage": sum(x["candidate_pool_has_success"] for x in rows) / len(rows),
        "tracks": {arm: aggregate(rows, arm) for arm in arms},
        "paired_vs_value_only": {arm: paired(rows, arm) for arm in arms[1:]},
        "interpretation_guards": {
            "fallback_outcomes_unexecuted": True,
            "binary_replan_equals_value_only_at_native_boundary": True,
            "outcome_oracle_is_not_deployable": True,
            "ground_truth_attribution_is_diagnostic_only_and_not_deployable": True,
            "d28_gate3_requires_four_tasks_and_is_not_satisfied_by_this_two_task_matrix": True,
        },
    }
    value_successes = summary["tracks"]["value_only"]["successes"]
    learned = summary["tracks"]["full_hierarchical_belief_dag"]
    gt = summary["tracks"]["ground_truth_attribution_current_selector"]
    success_harms = 0
    for row in rows:
        base = row["value_only"]["outcome"]
        other = row["ground_truth_attribution_current_selector"]["outcome"]
        if base is not None and bool(base["success"]) and (other is None or not bool(other["success"])):
            success_harms += 1
    gate_checks = {
        "gt_success_not_below_value_only": gt["successes"] >= value_successes,
        "no_value_success_lost_to_failure_or_unexecuted_fallback": success_harms == 0,
        "material_gain_or_fallback_reduction_vs_learned": (
            gt["successes"] - learned["successes"] >= 3
            or gt["fallback_unobserved"] * 2 <= learned["fallback_unobserved"]
        ),
    }
    summary["ground_truth_expansion_gate"] = {
        "preregistered_before_running": True,
        "checks": gate_checks,
        "passed": all(gate_checks.values()),
        "value_only_successes": value_successes,
        "learned_successes": learned["successes"],
        "learned_fallback_unobserved": learned["fallback_unobserved"],
        "ground_truth_successes": gt["successes"],
        "ground_truth_fallback_unobserved": gt["fallback_unobserved"],
        "value_successes_lost": success_harms,
    }
    by_condition = {}
    for condition in summary["conditions"]:
        subset = [x for x in rows if x["condition"] == condition]
        by_condition[condition] = {arm: aggregate(subset, arm) for arm in arms}
    summary["by_condition"] = by_condition
    (args.output_root / "d25_d27_scenarios.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_root / "d25_d27_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_root / "d25_d27_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("task", "state", "condition", "pool_covered", *arms))
        for row in rows:
            writer.writerow((row["task_id"], row["state"], row["condition"], row["candidate_pool_has_success"],
                             *(row[arm]["selected_candidate_id"] for arm in arms)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
