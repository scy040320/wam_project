#!/usr/bin/env python3
"""Verify that the frozen D21-v8 model reproduces training-report decisions via policy.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from wam_reranking import (
    CandidateUtilityModel,
    CandidateVisualEvidence,
    SelectorMode,
    initial_belief,
    select_hard_gate_value_tiebreak,
)


def load_prepare(path: Path):
    spec = importlib.util.spec_from_file_location("d21_v8_prepare", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prepare-script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepare = load_prepare(args.prepare_script)
    model = CandidateUtilityModel.from_record(json.loads(args.model.read_text()))
    paired = json.loads((args.bundle / "paired_scenarios.json").read_text())
    paired_by_key = {
        (int(x["task_id"]), int(x["state"]), int(x["moment_id"]), x["condition"]): x
        for x in paired
    }
    report = json.loads(args.training_report.read_text())
    expected = {
        tuple(row["key"]): row
        for split in ("train", "validation")
        for row in report[split]["rows"]
    }
    rows, mismatches = [], []
    for result_path in sorted((args.source_root / "outcomes").glob("*/results.json")):
        root = result_path.parent
        for outcome_row in json.loads(result_path.read_text()):
            key = (
                int(outcome_row["task_id"]), int(outcome_row["state"]),
                int(outcome_row["moment_id"]), str(outcome_row["condition"]),
            )
            paired_row = paired_by_key[key]
            record = paired_row["learned_attribution"]
            attribution = prepare.attribution_output(record, f"d21_v7:{key}")
            actions = [np.load(root / item["actions_path"]) for item in outcome_row["candidates"]]
            values = [float(item["value"]) for item in outcome_row["candidates"]]
            visuals = [CandidateVisualEvidence(**item) for item in paired_row["candidate_visual_evidence"]]
            selection = select_hard_gate_value_tiebreak(
                mode=SelectorMode.LEARNED_HARD_GATE,
                belief=initial_belief(key[0], bindings=prepare.TASK_BINDINGS),
                attribution=attribution,
                block_index=3,
                candidate_actions=actions,
                official_values=values,
                visual_evidence=visuals,
                utility_model=model,
            )
            if selection.selected_candidate_id is None:
                executed = outcome_row.get("fallback_outcomes", {}).get(selection.fallback)
            else:
                executed = outcome_row["candidates"][selection.selected_candidate_id]["outcome"]
            selected_success = None if executed is None else bool(executed.get("success", False))
            value_id = int(np.argmax(np.asarray(values, dtype=np.float64)))
            value_success = bool(outcome_row["candidates"][value_id]["outcome"]["success"])
            expected_row = expected[key]
            actual = {
                "key": list(key),
                "selected_candidate": selection.selected_candidate_id,
                "fallback": selection.fallback,
                "selected_success": selected_success,
                "value_candidate": value_id,
                "value_success": value_success,
                "epistemic_override": any(
                    item.components.get("baseline_preserving_epistemic_override", 0.0) >= 0.5
                    for item in selection.decisions
                ),
            }
            if (
                actual["selected_candidate"] != expected_row["selected_candidate"]
                or actual["selected_success"] != expected_row["selected_success"]
                or actual["fallback"] != expected_row["fallback"]
            ):
                mismatches.append({"actual": actual, "expected": expected_row})
            rows.append(actual)

    train = [row for row in rows if row["key"][1] <= 5]
    val = [row for row in rows if row["key"][1] >= 6]
    def summary(items):
        return {
            "scenarios": len(items),
            "learned_successes": sum(x["selected_success"] is True for x in items),
            "unobserved_fallbacks": sum(x["selected_success"] is None for x in items),
            "value_successes": sum(x["value_success"] for x in items),
            "improvements": sum(x["selected_success"] is True and not x["value_success"] for x in items),
            "harms": sum(x["value_success"] and x["selected_success"] is False for x in items),
            "epistemic_overrides": sum(x["epistemic_override"] for x in items),
        }
    payload = {
        "passed": len(rows) == 192 and not mismatches,
        "deployment_policy_replay_exact": not mismatches,
        "model_switch_margin": model.switch_margin,
        "train": summary(train),
        "validation": summary(val),
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
