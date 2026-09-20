"""Freeze the CF-WAM v1 development protocol before training or comparison.

The manifest is deterministic: each initial state belongs to exactly one split
and every task/phase/condition combination is listed once.  It intentionally
contains no simulator state and can be generated on Windows without LIBERO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path



def parse_protocol(path: Path) -> dict[str, object]:
    """Parse the deliberately small, frozen protocol YAML without extra deps."""
    text = path.read_text(encoding="utf-8")
    tasks = [line.split("-", 1)[1].strip() for line in text.splitlines() if line.startswith("  - libero_")]
    def scalar_list(key: str) -> list[str]:
        line = next(line for line in text.splitlines() if line.startswith(f"{key}:"))
        raw = line.split(":", 1)[1].strip().strip("[]")
        return [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]
    def number_list(key: str) -> list[int]:
        line = next(line for line in text.splitlines() if line.strip().startswith(f"{key}:"))
        raw = line.split(":", 1)[1].strip().strip("[]")
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return {
        "development_tasks": tasks,
        "phases": scalar_list("phases"),
        "conditions": scalar_list("conditions"),
        "matched_initial_states_per_task_phase_condition": int(next(line.split(":", 1)[1] for line in text.splitlines() if line.startswith("matched_initial_states_per_task_phase_condition:"))),
        "split_by_episode_id": {"train": number_list("train"), "validation": number_list("validation"), "test": number_list("test")},
        "online_policy": {"action_prefix": 4, "check_every_steps": 4},
        "offline_truth_policy": "simulator_offline_only",
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or root / "outputs" / "d15_protocol_freeze_v1"
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "development_protocol.yaml"
    protocol = parse_protocol(config_path)
    split_lookup = {
        episode: split
        for split, episodes in protocol["split_by_episode_id"].items()
        for episode in episodes
    }
    expected = protocol["matched_initial_states_per_task_phase_condition"]
    if sorted(split_lookup) != list(range(expected)):
        raise ValueError("The split must cover each matched initial state exactly once.")
    if len(set(split_lookup)) != expected:
        raise ValueError("An initial state appears in more than one split.")

    rows = []
    for task in protocol["development_tasks"]:
        for episode_id in range(expected):
            for phase in protocol["phases"]:
                for condition in protocol["conditions"]:
                    rows.append({
                        "run_id": f"{task}_seed{episode_id:02d}_{phase}_{condition}",
                        "task_id": task,
                        "episode_id": episode_id,
                        "split": split_lookup[episode_id],
                        "phase": phase,
                        "condition": condition,
                        "action_prefix": protocol["online_policy"]["action_prefix"],
                        "check_every_steps": protocol["online_policy"]["check_every_steps"],
                    })
    expected_total = len(protocol["development_tasks"]) * expected * len(protocol["phases"]) * len(protocol["conditions"])
    if len(rows) != expected_total:
        raise AssertionError("Unexpected manifest size.")

    fieldnames = list(rows[0])
    with (output / "development_manifest_v1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hash_targets = [config_path, *(sorted((root / "configs").glob("libero_task_*.yaml"))), *(sorted((root / "cfwam").glob("*.py")))]
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in hash_targets}
    (output / "configuration_and_code_sha256.json").write_text(
        json.dumps({"algorithm": "sha256", "files": hashes}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    split_counts = {split: sum(row["split"] == split for row in rows) for split in protocol["split_by_episode_id"]}
    (output / "manifest_summary.json").write_text(
        json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_total_records": expected_total,
            "split_counts": split_counts,
            "tasks": protocol["development_tasks"],
            "phases": protocol["phases"],
            "conditions": protocol["conditions"],
            "online_policy": protocol["online_policy"],
            "offline_truth_policy": protocol["offline_truth_policy"],
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    statistics_plan = {
        "frozen_split_rule": "group by task_id and episode_id; no state appears in more than one split",
        "calibration": "choose abstain thresholds once on validation only; do not retune on test",
        "methods": ["A_binary_global_refresh", "B_cause_global_refresh", "C_large_subgraph", "D_counterfactual_minimal_closure"],
        "metrics": ["macro_f1", "per_class_recall", "ece", "mask_iou", "irrelevant_nodes_invalidated", "success_rate", "false_stop_rate", "global_refresh_rate", "recovery_steps", "wam_calls", "mean_latency", "p95_latency", "safety_failure_rate"],
        "reporting": "report per-task values, aggregate mean, sample count, and bootstrap 95% confidence intervals grouped by matched initial state",
    }
    (output / "statistics_protocol_v1.json").write_text(json.dumps(statistics_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": len(rows), "split_counts": split_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
