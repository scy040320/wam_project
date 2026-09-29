#!/usr/bin/env python3
"""Freeze S1 from the preregistered S0 qualification report.

This script deliberately has no access to candidate images, values, or model
features.  It copies only tasks that the frozen S0 analyzer marked qualified,
while excluding every sentinel task.  An empty qualification set is a valid
scientific outcome and prevents S1 from being launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    report = json.loads(args.screen_report.read_text())
    if report["protocol"] != protocol["protocol"]:
        raise RuntimeError("screen report and source protocol do not match")
    if int(report["screen_state"]) != int(protocol["screen"]["state"]):
        raise RuntimeError("screen state changed after preregistration")
    if int(report["k"]) != int(protocol["screen"]["k"]):
        raise RuntimeError("screen K changed after preregistration")

    tasks = {int(row["task_id"]): row for row in protocol["tasks"]}
    sentinels = {task_id for task_id, row in tasks.items() if row.get("sentinel", False)}
    qualified = [int(task_id) for task_id in report["qualified_task_ids"]]
    if len(qualified) != len(set(qualified)):
        raise RuntimeError("duplicate qualified task id")
    unknown = set(qualified) - set(tasks)
    if unknown:
        raise RuntimeError(f"qualified tasks absent from protocol: {sorted(unknown)}")
    leaked = set(qualified) & sentinels
    if leaked:
        raise RuntimeError(f"sentinel tasks cannot enter held-state validation: {sorted(leaked)}")

    held = {
        "protocol": "candidate_pool_held_state_v1",
        "parent_protocol": protocol["protocol"],
        "source_protocol_sha256": sha256(args.protocol),
        "source_screen_report_sha256": sha256(args.screen_report),
        "screen_state": int(protocol["screen"]["state"]),
        "state": int(protocol["held_state"]["state"]),
        "k": int(protocol["held_state"]["k"]),
        "conditions": list(protocol["held_state"]["conditions"]),
        "qualification_rule": protocol["qualification"],
        "qualified_task_ids": qualified,
        "tasks": [tasks[task_id] for task_id in qualified],
        "empty_pool": not qualified,
        "role": "development held-state validation; not independent confirmation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(held, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "qualified_task_ids": qualified,
        "empty_pool": not qualified,
    }, indent=2))


if __name__ == "__main__":
    main()
