#!/usr/bin/env python3
"""Classify a frozen K-candidate clean competence screen into fixed strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--outcomes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    expected_k = int(protocol["screen"]["k"])
    expected_tasks = {int(row["task_id"]) for row in protocol["tasks"]}
    sentinel_tasks = {
        int(row["task_id"]) for row in protocol["tasks"] if bool(row.get("sentinel", False))
    }
    rows = []
    for path in sorted(args.outcomes.glob("*/results.json")):
        data = json.loads(path.read_text())
        if len(data) != 1 or data[0]["condition"] != "clean":
            raise RuntimeError(f"screen shard must contain exactly one clean scenario: {path}")
        row = data[0]
        candidates = row["candidates"]
        if len(candidates) != expected_k:
            raise RuntimeError(f"{path} contains {len(candidates)} candidates, expected {expected_k}")
        successful = [item for item in candidates if item["outcome"]["success"]]
        steps = [int(item["outcome"]["executed_steps"]) for item in successful]
        calls = [int(item["outcome"]["continuation_wam_calls"]) for item in successful]
        step_range = max(steps)-min(steps) if steps else 0
        call_range = max(calls)-min(calls) if calls else 0
        if not successful:
            stratum = "generator_limited"
        elif len(successful) < expected_k:
            stratum = "selection_discriminative"
        elif step_range >= 3 or call_range >= 1:
            stratum = "cost_discriminative"
        else:
            stratum = "trivial"
        scientifically_qualified = stratum in {"selection_discriminative", "cost_discriminative"}
        is_sentinel = int(row["task_id"]) in sentinel_tasks
        rows.append({
            "task_id": int(row["task_id"]), "state": int(row["state"]),
            "successful_candidates": len(successful), "success_candidate_ids": [int(x["candidate_id"]) for x in successful],
            "step_range": step_range, "call_range": call_range, "stratum": stratum,
            "diagnostic_qualified": scientifically_qualified,
            "sentinel": is_sentinel,
            "qualified": scientifically_qualified and not is_sentinel,
        })
    found = {row["task_id"] for row in rows}
    if found != expected_tasks:
        raise RuntimeError(f"task mismatch missing={sorted(expected_tasks-found)} extra={sorted(found-expected_tasks)}")
    report = {
        "protocol": protocol["protocol"], "screen_state": protocol["screen"]["state"],
        "k": expected_k, "tasks": sorted(rows, key=lambda row:row["task_id"]),
        "qualified_task_ids": sorted(row["task_id"] for row in rows if row["qualified"]),
        "sentinel_task_ids": sorted(sentinel_tasks),
        "sentinel_diagnostic_qualified": sorted(
            row["task_id"] for row in rows if row["sentinel"] and row["diagnostic_qualified"]
        ),
    }
    report["strata_counts"] = {
        name: sum(row["stratum"] == name for row in rows)
        for name in ("generator_limited", "selection_discriminative", "cost_discriminative", "trivial")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
