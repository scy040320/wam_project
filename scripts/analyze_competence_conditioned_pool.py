#!/usr/bin/env python3
"""Audit and stratify a preregistered competence-conditioned pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_one(path: Path, *, condition: str, k: int) -> dict:
    data = json.loads(path.read_text())
    if len(data) != 1 or data[0]["condition"] != condition:
        raise RuntimeError(f"expected one {condition} scenario: {path}")
    if len(data[0]["candidates"]) != k:
        raise RuntimeError(f"expected K={k}: {path}")
    return data[0]


def success_summary(row: dict) -> dict:
    successful = [c for c in row["candidates"] if c["outcome"]["success"]]
    steps = [int(c["outcome"]["executed_steps"]) for c in successful]
    calls = [int(c["outcome"]["continuation_wam_calls"]) for c in successful]
    return {
        "successful_candidates": len(successful),
        "success_candidate_ids": [int(c["candidate_id"]) for c in successful],
        "step_range": max(steps) - min(steps) if steps else 0,
        "call_range": max(calls) - min(calls) if calls else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--clean-outcomes", type=Path, required=True)
    parser.add_argument("--stress-outcomes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    c0 = protocol["screen"]
    c1 = protocol["stress"]
    rows = []
    for cell in protocol["cells"]:
        task, state = int(cell["task_id"]), int(cell["state"])
        name = f"task{task}_state{state}_moment0"
        clean = load_one(args.clean_outcomes / name / "results.json", condition="clean", k=int(c0["k"]))
        clean_stats = success_summary(clean)
        eligible = clean_stats["successful_candidates"] >= int(c0["minimum_successful_candidates"])
        record = {
            "task_id": task, "state": state, "sentinel": bool(cell.get("sentinel", False)),
            "clean": clean_stats, "stress_eligible": eligible,
        }
        stress_path = args.stress_outcomes / name / "results.json" if args.stress_outcomes else None
        if stress_path and stress_path.exists():
            stress = load_one(stress_path, condition=str(c1["condition"]), k=int(c1["k"]))
            stats = success_summary(stress)
            n = stats["successful_candidates"]
            if int(c1["primary_non_bimodal_range"][0]) <= n <= int(c1["primary_non_bimodal_range"][1]):
                stratum = "primary_non_bimodal"
            elif n in {int(c1["boundary_range"][0]), int(c1["boundary_range"][1])}:
                stratum = "boundary_non_bimodal"
            elif n == int(c1["k"]) and (
                stats["step_range"] >= int(c1["cost_step_range"])
                or stats["call_range"] >= int(c1["cost_call_range"])
            ):
                stratum = "cost_discriminative"
            elif n == 0:
                stratum = "generator_limited"
            else:
                stratum = "trivial"
            record["stress"] = stats | {"stratum": stratum}
        rows.append(record)
    report = {
        "protocol": protocol["protocol"],
        "cells": rows,
        "new_eligible_cells": [
            [r["task_id"], r["state"]] for r in rows if r["stress_eligible"] and not r["sentinel"]
        ],
        "sentinel_eligible_cells": [
            [r["task_id"], r["state"]] for r in rows if r["stress_eligible"] and r["sentinel"]
        ],
    }
    report["stress_strata_counts"] = {
        name: sum(r.get("stress", {}).get("stratum") == name for r in rows)
        for name in ("primary_non_bimodal", "boundary_non_bimodal", "cost_discriminative", "generator_limited", "trivial")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
