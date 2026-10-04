#!/usr/bin/env python3
"""Strict structural audit for the preregistered D25 candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    tasks = [int(item) for item in protocol["candidate_pool"]["tasks"]]
    conditions = list(protocol["candidate_pool"]["conditions"])
    k = int(protocol["candidate_pool"]["k"])
    expected = {(task, state, condition) for task in tasks for state in args.states for condition in conditions}
    rows, errors, hashes = [], [], {}

    for task, state, condition in sorted(expected):
        key = f"task{task}_state{state}_moment0_{condition}"
        evidence = args.root / "evidence" / key
        outcome = args.root / "outcomes" / key
        manifest = evidence / "manifest.json"
        result_path = outcome / "results.json"
        if not manifest.is_file() or not result_path.is_file():
            errors.append({"key": key, "error": "missing_manifest_or_results"})
            continue
        try:
            manifest_payload = json.loads(manifest.read_text())
            result_payload = json.loads(result_path.read_text())
        except Exception as exc:
            errors.append({"key": key, "error": f"json_decode:{exc}"})
            continue
        if len(result_payload) != 1:
            errors.append({"key": key, "error": f"result_rows={len(result_payload)}"})
            continue
        row = result_payload[0]
        candidates = row.get("candidates", [])
        if len(candidates) != k or [c.get("candidate_id") for c in candidates] != list(range(k)):
            errors.append({"key": key, "error": "candidate_ids_or_k"})
            continue
        if str(row.get("condition")) != condition:
            errors.append({"key": key, "error": f"condition={row.get('condition')}"})
            continue
        for candidate in candidates:
            value = float(candidate.get("value", float("nan")))
            outcome_record = candidate.get("outcome", {})
            if not math.isfinite(value):
                errors.append({"key": key, "error": "nonfinite_value"})
            if not isinstance(outcome_record.get("success"), bool):
                errors.append({"key": key, "error": "invalid_success"})
            if int(outcome_record.get("executed_steps", -1)) < 0:
                errors.append({"key": key, "error": "invalid_steps"})
            for field in ("actions_path", "predicted_primary_path", "predicted_wrist_path"):
                path = outcome / str(candidate.get(field, ""))
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append({"key": key, "error": f"missing_{field}"})
        for field in ("current_primary_path", "current_wrist_path"):
            path = outcome / str(row.get(field, ""))
            if not path.is_file() or path.stat().st_size == 0:
                errors.append({"key": key, "error": f"missing_{field}"})
        # Snapshot-fork collector v5 names these entries ``episodes``. Older
        # collectors used ``records``. Accept either schema, but apply the
        # same strict cardinality and identity checks.
        records = manifest_payload.get("episodes", manifest_payload.get("records", []))
        if len(records) != 1:
            errors.append({"key": key, "error": f"manifest_episodes={len(records)}"})
        else:
            episode = records[0]
            expected_label = "normal" if condition == "clean" else condition
            if int(episode.get("task_id", -1)) != task or int(episode.get("state_index", -1)) != state:
                errors.append({"key": key, "error": "manifest_task_or_state"})
            if str(episode.get("condition")) != condition or str(episode.get("label")) != expected_label:
                errors.append({"key": key, "error": "manifest_condition_or_label"})
            if not bool(episode.get("snapshot_fork")):
                errors.append({"key": key, "error": "snapshot_fork_false"})
            for field in ("query_observation_sha256", "planned_actions_sha256", "start_state_sha256"):
                value = str(episode.get(field, ""))
                if len(value) != 64:
                    errors.append({"key": key, "error": f"invalid_{field}"})
        rows.append({
            "key": key,
            "task_id": task,
            "state": state,
            "condition": condition,
            "candidates": len(candidates),
            "successful_candidates": sum(bool(c["outcome"]["success"]) for c in candidates),
        })
        hashes[str(result_path.relative_to(args.root))] = sha256(result_path)
        hashes[str(manifest.relative_to(args.root))] = sha256(manifest)

    missing = len(expected) - len(rows)
    passed = not errors and (not args.require_complete or missing == 0)
    report = {
        "protocol": protocol["protocol"],
        "states": args.states,
        "expected_pools": len(expected),
        "audited_pools": len(rows),
        "candidate_records": sum(item["candidates"] for item in rows),
        "missing_pools": missing,
        "passed": passed,
        "errors": errors,
        "pools": rows,
        "sha256": hashes,
    }
    destination = args.root / ("completion_audit.json" if args.require_complete else "pilot_audit.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("expected_pools", "audited_pools", "candidate_records", "missing_pools", "passed")}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
