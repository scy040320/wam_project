"""Summarize staged D30 qualification without changing experiment state."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_RECOVERY = {
    "visual_occlusion": "reobserve",
    "object_shift": "local_state_update",
    "action_noise": "local_action_correction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.rglob("online_decision_log.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["path"] = str(path.parent)
        runs.append(payload)
    return runs


def intervention_decision(run: dict) -> dict | None:
    return next((row for row in run["decisions"] if row.get("intervention_injected")), None)


def summarize(runs: list[dict]) -> tuple[dict, list[dict]]:
    details = []
    for run in runs:
        decision = intervention_decision(run)
        condition = run["condition"]
        expected = EXPECTED_RECOVERY.get(condition)
        details.append({
            "task": run["task_id"],
            "seed": run["episode_id"],
            "approach": run["method"],
            "condition": condition,
            "success": bool(run["success"]),
            "safety_stopped": bool(run["safety_stopped"]),
            "intervention_cause": "" if decision is None else decision["cause"],
            "intervention_recovery": "" if decision is None else decision["recovery_action"],
            "expected_recovery": expected or "",
            "intervention_correct": bool(
                decision is not None and (
                    (condition == "unknown" and decision["recovery_action"] in {"safe_stop_global_refresh", "guarded_reobserve"})
                    or (expected is not None and decision["recovery_action"] == expected)
                    or condition == "normal"
                )
            ),
            "guarded_reobserves": int(run.get("guarded_reobserves", 0)),
            "guard_hold_steps": int(run.get("guard_hold_steps", 0)),
            "path": run["path"],
        })
    by_task = defaultdict(lambda: {"runs": 0, "successes": 0, "safety_stops": 0})
    for row in details:
        bucket = by_task[str(row["task"])]
        bucket["runs"] += 1
        bucket["successes"] += int(row["success"])
        bucket["safety_stops"] += int(row["safety_stopped"])
    known = [row for row in details if row["condition"] in EXPECTED_RECOVERY]
    unknown = [row for row in details if row["condition"] == "unknown"]
    clean = [row for row in details if row["condition"] == "normal"]
    summary = {
        "run_count": len(details),
        "success_count": sum(int(row["success"]) for row in details),
        "safety_stop_count": sum(int(row["safety_stopped"]) for row in details),
        "by_task": dict(by_task),
        "condition_counts": dict(Counter(row["condition"] for row in details)),
        "known_intervention_correct": sum(int(row["intervention_correct"]) for row in known),
        "known_intervention_total": len(known),
        "unknown_safety_stops": sum(int(row["safety_stopped"]) for row in unknown),
        "unknown_total": len(unknown),
        "clean_false_safety_stops": sum(int(row["safety_stopped"]) for row in clean),
        "clean_total": len(clean),
    }
    return summary, details


def main() -> int:
    args = parse_args()
    runs = load_runs(args.root)
    summary, details = summarize(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if details:
        with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader(); writer.writerows(details)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
