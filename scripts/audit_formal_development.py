"""Read-only integrity audit for the frozen 2,400-record CF-WAM development set.

The audit validates the on-disk four-step contract without using simulator state.
It checks the complete task/seed/phase/condition Cartesian product, required
artifacts, strict query-to-real timestamps, grouped split membership, and the
separation between online records and offline labels.  It never modifies a
record; only the requested JSON report is written.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


TASKS = range(4)
SEEDS = range(40)
PHASES = ("approach", "grasp", "transport")
CONDITIONS = ("normal", "visual_occlusion", "object_shift", "action_noise", "unknown")
REQUIRED_FILES = {
    "primary_current.png", "wrist_current.png", "primary_predicted.png",
    "wrist_predicted.png", "primary_real_t_plus_4.png", "wrist_real_t_plus_4.png",
    "planned_actions.npy", "executed_actions.npy", "action_execution_delta.npy",
    "proprio_current.npy", "proprio_real_t_plus_4.npy", "manifest.json",
    "online_record.json", "offline_label.json",
}
ONLINE_REQUIRED = {
    "run_id", "task_id", "split", "phase", "episode_id", "t_query", "t_real",
    "requested_prefix", "executed_prefix", "value_prediction", "belief_snapshot",
    "primary_current", "wrist_current", "primary_prediction", "wrist_prediction",
    "primary_actual", "wrist_actual", "action_plan", "action_executed",
    "action_execution_delta", "proprio_current", "proprio_actual",
}
NAME = re.compile(r"^task(?P<task>[0-3])_seed(?P<seed>\d{2})_(?P<phase>approach|grasp|transport)_(?P<condition>normal|visual_occlusion|object_shift|action_noise|unknown)$")


def expected_split(seed: int) -> str:
    if seed < 24:
        return "train"
    if seed < 32:
        return "validation"
    return "test"


def expected_names() -> set[str]:
    return {
        f"task{task}_seed{seed:02d}_{phase}_{condition}"
        for task in TASKS for seed in SEEDS for phase in PHASES for condition in CONDITIONS
    }


def audit_record(directory: Path) -> list[str]:
    errors: list[str] = []
    match = NAME.fullmatch(directory.name)
    if not match:
        return ["invalid record directory name"]
    missing = sorted(name for name in REQUIRED_FILES if not (directory / name).is_file())
    if missing:
        return ["missing files: " + ", ".join(missing)]
    empty = sorted(name for name in REQUIRED_FILES if (directory / name).stat().st_size == 0)
    if empty:
        errors.append("empty files: " + ", ".join(empty))
    try:
        online = json.loads((directory / "online_record.json").read_text(encoding="utf-8"))
        offline = json.loads((directory / "offline_label.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return errors + [f"invalid JSON: {exc}"]
    missing_keys = sorted(ONLINE_REQUIRED - online.keys())
    if missing_keys:
        errors.append("missing online keys: " + ", ".join(missing_keys))
        return errors
    task, seed = int(match["task"]), int(match["seed"])
    if online["run_id"] != directory.name or offline.get("run_id") != directory.name:
        errors.append("run_id does not match directory name")
    if online["task_id"] != f"libero_10_task_{task}":
        errors.append("task_id does not match directory task")
    if online["episode_id"] != seed or online["split"] != expected_split(seed):
        errors.append("episode_id or grouped split mismatch")
    if online["phase"] != match["phase"] or offline.get("cause") != match["condition"]:
        errors.append("phase or offline cause mismatch")
    if online["requested_prefix"] != 4 or online["executed_prefix"] != 4:
        errors.append("prefix must be exactly four actions")
    if online["t_real"] != online["t_query"] + online["executed_prefix"]:
        errors.append("t_real is not t_query + executed_prefix")
    if not isinstance(online["belief_snapshot"], dict) or not online["belief_snapshot"]:
        errors.append("belief snapshot absent or empty")
    value = online["value_prediction"]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append("value_prediction is not finite")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    actual = {path.name: path for path in args.root.iterdir() if path.is_dir() and NAME.fullmatch(path.name)}
    expected = expected_names()
    results = []
    for name in sorted(expected & set(actual)):
        errors = audit_record(actual[name])
        results.append({"run_id": name, "status": "PASS" if not errors else "FAIL", "errors": errors})
    counts = Counter(item["status"] for item in results)
    payload = {
        "schema": "cfwam_formal_development_audit_v1",
        "expected_records": len(expected),
        "discovered_record_directories": len(actual),
        "missing_expected_directories": sorted(expected - set(actual)),
        "unexpected_record_directories": sorted(set(actual) - expected),
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "passed_all": counts["PASS"] == len(expected) and not (expected - set(actual)),
        "records": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "expected_records", "discovered_record_directories", "passed", "failed", "passed_all"
    )}, ensure_ascii=False))
    return 0 if payload["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
