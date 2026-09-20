"""Strict, read-only acceptance audit for the 15-record CF-WAM v1 smoke batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASES = ("approach", "grasp", "transport")
CONDITIONS = (
    "normal",
    "visual_occlusion",
    "object_shift",
    "action_noise",
    "unknown",
)
REQUIRED_ONLINE_KEYS = {
    "run_id", "task_id", "split", "phase", "episode_id", "t_query", "t_real",
    "requested_prefix", "executed_prefix", "value_prediction", "belief_snapshot",
}
REQUIRED_FILES = {
    "primary_current.png", "wrist_current.png", "primary_predicted.png",
    "wrist_predicted.png", "primary_real_t_plus_4.png", "wrist_real_t_plus_4.png",
    "planned_actions.npy", "executed_actions.npy", "action_execution_delta.npy",
    "proprio_current.npy", "proprio_real_t_plus_4.npy", "manifest.json",
    "online_record.json", "offline_label.json",
}


def audit_record(record: Path, phase: str, condition: str) -> list[str]:
    errors: list[str] = []
    missing = sorted(name for name in REQUIRED_FILES if not (record / name).is_file())
    if missing:
        errors.append("missing files: " + ", ".join(missing))
        return errors
    online = json.loads((record / "online_record.json").read_text(encoding="utf-8"))
    offline = json.loads((record / "offline_label.json").read_text(encoding="utf-8"))
    missing_keys = sorted(REQUIRED_ONLINE_KEYS - online.keys())
    if missing_keys:
        errors.append("missing online keys: " + ", ".join(missing_keys))
    if online.get("phase") != phase:
        errors.append(f"phase={online.get('phase')!r}, expected {phase!r}")
    if online.get("run_id") != record.name:
        errors.append("online run_id does not match directory")
    if offline.get("run_id") != record.name:
        errors.append("offline run_id does not match directory")
    if offline.get("cause") != condition:
        errors.append(f"offline cause={offline.get('cause')!r}, expected {condition!r}")
    if online.get("executed_prefix") != 4 or online.get("requested_prefix") != 4:
        errors.append("prefix must be exactly four actions")
    if online.get("t_real") != online.get("t_query", -99) + online.get("executed_prefix", -99):
        errors.append("t_real is not t_query + executed_prefix")
    parameters = offline.get("intervention_parameters", {})
    if condition == "normal" and offline.get("affected_nodes"):
        errors.append("normal condition must affect no nodes")
    if condition == "visual_occlusion" and parameters.get("occlusion") is not True:
        errors.append("visual_occlusion must record occlusion=true")
    if condition == "object_shift" and parameters.get("shift_delta") is None:
        errors.append("object_shift is missing shift_delta")
    if condition == "action_noise" and parameters.get("noise_delta") is None:
        errors.append("action_noise is missing noise_delta")
    if condition == "unknown" and not offline.get("affected_nodes"):
        errors.append("unknown must retain an offline affected-node label")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    results: list[dict[str, object]] = []
    for phase in PHASES:
        for condition in CONDITIONS:
            name = f"task0_seed00_{phase}_{condition}"
            record = args.root / name
            errors = audit_record(record, phase, condition) if record.is_dir() else ["record directory missing"]
            results.append({"record": name, "status": "PASS" if not errors else "FAIL", "errors": errors})
    passed = sum(item["status"] == "PASS" for item in results)
    payload = {"batch": "task0_seed00", "expected": len(results), "passed": passed, "passed_all": passed == len(results), "records": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("batch", "expected", "passed", "passed_all")}, ensure_ascii=False))
    for item in results:
        print(f"{item['status']:4} {item['record']}" + (f" :: {'; '.join(item['errors'])}" if item["errors"] else ""))
    return 0 if payload["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
