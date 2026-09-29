#!/usr/bin/env python3
"""Validate a candidate-pool protocol against public LIBERO task metadata.

This check does not load Cosmos or execute a policy. It verifies task IDs,
held-state availability, and intervention joint names before costly rollout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from libero.libero import benchmark
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_pool_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--base-collector", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    base = load_module(args.base_collector.resolve())
    cfg = base.build_config(args.snapshot.resolve(), 1)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task_rows = protocol.get("tasks", protocol.get("cells"))
    if not task_rows:
        raise ValueError("protocol must define tasks or cells")
    shared_states = None
    if "held_state" in protocol and "state" in protocol["screen"]:
        shared_states = {int(protocol["screen"]["state"]), int(protocol["held_state"]["state"])}
    elif "candidate_pool" in protocol and "states" in protocol["candidate_pool"]:
        shared_states = {int(value) for value in protocol["candidate_pool"]["states"]}
    elif "candidate_pool" in protocol and "state" in protocol["candidate_pool"]:
        shared_states = {int(protocol["candidate_pool"]["state"])}
    checks = []
    for row in task_rows:
        task_id = int(row["task_id"])
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, language = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        try:
            sim = getattr(env, "env", env).sim
            joints = {sim.model.joint_id2name(index) for index in range(sim.model.njnt)}
            expected_joint = (
                row["target"] if row.get("kind", "rigid_translation") == "articulated_joint"
                else f"{row['target']}_joint0"
            )
            failures = []
            if language != row["instruction"]:
                failures.append("instruction_mismatch")
            states = shared_states if shared_states is not None else {int(row["state"])}
            if max(states) >= len(initial_states):
                failures.append("state_index_out_of_range")
            if expected_joint not in joints:
                failures.append(f"missing_joint:{expected_joint}")
            checks.append({
                "task_id": task_id, "instruction": language,
                "expected_joint": expected_joint, "initial_states": len(initial_states),
                "passed": not failures, "failures": failures,
            })
        finally:
            env.close()
    report = {
        "protocol": protocol["protocol"], "tasks": checks,
        "passed": all(row["passed"] for row in checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
