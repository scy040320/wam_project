import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_held_state_protocol.py"
COMPETENCE_SCRIPT = ROOT / "scripts" / "analyze_competence_conditioned_pool.py"


class CandidatePoolProtocolTest(unittest.TestCase):
    def run_builder(self, protocol, report):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol_path = root / "protocol.json"
            report_path = root / "report.json"
            output_path = root / "held.json"
            protocol_path.write_text(json.dumps(protocol))
            report_path.write_text(json.dumps(report))
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--protocol", str(protocol_path),
                 "--screen-report", str(report_path), "--output", str(output_path)],
                text=True, capture_output=True,
            )
            return result, json.loads(output_path.read_text()) if output_path.exists() else None

    def base(self):
        protocol = {
            "protocol": "screen-v1", "screen": {"state": 45, "k": 4},
            "held_state": {"state": 46, "k": 8, "conditions": ["clean", "object_shift"]},
            "qualification": {"selection": "1<=success<k"},
            "tasks": [{"task_id": 1}, {"task_id": 2, "sentinel": True}],
        }
        report = {
            "protocol": "screen-v1", "screen_state": 45, "k": 4,
            "qualified_task_ids": [1],
        }
        return protocol, report

    def test_copies_only_qualified_non_sentinel_tasks(self):
        protocol, report = self.base()
        result, held = self.run_builder(protocol, report)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(held["qualified_task_ids"], [1])
        self.assertEqual([row["task_id"] for row in held["tasks"]], [1])
        self.assertFalse(held["empty_pool"])

    def test_empty_pool_is_preserved(self):
        protocol, report = self.base()
        report["qualified_task_ids"] = []
        result, held = self.run_builder(protocol, report)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(held["empty_pool"])

    def test_refuses_sentinel_leakage(self):
        protocol, report = self.base()
        report["qualified_task_ids"] = [2]
        result, held = self.run_builder(protocol, report)
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(held)

    def test_competence_conditioned_pool_keeps_all_stress_strata(self):
        protocol = {
            "protocol": "competence-v1",
            "screen": {"condition": "clean", "k": 4, "minimum_successful_candidates": 3},
            "stress": {
                "condition": "object_shift", "k": 8,
                "primary_non_bimodal_range": [2, 6], "boundary_range": [1, 7],
                "cost_step_range": 3, "cost_call_range": 1,
            },
            "cells": [
                {"task_id": 1, "state": 10},
                {"task_id": 2, "state": 11, "sentinel": True},
            ],
        }

        def scenario(task, state, condition, k, success_ids, steps=None):
            steps = steps or {}
            return [{
                "task_id": task, "state": state, "condition": condition,
                "candidates": [{
                    "candidate_id": i,
                    "outcome": {
                        "success": i in success_ids,
                        "executed_steps": steps.get(i, 20),
                        "continuation_wam_calls": 1,
                    },
                } for i in range(k)],
            }]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clean_root, stress_root = root / "clean", root / "stress"
            for task, state in ((1, 10), (2, 11)):
                name = f"task{task}_state{state}_moment0"
                (clean_root / name).mkdir(parents=True)
                (stress_root / name).mkdir(parents=True)
                (clean_root / name / "results.json").write_text(json.dumps(
                    scenario(task, state, "clean", 4, {0, 1, 2, 3})
                ))
            (stress_root / "task1_state10_moment0" / "results.json").write_text(json.dumps(
                scenario(1, 10, "object_shift", 8, {0, 2, 5})
            ))
            (stress_root / "task2_state11_moment0" / "results.json").write_text(json.dumps(
                scenario(2, 11, "object_shift", 8, set())
            ))
            protocol_path, output_path = root / "protocol.json", root / "report.json"
            protocol_path.write_text(json.dumps(protocol))
            result = subprocess.run([
                sys.executable, str(COMPETENCE_SCRIPT), "--protocol", str(protocol_path),
                "--clean-outcomes", str(clean_root), "--stress-outcomes", str(stress_root),
                "--output", str(output_path),
            ], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output_path.read_text())
            self.assertEqual(report["new_eligible_cells"], [[1, 10]])
            self.assertEqual(report["sentinel_eligible_cells"], [[2, 11]])
            self.assertEqual(report["stress_strata_counts"]["primary_non_bimodal"], 1)
            self.assertEqual(report["stress_strata_counts"]["generator_limited"], 1)


if __name__ == "__main__":
    unittest.main()
