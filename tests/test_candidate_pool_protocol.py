import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_held_state_protocol.py"


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


if __name__ == "__main__":
    unittest.main()
