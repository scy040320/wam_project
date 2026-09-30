import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FormalD21ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(
            (ROOT / "configs" / "d21_formal_896_protocol.json").read_text()
        )

    def test_expected_pool_and_candidate_counts(self):
        p = self.protocol
        pools = len(p["tasks"]) * len(p["states_per_task"]) * len(p["moments"]) * len(p["conditions"])
        self.assertEqual(pools, 896)
        self.assertEqual(p["expected_pools"], pools)
        self.assertEqual(p["expected_candidate_outcomes"], pools * p["k"])

    def test_grouped_split_is_complete_and_disjoint(self):
        split = self.protocol["split"]
        train = set(split["train_states"])
        val = set(split["validation_states"])
        self.assertFalse(train & val)
        self.assertEqual(train | val, set(self.protocol["states_per_task"]))
        self.assertEqual(split["train_pools"] + split["validation_pools"], 896)

    def test_parallel_lanes_are_disjoint_and_complete(self):
        tasks = self.protocol["tasks"]
        lanes = [task for owned in self.protocol["parallel_lane_ownership"].values() for task in owned]
        self.assertEqual(len(lanes), len(set(lanes)))
        self.assertEqual(set(lanes), set(tasks))

    def test_pilot_spans_every_task_once(self):
        pilot = self.protocol["pilot_groups"]
        self.assertEqual(len(pilot), len(self.protocol["tasks"]))
        self.assertEqual({row["task"] for row in pilot}, set(self.protocol["tasks"]))
        expected = len(pilot) * len(self.protocol["moments"]) * len(self.protocol["conditions"])
        self.assertEqual(expected, self.protocol["pilot_gate"]["expected_pools"])

    def test_zero_coverage_is_not_filtered(self):
        self.assertTrue(self.protocol["invariants"]["zero_coverage_pools_remain_in_denominator"])
        self.assertIn("never lower gates", self.protocol["pilot_gate"]["failure_policy"])


if __name__ == "__main__":
    unittest.main()
