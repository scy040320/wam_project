import unittest
from recovery_decision.protocol import (ARMS, build_manifest, eligible, group_split,
                                       masked_at, noisy_at, pair_class, select_stages)
from recovery_decision.analysis import summarize


class TreatmentContractTests(unittest.TestCase):
    def records(self):
        return [dict(task=0, seed=0, stage="approach", condition="object_shift",
                     arm=a, success=True, environment_steps=200,
                     actual_new_calls=5, wam_seconds=4.0) for a in ARMS]

    def test_matrix_and_grouping(self):
        rows = build_manifest()["branches"]
        self.assertEqual(len(rows), 576)
        self.assertEqual(len({tuple(r[k] for k in ("task", "seed", "stage", "condition", "arm")) for r in rows}), 576)
        for seed in range(12):
            self.assertEqual({r["split"] for r in rows if r["seed"] == seed}, {group_split(seed)})

    def test_branch_tail(self):
        self.assertTrue(eligible(171))
        self.assertFalse(eligible(172))

    def test_schedule(self):
        self.assertTrue(masked_at(167, 160, "visual_occlusion"))
        self.assertFalse(masked_at(168, 160, "visual_occlusion"))
        self.assertFalse(masked_at(164, 160, "normal"))
        self.assertTrue(noisy_at(163, 160, "action_noise"))
        self.assertFalse(noisy_at(164, 160, "action_noise"))

    def test_alignment(self):
        for n, altered, terminal, expected in (
            (16, False, False, "aligned_endpoint"),
            (16, True, False, "invalid_action_condition_changed"),
            (4, False, False, "discarded_tail"),
            (12, False, True, "terminal_short")):
            self.assertEqual(pair_class(n, altered, terminal), expected)
        with self.assertRaises(ValueError):
            pair_class(17, True, False)

    def test_first_stage_and_missing(self):
        rows = [dict(step=n, distance_m=d, approaching=True,
                     finger_contact=False, closing_command=c)
                for n, d, c in ((160, .12, False), (161, .11, False), (176, .04, True))]
        self.assertEqual(select_stages(rows)["approach"]["step"], 160)
        with self.assertRaises(ValueError):
            select_stages(rows[:1])
        with self.assertRaises(ValueError):
            select_stages(list(reversed(rows)))

    def test_rescue_not_efficiency(self):
        rows = self.records()
        rows[0]["success"] = False
        report = summarize(rows)
        self.assertTrue(report["paired"][0]["rescued"])
        self.assertIsNone(report["paired"][0]["steps_delta"])

    def test_harm_and_joint_success(self):
        rows = self.records()
        rows[1]["success"] = False
        rows[2]["environment_steps"] = 190
        report = summarize(rows)
        self.assertTrue(report["paired"][0]["harmed"])
        self.assertEqual(report["paired"][1]["steps_delta"], -10)

    def test_reject_incomplete_duplicate_bad_labels(self):
        for rows in ([], self.records()[:2], self.records() + self.records()[:1]):
            with self.assertRaises(ValueError):
                summarize(rows)
        rows = self.records()
        rows[0]["success"] = "false"
        with self.assertRaises(ValueError):
            summarize(rows)

    def test_reject_invalid_cost(self):
        rows = self.records()
        rows[0]["wam_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            summarize(rows)


if __name__ == "__main__":
    unittest.main()
