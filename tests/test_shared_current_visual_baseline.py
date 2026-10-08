import unittest
from dataclasses import asdict, replace
from unittest.mock import patch
import numpy as np
from wam_reranking import shared_current_visual_baseline as shared
from wam_reranking.candidate_effects import _centroid_geometry, _relation_score, build_candidate_visual_evidence
from wam_reranking.shared_current_visual_baseline import (
    BEFORE_FIELDS, audit_shared_current_pool, build_shared_current_visual_evidence,
)


def peak(x, y, scale=1.):
    a = np.full((8, 8), .01)
    a[y, x] = scale
    return a


class SharedCurrentBaselineTests(unittest.TestCase):
    def data(self, weak=False):
        return dict(current_subject_maps=(peak(1, 2), peak(6, 3)),
            current_anchor_maps=(peak(5, 4), peak(2, 5)),
            current_gripper_maps=(peak(2, 2), peak(5, 3)),
            predicted_subject_maps=(peak(2, 2), np.ones((8, 8)) if weak else peak(5, 3)),
            predicted_anchor_maps=(peak(5, 4), peak(2, 5)),
            predicted_gripper_maps=(peak(3, 2), peak(4, 3)), relation="right_of")

    def test_forecast_quality_cannot_move_current_baseline(self):
        a, da = build_shared_current_visual_evidence(**self.data())
        b, db = build_shared_current_visual_evidence(**self.data(True))
        self.assertTrue(audit_shared_current_pool([a, b], [da, db])["passed"])
        self.assertNotEqual(a.relation_score_after, b.relation_score_after)
        for f in BEFORE_FIELDS:
            self.assertEqual(getattr(a, f), getattr(b, f))

    def test_exposes_historical_candidate_dependent_baseline(self):
        a = build_candidate_visual_evidence(**self.data())
        b = build_candidate_visual_evidence(**self.data(True))
        self.assertNotEqual(a.relation_score_before, b.relation_score_before)

    def test_single_camera_keeps_forecast_and_uses_direct_CURRENT_geometry(self):
        kw = self.data()
        for k in tuple(kw):
            if k != "relation": kw[k] = kw[k][:1]
        a = build_candidate_visual_evidence(**kw)
        b, _ = build_shared_current_visual_evidence(**kw)
        # Historical AFTER arithmetic stays exactly unchanged. BEFORE now
        # intentionally bypasses q*x/q; compare to its direct CURRENT formula,
        # not the old rounded single-view weighted output.
        for name, value in asdict(a).items():
            if name not in BEFORE_FIELDS:
                self.assertEqual(value, getattr(b, name))
        cs, ca, cg = (kw[name][0] for name in
                      ("current_subject_maps", "current_anchor_maps", "current_gripper_maps"))
        rd, ra, _ = _centroid_geometry(cs, ca)
        gd, ga, _ = _centroid_geometry(cs, cg)
        expected = dict(target_anchor_distance_before=rd, target_anchor_affinity_before=ra,
                        target_gripper_distance_before=gd, target_gripper_affinity_before=ga,
                        relation_score_before=_relation_score(cs, ca, kw["relation"]),
                        grasp_support_before=float(np.clip(max(ga, 1.0 - 2.0 * gd), 0.0, 1.0)))
        for name, value in expected.items():
            self.assertEqual(value, getattr(b, name))

    def test_current_changes_do_change_baseline(self):
        kw = self.data(); a, _ = build_shared_current_visual_evidence(**kw)
        kw["current_subject_maps"] = (peak(4, 6), peak(5, 1))
        b, _ = build_shared_current_visual_evidence(**kw)
        self.assertNotEqual(a.relation_score_before, b.relation_score_before)

    def test_no_in_place_writes_and_no_physical_certificate(self):
        kw = self.data(); saved = kw["current_subject_maps"][0].copy()
        _, d = build_shared_current_visual_evidence(**kw)
        np.testing.assert_array_equal(kw["current_subject_maps"][0], saved)
        self.assertFalse(d["physical_certificate"])

    def test_mixed_current_sources_are_rejected(self):
        a, da = build_shared_current_visual_evidence(**self.data())
        kw = self.data(); kw["current_subject_maps"] = (peak(6, 1), peak(1, 6))
        b, db = build_shared_current_visual_evidence(**kw)
        with self.assertRaises(ValueError): audit_shared_current_pool([a, b], [da, db])

    def test_missing_or_nonfinite_maps_rejected(self):
        kw = self.data(); kw["predicted_subject_maps"] = ()
        with self.assertRaises(ValueError): build_shared_current_visual_evidence(**kw)

    def test_before_does_not_read_any_historical_single_view_before_field(self):
        kw = self.data()
        expected, _ = build_shared_current_visual_evidence(**kw)
        historical = shared.build_candidate_visual_evidence
        def altered_single_view(**inputs):
            evidence = historical(**inputs)
            # Mutate only the intermediate historical BEFORE fields; the new
            # CURRENT baseline must not even read them. AFTER remains real.
            return replace(evidence, **{name: .123456 for name in BEFORE_FIELDS})
        with patch.object(shared, "build_candidate_visual_evidence", altered_single_view):
            actual, diagnostic = build_shared_current_visual_evidence(**kw)
        self.assertEqual(actual, expected)
        self.assertFalse(diagnostic["before_uses_historical_candidate_fusion"])

    def test_forecast_dependent_single_view_ULP_cannot_change_exact_before(self):
        # Before v2, forecast index 5 changes affinity by 5.55e-17 and grasp
        # support by 2.22e-16 with this fixed synthetic seed. No tolerance is
        # used: equality remains exact, matching the real RGB failure audit.
        rng = np.random.default_rng(20261009)
        current = {role: tuple(rng.random((16, 16), dtype=np.float32) for _ in range(2))
                   for role in ("subject", "anchor", "gripper")}
        kw = {"current_" + role + "_maps": maps for role, maps in current.items()}
        first, first_diagnostic = build_shared_current_visual_evidence(**kw,
            **{"predicted_" + role + "_maps": maps for role, maps in current.items()},
            relation="articulated")
        evidences, diagnostics = [first], [first_diagnostic]
        for _ in range(6):
            predicted = {role: tuple(rng.random((16, 16), dtype=np.float32) for _ in range(2))
                         for role in current}
            evidence, diagnostic = build_shared_current_visual_evidence(**kw,
                **{"predicted_" + role + "_maps": maps for role, maps in predicted.items()},
                relation="articulated")
            evidences.append(evidence); diagnostics.append(diagnostic)
        self.assertTrue(audit_shared_current_pool(evidences, diagnostics)["passed"])
        for evidence in evidences:
            for name in BEFORE_FIELDS:
                self.assertEqual(getattr(evidence, name), getattr(first, name))
        kw = self.data(); kw["current_subject_maps"][0][0, 0] = np.nan
        with self.assertRaises(ValueError): build_shared_current_visual_evidence(**kw)


if __name__ == "__main__": unittest.main()
