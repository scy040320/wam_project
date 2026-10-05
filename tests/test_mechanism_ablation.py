import copy
import unittest
from unittest.mock import patch

from wam_reranking.mechanism_ablation import (
    DEFAULT_GRAPH, EMPTY_GRAPH, METHODS, prepare_mechanism_decisions,
)
from wam_reranking.belief import initial_belief, update_belief
from wam_reranking.contracts import CoarseCause, TriValue
from test_evidence_residual import attr


class MechanismAblationTests(unittest.TestCase):
    def test_four_arms_only(self):
        self.assertEqual(METHODS, ("value_only", "candidate_only", "no_dependency", "full"))

    def test_full_keeps_frozen_graph(self):
        with patch("wam_reranking.mechanism_ablation.prepare_arbitrated_decisions") as fn:
            prepare_mechanism_decisions(method="full", belief="same", attribution="same")
            self.assertIs(fn.call_args.kwargs["graph"], DEFAULT_GRAPH)
            self.assertEqual(fn.call_args.kwargs["attribution"], "same")

    def test_no_dependency_only_changes_edges(self):
        with patch("wam_reranking.mechanism_ablation.prepare_arbitrated_decisions") as fn:
            prepare_mechanism_decisions(method="no_dependency", backbone="same")
            self.assertIs(fn.call_args.kwargs["graph"], EMPTY_GRAPH)
            self.assertEqual(fn.call_args.kwargs["backbone"], "same")

    def test_control_cannot_use_attribution_gate(self):
        for method in ("value_only", "candidate_only", "oracle", "flat"):
            with self.assertRaises(ValueError): prepare_mechanism_decisions(method=method)

    def test_no_external_graph_override(self):
        with self.assertRaises(ValueError): prepare_mechanism_decisions(method="full", graph=EMPTY_GRAPH)

    def test_direct_invalidation_preserved_but_descendants_not_propagated(self):
        a = attr(CoarseCause.OBJECT_SHIFT)
        full, shallow = initial_belief(0), initial_belief(0)
        update_belief(full, a, 3, DEFAULT_GRAPH)
        update_belief(shallow, copy.deepcopy(a), 3, EMPTY_GRAPH)
        self.assertIs(full.facts["target_pose_current"].value, TriValue.FALSE)
        self.assertIs(shallow.facts["target_pose_current"].value, TriValue.FALSE)
        self.assertTrue(any(not c.direct for c in full.history))
        self.assertFalse(any(not c.direct for c in shallow.history))


if __name__ == "__main__": unittest.main()
