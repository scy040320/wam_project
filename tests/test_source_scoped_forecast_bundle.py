"""Bounded builder contract tests; not classifier, benefit or fact evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/build_source_scoped_forecast_bundle.py"
SPEC = importlib.util.spec_from_file_location("test_source_scoped_bundle_builder", PATH)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class SourceScopedBundleTests(unittest.TestCase):
    def meta(self):
        return dict(block_index=2,source_endpoint_reference={"endpoint_block_index":3},
            alignment_valid_t_plus_H=True,executed_steps=16,actual_observation_policy_step=48,
            prediction_target_policy_step=48)

    def test_identity_uses_explicit_source_and_aligned_endpoint(self):
        ident=BUILD.source_identity((0,6,0,"object_shift"),self.meta(),"libero90_task0_block2","original192")
        self.assertEqual(ident["block_index"],3);self.assertEqual(ident["block_id"],"libero90_task0_block2")
        self.assertEqual(ident["dataset"],"original192");self.assertEqual(ident["state"],6)

    def test_wrong_source_short_block_and_unaligned_endpoint_rejected(self):
        for field,value in (("block_index",True),("alignment_valid_t_plus_H",False),
                            ("executed_steps",15),("actual_observation_policy_step",47)):
            with self.subTest(field=field),self.assertRaises(ValueError):
                BUILD.source_identity((0,6,0,"object_shift"),{**self.meta(),field:value},
                                      "libero90_task0_block2","original192")
        with self.assertRaises(ValueError):
            BUILD.source_identity((0,6,0,"object_shift"),self.meta(),"libero90_task9_block2","original192")
        with self.assertRaises(ValueError):
            BUILD.source_identity((0,6,0,"object_shift"),
                {**self.meta(),"source_endpoint_reference":{"endpoint_block_index":4}},
                "libero90_task0_block2","original192")

    def test_separate_parser_load_does_not_replace_frozen_package_module(self):
        import wam_reranking.candidate_effects as frozen
        original=sys.modules["wam_reranking.candidate_effects"]
        separate=BUILD.load_payload("wam_reranking.test_audit_ordered_parser",Path(frozen.__file__))
        self.assertIs(sys.modules["wam_reranking.candidate_effects"],original)
        self.assertIsNot(separate,frozen)
        self.assertIn("masked_soft_only",BUILD.MODES)
        self.assertIn("ranker_command",BUILD.MODES)
        self.assertIn("learned_no_dag",BUILD.MODES)

    def test_two_missing_or_boolean_policy_steps_do_not_prove_alignment(self):
        for value in (None, True, -1, 48.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BUILD.source_identity((0,6,0,"object_shift"),
                    {**self.meta(), "actual_observation_policy_step":value,
                     "prediction_target_policy_step":value},
                    "libero90_task0_block2","original192")

    def test_journal_receipt_must_be_preapproved_exact_sha_not_self_pin(self):
        self.assertEqual(BUILD.verify_preapproved_receipt("path","a"*64,lambda p:"a"*64),"a"*64)
        for value in ("A"*64,"a"*63,"b"*64,None):
            with self.subTest(expected=value),self.assertRaises(ValueError):
                BUILD.verify_preapproved_receipt("path",value,lambda p:"a"*64)

    def test_metadata_pin_transferred_only_from_prior_frozen_inventory(self):
        path=Path("immutable_block.json").resolve(); pins={}
        self.assertEqual(BUILD.import_frozen_source_pin(path,{str(path):"a"*64},pins,
                                                       lambda p:"a"*64),"a"*64)
        self.assertEqual(pins,{str(path):"a"*64})
        for expected,existing in (({},{}),({str(path):"b"*64},{}),
                                 ({str(path):"a"*64},{str(path):"b"*64})):
            with self.subTest(expected=expected,existing=existing),self.assertRaises(ValueError):
                BUILD.import_frozen_source_pin(path,expected,existing,lambda p:"a"*64)


if __name__ == "__main__": unittest.main()
