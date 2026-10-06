import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location("_held_audit_test",Path(__file__).resolve().parents[1]/"scripts/audit_existing_held_separation.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class HeldEvidenceAuditTests(unittest.TestCase):
    def test_physical_false_needs_measurement_not_missing_certificate(self):
        self.assertFalse(m.physical_false(dict(value=False,measurement_valid=False)))
        self.assertFalse(m.physical_false(dict(value=None,measurement_valid=True)))
        self.assertTrue(m.physical_false(dict(value=False,measurement_valid=True)))
    def test_positive_needs_original_observable_mask(self):
        self.assertFalse(m.certified_positive(dict(physical_value=True,measurement_valid=True,new_mask=False)))
        self.assertTrue(m.certified_positive(dict(physical_value=True,measurement_valid=True,new_mask=True,deployment_allowed=False)))
    def test_deadline_counts_do_not_count_same_candidate_twice(self):
        rows=[dict(pool_id="p",candidate_id=0,split="train",step=s) for s in (5,10)]
        self.assertEqual(m.deadline_counts(rows),{"4":0,"8":1,"12":1,"16":1})
    def test_before_frame_four_has_no_real_three_frame_identity_window(self):
        self.assertFalse(m.continuous_identity_window({},3,"primary"))
    def test_fixed_scope_not_a_new_confirmation(self):
        self.assertIn("full",m.SOURCE);self.assertIn("v3_perspective",m.LABELS)

if __name__=="__main__":unittest.main()
