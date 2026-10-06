import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('_v10_reuse_audit',Path(__file__).resolve().parents[1]/'scripts/audit_v10_reusable_supervision.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReuseAuditTests(unittest.TestCase):
 def test_unknown_is_masked_not_negative(self):
  self.assertEqual(m.atom_status({'new_mask':True,'physical_value':None}),'masked')
 def test_missing_mask_is_not_negative(self):
  self.assertEqual(m.atom_status({'new_mask':False,'physical_value':False}),'masked')
 def test_explicit_negative_requires_certified_mask(self):
  self.assertEqual(m.atom_status({'new_mask':True,'physical_value':False}),'negative')
 def test_no_endpoint_available_fabrication(self):
  self.assertEqual(m.endpoint_status({'after':'false','after_mask':False},'after'),'masked')
 def test_seven_heads_not_all_nine(self):
  self.assertEqual(len(m.HEADS),7)
  self.assertNotIn('placed',m.HEADS);self.assertNotIn('execution_consistent',m.HEADS)

if __name__=='__main__':unittest.main()
