"""A structural source audit must not manufacture runtime timestamps/facts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_actual_window_sources.py"
_spec = importlib.util.spec_from_file_location("window_source_audit_under_test", _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


class WindowSourceAuditTests(unittest.TestCase):
    def timing(self, **changes):
        raw = dict(executed_length=16, step_indices=list(range(17))) | changes
        return audit.time_record(raw, dict(executed_length=16, only_first_block=True),
            dict(block=3), dict(seed=10031048, value=.3, seconds=1.5))

    def test_local_steps_do_not_certify_captured_absolute_or_PRE_time(self):
        result = self.timing()
        self.assertTrue(result["local_temporal_order_verified"])
        self.assertFalse(result["captured_absolute_runtime_time_verified"])
        self.assertFalse(result["runtime_PRE_binding_verified"])
        self.assertFalse(result["block_number_multiplied_by_chunk_length"])
        self.assertEqual(result["query_block_diagnostic"], 3)
        self.assertIn("captured_absolute_policy_steps", result["missing_runtime_fields"])
        self.assertNotIn("policy_step_start", result)

    def test_declared_timestamp_is_recorded_but_not_independently_certified(self):
        result = self.timing(capture_policy_steps=list(range(48, 65)), decision_policy_step=64)
        self.assertEqual(result["captured_absolute_fields"]["record"]["decision_policy_step"], 64)
        self.assertFalse(result["captured_absolute_runtime_time_verified"])
        self.assertFalse(result["runtime_PRE_binding_verified"])
        self.assertIn("independently_bound_recipient_SourceSnapshot", result["missing_runtime_fields"])

    def test_local_gaps_or_bool_indices_fail(self):
        for steps in (list(range(16)), [False] + list(range(1, 17)), [1] + list(range(1, 17))):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.timing(step_indices=steps)

    def test_wrong_executed_length_cannot_be_padded(self):
        with self.assertRaises(ValueError):
            self.timing(executed_length=15)

    def test_sources_cannot_escape_manifest_root(self):
        root = Path.cwd()
        for relative in ("../outside.npy", str(root.resolve()), ""):
            with self.subTest(path=relative), self.assertRaises(ValueError):
                audit.inside(root, relative)

    def test_numeric_array_shape_and_finite_dtype_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "actual.npy"
            np.save(path, np.zeros((17, 9)), allow_pickle=False)
            _, record = audit.array_record(path, (17, 9), "numeric")
            self.assertEqual(record["shape"], [17, 9])
            for array in (np.zeros((16, 9)), np.zeros((17, 9), bool), np.full((17, 9), np.nan)):
                np.save(path, array, allow_pickle=False)
                with self.assertRaises(ValueError):
                    audit.array_record(path, (17, 9), "numeric")

    def test_RGB_requires_uint8_and_canonical_array_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "actual.npy"
            array = np.zeros((2, 3, 3), np.uint8)
            np.save(path, array, allow_pickle=False)
            _, record = audit.array_record(path, array.shape, "rgb")
            self.assertEqual(record["canonical_array_sha256"], audit.canonical_array_sha(array))
            np.save(path, array.astype(float), allow_pickle=False)
            with self.assertRaises(ValueError):
                audit.array_record(path, array.shape, "rgb")

    def test_missing_immutable_pin_is_not_replaced_by_observed_hash(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            path = Path(folder) / "metadata.json"
            path.write_text("{}", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "immutable hash manifest"):
                audit.pinned(Path(folder), "metadata.json", {}, {})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                audit.pinned(Path(folder), "metadata.json", {"metadata.json": "a" * 64}, {})


if __name__ == "__main__":
    unittest.main()
