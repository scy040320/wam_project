"""Engineering checks; synthetic maps are not experimental evidence."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from wam_reranking import shared_current_visual_baseline as shared
from wam_reranking.relation_evidence import RelationBinding

_path = Path(__file__).resolve().parents[1] / "scripts" / "reextract_shared_current_forecast_evidence.py"
_spec = importlib.util.spec_from_file_location("shared_reextraction_test_script", _path)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)


class FakeFrozenLocalizer:
    def __init__(self):
        self.calls = []

    def relevance(self, images, prompt):
        self.calls.append((tuple(images), prompt))
        maps = []
        for image in images:
            array = np.full((8, 8), .01, np.float32)
            offset = {"black bowl": 0, "plate": 2, "robot gripper": 1}[prompt]
            array[3, (int(image) + offset) % 8] = .9
            maps.append(array)
        return np.stack(maps)


class SharedCurrentReextractionTests(unittest.TestCase):
    def extract(self, forecasts=None):
        localizer = FakeFrozenLocalizer()
        result = script.reextract_pool(localizer=localizer, current_images=[1, 6],
            forecast_images=forecasts or [[2, 5], [3, 4], [4, 3], [5, 2]],
            binding=RelationBinding("black bowl", "plate", "on"), shared_module=shared)
        return localizer, result

    def test_CURRENT_inference_runs_once_per_role_without_forecast_images(self):
        localizer, result = self.extract()
        self.assertEqual(len(localizer.calls), 6)
        for images, prompt in localizer.calls[:3]:
            self.assertEqual(images, (1, 6))
            self.assertIn(prompt, ("black bowl", "plate", "robot gripper"))
        for images, _ in localizer.calls[3:]:
            self.assertEqual(len(images), 8)
        self.assertTrue(result["audit"]["passed"])
        self.assertEqual(result["shared_current"]["predicate_facts"], [])

    def test_before_and_weights_are_shared_across_all_K4_candidates(self):
        _, result = self.extract()
        for candidate in result["candidates"]:
            diagnostic = candidate["fusion_diagnostic"]
            for name in ("before", "relation_weights", "contact_weights"):
                self.assertEqual(diagnostic[name], result["shared_current"][name])
            for name in shared.BEFORE_FIELDS:
                self.assertNotIn(name, candidate["forecast_expected_effects"])
                self.assertEqual(candidate["candidate_visual_evidence"][name], result["shared_current"]["before"][name])
            self.assertFalse(candidate["predicted_effects_are_current_facts"])

    def test_forecast_changes_cannot_change_current_maps_or_before_proxy(self):
        _, a = self.extract()
        _, b = self.extract([[7, 0], [6, 1], [1, 7], [0, 6]])
        self.assertEqual(a["shared_current"], b["shared_current"])
        for role in script.ROLES:
            np.testing.assert_array_equal(a["current_maps"][role], b["current_maps"][role])
        self.assertNotEqual(a["candidates"][0]["forecast_expected_effects"], b["candidates"][0]["forecast_expected_effects"])

    def test_forecast_order_only_permutes_own_expected_effects(self):
        data = [[2, 5], [3, 4], [4, 3], [5, 2]]
        _, a = self.extract(data)
        order = (2, 0, 3, 1)
        _, b = self.extract([data[index] for index in order])
        self.assertEqual(a["shared_current"], b["shared_current"])
        for cid, old_cid in enumerate(order):
            self.assertEqual(b["candidates"][cid]["forecast_expected_effects"],
                             a["candidates"][old_cid]["forecast_expected_effects"])

    def test_maps_are_read_only_and_not_calibrated_physical_certificates(self):
        _, result = self.extract()
        for role in script.ROLES:
            self.assertFalse(result["current_maps"][role].flags.writeable)
        self.assertFalse(result["shared_current"]["confidence_calibrated"])
        self.assertFalse(result["shared_current"]["physical_state_certified"])

    def test_fixed_engineering_pilot_is_outcome_blind_and_train_only(self):
        pairs = {(0, 0, 0, "clean"): {"split": "train", "success": False},
                 (0, 1, 0, "clean"): {"split": "train", "success": True},
                 (9, 0, 0, "clean"): {"split": "train", "success": True},
                 (9, 6, 0, "clean"): {"split": "val", "success": False}}
        expected = ((0, 0, 0, "clean"), (9, 0, 0, "clean"))
        self.assertEqual(script.engineering_pilot_keys(pairs), expected)
        for value in pairs.values():
            value["success"] = not value["success"]
        self.assertEqual(script.engineering_pilot_keys(pairs), expected)

    def test_default_LIBERO10_bindings_cannot_silently_replace_frozen90(self):
        bindings = {key: RelationBinding(*value) for key, value in script.EXPECTED_BINDINGS.items()}
        script.verify_bindings(bindings)
        bindings[(0, 0)] = RelationBinding("alphabet soup", "basket", "toward")
        with self.assertRaisesRegex(ValueError, "LIBERO90"):
            script.verify_bindings(bindings)

    def test_missing_candidate_or_view_and_nonfinite_localizer_output_fail(self):
        with self.assertRaises(ValueError):
            self.extract([[2, 5], [3, 4]])
        class BadLocalizer:
            def relevance(self, images, target):
                return np.full((len(images), 8, 8), np.nan)
        with self.assertRaises(ValueError):
            script.reextract_pool(localizer=BadLocalizer(), current_images=[1, 6],
                forecast_images=[[2, 5]] * 4, binding=RelationBinding("black bowl", "plate", "on"), shared_module=shared)

    def test_future_physical_or_outcome_inputs_have_no_API_slot(self):
        with self.assertRaises(TypeError):
            script.reextract_pool(localizer=FakeFrozenLocalizer(), current_images=[1, 6],
                forecast_images=[[2, 5]] * 4, binding=RelationBinding("black bowl", "plate", "on"),
                shared_module=shared, actual_future_grasp=True)

    def test_isolated_payload_loader_does_not_require_new_repo_module(self):
        calls = []
        fake_module = SimpleNamespace(__file__="frozen_payload/module.py")
        def load(name, path):
            calls.append((name, path))
            return fake_module
        package = SimpleNamespace()
        payload = Path(shared.__file__)
        with patch.dict(sys.modules, {"wam_reranking": package}):
            actual = script.load_shared_payload(SimpleNamespace(rt=SimpleNamespace(load=load)), payload)
            self.assertIs(actual, fake_module)
            self.assertIs(package.shared_current_visual_baseline, fake_module)
        self.assertEqual(calls, [("wam_reranking.shared_current_visual_baseline", payload.resolve())])
        with self.assertRaisesRegex(ValueError, "missing"):
            script.load_shared_payload(SimpleNamespace(), payload.parent / "absent.py")

    def test_cache_copy_requires_pinned_source_not_merely_observed_new_digest(self):
        source, copy = Path("source.npy").resolve(), Path("cache.npy").resolve()
        pins = {str(source): "frozen_sha"}
        with patch.object(script, "sha", return_value="frozen_sha"):
            self.assertEqual(script.pin_identical_cache_copy(copy, source, pins, {}), copy)
            self.assertEqual(pins[str(copy)], "frozen_sha")
            with self.assertRaisesRegex(ValueError, "independently pinned"):
                script.pin_identical_cache_copy(copy, source, {}, {})
            with self.assertRaisesRegex(ValueError, "existing immutable audit"):
                script.pin_identical_cache_copy(copy, source, pins, {str(copy): "other_sha"})
        with patch.object(script, "sha", return_value="modified_sha"):
            with self.assertRaisesRegex(ValueError, "cache/source"):
                script.pin_identical_cache_copy(copy, source, pins, {})


if __name__ == "__main__":
    unittest.main()
