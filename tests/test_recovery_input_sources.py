"""Adversarial X byte checks; fixtures are synthetic, not admitted data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from wam_reranking.joint_recovery_contract import ArtifactReader
from wam_reranking.recovery_input_sources import validate_input_source_contents


class InputSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reader = ArtifactReader(self.root)
        def ref(name, value):
            p = self.root / name
            if p.suffix == ".npy":
                np.save(p, value, allow_pickle=False)
            else:
                p.write_text(json.dumps(value), encoding="utf8")
            return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        self.ref = ref
        rgb = ref("rgb.npy", np.zeros((8, 8, 3), dtype=np.uint8))
        quality = {"primary_reliable": True, "wrist_reliable": False,
            "execution_reliable": True, "cross_view_conflict": False,
            "observer_evidence_available": None}
        visual = {"example": {"value": .2, "available": True}}
        binding = {"language": "task instruction", "target": "subject", "anchor": "reference", "relation": "inside"}
        self.x = dict(actual_before=dict(primary_rgb=rgb, wrist_rgb=rgb,
            proprio=[0.] * 9, proprio_source=ref("proprio.npy", np.zeros(9)),
            quality=quality, quality_source=ref("quality.json", quality)),
            candidate_predicted=dict(primary_endpoint_rgb=rgb, wrist_endpoint_rgb=rgb,
            visual=visual, visual_source=ref("visual.json", visual)),
            planned_actions=np.zeros((16, 7)).tolist(),
            planned_actions_source=ref("actions.npy", np.zeros((16, 7))),
            task_context=dict(binding, entity_binding_source=ref("binding.json", binding)))

    def tearDown(self):
        self.temp.cleanup()

    def test_content_sources_pass(self):
        self.assertTrue(validate_input_source_contents(self.x, self.reader))

    def test_text_cannot_impersonate_rgb(self):
        self.x["actual_before"]["primary_rgb"] = self.ref("fake.py", {"image": "not pixels"})
        with self.assertRaisesRegex(ValueError, "decodable RGB"):
            validate_input_source_contents(self.x, self.reader)

    def test_changed_inline_proprio_rejected(self):
        self.x["actual_before"]["proprio"][0] = .5
        with self.assertRaisesRegex(ValueError, "proprio inline"):
            validate_input_source_contents(self.x, self.reader)

    def test_changed_inline_plan_rejected(self):
        self.x["planned_actions"][4][6] = 1.
        with self.assertRaisesRegex(ValueError, "actions inline"):
            validate_input_source_contents(self.x, self.reader)

    def test_changed_inline_quality_rejected(self):
        self.x["actual_before"]["quality"]["observer_evidence_available"] = True
        with self.assertRaisesRegex(ValueError, "quality inline"):
            validate_input_source_contents(self.x, self.reader)

    def test_changed_inline_visual_rejected(self):
        self.x["candidate_predicted"]["visual"]["example"]["value"] = .8
        with self.assertRaisesRegex(ValueError, "visual inline"):
            validate_input_source_contents(self.x, self.reader)

    def test_changed_inline_binding_rejected(self):
        self.x["task_context"]["target"] = "other instance"
        with self.assertRaisesRegex(ValueError, "binding inline"):
            validate_input_source_contents(self.x, self.reader)

    def test_proprio_shape_not_padded(self):
        self.x["actual_before"]["proprio_source"] = self.ref("short.npy", np.zeros(7))
        with self.assertRaisesRegex(ValueError, "shape/type/value"):
            validate_input_source_contents(self.x, self.reader)

    def test_float_or_grayscale_array_not_canonical_rgb(self):
        for pixels in (np.zeros((8, 8)), np.zeros((8, 8, 3))):
            self.x["candidate_predicted"]["primary_endpoint_rgb"] = self.ref("bad.npy", pixels)
            with self.assertRaisesRegex(ValueError, "uint8 HxWx3"):
                validate_input_source_contents(self.x, ArtifactReader(self.root))

    def test_changed_bytes_rejected_after_new_reader(self):
        path = Path(self.x["planned_actions_source"]["path"])
        np.save(path, np.ones((16, 7)), allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            validate_input_source_contents(self.x, ArtifactReader(self.root))


if __name__ == "__main__":
    unittest.main()
