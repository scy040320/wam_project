"""Synthetic namespace/source guards. These tests never fit or collect data."""
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch_observed_event_overlay.py"
SPEC = importlib.util.spec_from_file_location("namespace_overlay_launcher_tested", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class NamespaceOverlayLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="overlay_namespace_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.code = self.root / "private_code"
        self.lib = self.code / "lib" / "wam_reranking"
        self.lib.mkdir(parents=True)
        for name in ("__init__.py", "direct_recovery.py", "recovery_contract.py"):
            (self.lib / name).write_text("PRIVATE_FIXTURE = True\n", encoding="utf-8")
        self.preparer = self.code / "prepare_observed_event_overlay.py"
        self.preparer.write_text("def prepare(*args):\n return dict(passed=True, training_ready=False, training_started=False, overlay_candidates=160, inherited_rank_rows=768, inherited_auxiliary_rows=957)\n", encoding="utf-8")
        for name in ("launch_observed_event_overlay.py", "finish_observed_event_overlay.sh"):
            (self.code / name).write_text("# synthetic fixture, never executed\n", encoding="utf-8")
        self.freeze()
        self.input = self.root / "input.json"
        self.input.write_text("{}\n", encoding="utf-8")
        self.source, self.labels = self.root / "source", self.root / "labels"
        for directory, name in ((self.source, "completion_audit.json"), (self.labels, "observability_label_audit.json")):
            directory.mkdir()
            (directory / name).write_text(json.dumps(dict(passed=True, completed_candidates=160, training_ready=False)), encoding="utf-8")
        self.output = self.root / "output"

    def freeze(self):
        self.preparer_sha = launcher.digest(self.preparer)
        names = [p.relative_to(self.code).as_posix() for p in self.lib.glob("*.py")] + [self.preparer.name]
        hashes = {name:launcher.digest(self.code / name) for name in names}
        (self.code / "frozen_parent_code_sha256.json").write_text(json.dumps(hashes), encoding="utf-8")
        hashes = {name:launcher.digest(self.code / name) for name in (
            "launch_observed_event_overlay.py", "finish_observed_event_overlay.sh", "frozen_parent_code_sha256.json")}
        (self.code / "deployed_code_sha256.json").write_text(json.dumps(hashes), encoding="utf-8")

    def verify_code(self):
        with patch.object(launcher, "FROZEN_PREPARER_SHA", self.preparer_sha):
            return launcher.verify_frozen_code(self.code)

    def verify_sources(self):
        with patch.object(launcher, "FROZEN_INPUT_SHA", launcher.digest(self.input)):
            launcher.verify_sources(self.source, self.labels, self.input)

    def test_unchanged_parent_private_package_hashes_pass(self):
        self.assertIn("lib/wam_reranking/direct_recovery.py", self.verify_code())

    def test_missing_private_package_member_is_not_silently_reused(self):
        (self.lib / "direct_recovery.py").unlink()
        with self.assertRaisesRegex(ValueError, "completeness"):
            self.verify_code()

    def test_extra_private_module_does_not_escape_parent_manifest(self):
        (self.lib / "unfrozen.py").write_text("EXTRA = True\n")
        with self.assertRaisesRegex(ValueError, "completeness"):
            self.verify_code()

    def test_changed_private_source_rejected(self):
        (self.lib / "direct_recovery.py").write_text("CHANGED = True\n")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.verify_code()

    def test_changed_new_launcher_rejected(self):
        (self.code / "finish_observed_event_overlay.sh").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "deployment hash changed"):
            self.verify_code()

    def test_manifest_cannot_rebind_frozen_preparer(self):
        with patch.object(launcher, "FROZEN_PREPARER_SHA", "a" * 64), self.assertRaisesRegex(ValueError, "preparer fingerprint"):
            launcher.verify_frozen_code(self.code)

    def test_private_prefix_wins_over_project_cwd_and_pythonpath(self):
        old = self.root / "old_project"
        (old / "wam_reranking").mkdir(parents=True)
        (old / "wam_reranking" / "__init__.py").write_text("OLD_FIXTURE = True\n")
        code = "import importlib.util,json; s=importlib.util.spec_from_file_location('test_launcher'," + repr(str(SCRIPT)) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.load_private_namespace(" + repr(str(self.code)) + ")))"
        env = dict(os.environ, PYTHONPATH=str(old), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=old, env=env,
            capture_output=True, text=True, check=True)
        origins = json.loads(result.stdout)
        self.assertEqual(set(launcher.MODULE_NAMES), set(origins))
        for path in origins.values():
            self.assertIn(self.code / "lib", Path(path).parents)

    def test_preloaded_foreign_package_stops_instead_of_silent_eviction(self):
        foreign = types.SimpleNamespace(__file__=str(self.root / "outside.py"))
        with patch.dict(sys.modules, {"wam_reranking":foreign}), self.assertRaisesRegex(ValueError, "already loaded"):
            launcher.load_private_namespace(self.code)

    def test_private_path_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, "leaves"):
            launcher.private_path(self.code, "../outside.py")

    def test_full_technical_source_audits_do_not_need_or_set_training_ready(self):
        self.verify_sources()
        self.assertFalse(json.loads((self.labels / "observability_label_audit.json").read_text())["training_ready"])

    def test_partial_or_boolean_candidate_count_rejected(self):
        for count in (159, True):
            with self.subTest(count=count):
                path = self.labels / "observability_label_audit.json"
                path.write_text(json.dumps(dict(passed=True, completed_candidates=count)))
                with self.assertRaisesRegex(ValueError, "integrity gate"):
                    self.verify_sources()

    def test_source_failure_cannot_be_retried_automatically(self):
        (self.source / "failure.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "automatically resumed"):
            self.verify_sources()

    def test_canonical_input_hash_mismatch_stops(self):
        with patch.object(launcher, "FROZEN_INPUT_SHA", "b" * 64), self.assertRaisesRegex(ValueError, "input fingerprint"):
            launcher.verify_sources(self.source, self.labels, self.input)

    def test_join_only_wrapper_cannot_upgrade_training_readiness(self):
        self.preparer.write_text("def prepare(*args):\n return dict(passed=True, training_ready=True, training_started=False)\n")
        self.freeze()
        with patch.object(launcher, "FROZEN_PREPARER_SHA", self.preparer_sha), patch.object(
            launcher, "FROZEN_INPUT_SHA", launcher.digest(self.input)), patch.object(
            launcher, "load_private_namespace", return_value={}), self.assertRaisesRegex(ValueError, "promote readiness"):
            launcher.run_overlay(self.code, self.source, self.labels, self.input, self.output)

    def test_existing_output_cannot_be_overwritten(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            launcher.run_overlay(self.code, self.source, self.labels, self.input, self.output)


if __name__ == "__main__":
    unittest.main()
