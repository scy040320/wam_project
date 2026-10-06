"""CPU manifest/membership tests only; no renderer or pickle execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC=importlib.util.spec_from_file_location("_archived_release_replay_test",
    Path(__file__).resolve().parents[1]/"scripts/replay_archived_release_evidence.py")
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class HistoricalManifestTests(unittest.TestCase):
    def test_official_close_then_open_is_schedule_only(self):
        x=np.zeros((16,7));x[:8,6]=1;x[8:,6]=-1
        self.assertTrue(m.close_before_open(x))
        self.assertFalse(m.close_before_open(x[::-1]))

    def test_nonfinite_and_short_action_blocks_rejected(self):
        for a in (np.zeros((15,7)),np.full((16,7),np.nan)):
            with self.assertRaises(ValueError):m.close_before_open(a)

    def test_threshold_signs_not_relaxed(self):
        a=np.zeros((16,7));a[:8,6]=.5;a[8:,6]=-.5
        self.assertFalse(m.close_before_open(a))

    def test_path_escape_and_absolute_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for ref in ("../outside",str(Path(d).resolve()/"a")):
                with self.assertRaises(ValueError):m.within(Path(d),ref)

    def test_contiguous_prefix_mandatory_no_partial_state_substitute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"block_3").mkdir();(root/"block_5").mkdir()
            with self.assertRaises(ValueError):m.chain_block_numbers(root,5)
            (root/"block_4").mkdir()
            self.assertEqual([x.name for x in m.chain_block_numbers(root,5)],["block_3","block_4","block_5"])

    def test_first_block_before_shared_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.chain_block_numbers(Path(d),2)

    def test_provenance_requires_explicit_v8_frozen_snapshot_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scope="task9_state10_clean";base=root/"outputs"/m.ROOT_SOURCE/"scenarios"/scope
            expected={}
            for name in m.REQUIRED_ROOT_FILES:
                p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"test bytes not pickle executed")
                expected[str(p.relative_to(root)).replace("\\","/")]=m.sha(p)
            m.dump(base/"shared_root_pool/pool.json",dict(observation_sha256="matched"))
            expected[str((base/"shared_root_pool/pool.json").relative_to(root)).replace("\\","/")]=m.sha(base/"shared_root_pool/pool.json")
            m.dump(root/"outputs"/m.SOURCE/"scenarios"/scope/"scenario.json",dict(root_observation_sha256="matched"))
            result=m.validate_root_provenance(root,scope,expected)
            self.assertEqual(result["root_source"],m.ROOT_SOURCE)
            scenario_relative=str((root/"outputs"/m.SOURCE/"scenarios"/scope/"scenario.json").relative_to(root)).replace("\\","/")
            self.assertIn(scenario_relative,result["source_read_sha256"])
            m.verify_source_table(root,result["source_read_sha256"])
            m.dump(root/"outputs"/m.SOURCE/"scenarios"/scope/"scenario.json",dict(root_observation_sha256="changed_after_read"))
            with self.assertRaises(ValueError):m.verify_source_table(root,result["source_read_sha256"])
            del expected[str((base/"root_runtime_snapshot.pkl").relative_to(root)).replace("\\","/")]
            with self.assertRaises(ValueError):m.validate_root_provenance(root,scope,expected)

    def test_same_task_state_does_not_replace_observation_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scope="task9_state10_clean";base=root/"outputs"/m.ROOT_SOURCE/"scenarios"/scope;expected={}
            for name in m.REQUIRED_ROOT_FILES:
                p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"bytes")
                expected[str(p.relative_to(root)).replace("\\","/")]=m.sha(p)
            m.dump(base/"shared_root_pool/pool.json",dict(observation_sha256="original"))
            expected[str((base/"shared_root_pool/pool.json").relative_to(root)).replace("\\","/")]=m.sha(base/"shared_root_pool/pool.json")
            m.dump(root/"outputs"/m.SOURCE/"scenarios"/scope/"scenario.json",dict(root_observation_sha256="different"))
            with self.assertRaises(ValueError):m.validate_root_provenance(root,scope,expected)

    def test_source_mutation_detected_without_deserializing_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"snapshot.pkl";p.write_bytes(b"frozen bytes")
            row=dict(root_provenance=dict(frozen_inputs_sha256={"snapshot.pkl":m.sha(p)}),chain=[])
            m.verify_chain_sources(root,row);p.write_bytes(b"changed bytes")
            with self.assertRaises(ValueError):m.verify_chain_sources(root,row)

    def test_flat_base_requires_exact_frozen_sha_no_root_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.verified_base_path(Path(d))
            p=Path(d)/"replay_temporal_recovery_evidence.py";p.write_bytes(b"wrong source")
            with self.assertRaises(ValueError):m.verified_base_path(Path(d))
            with patch.object(m,"BASE_SHA",m.sha(p)):
                self.assertEqual(m.verified_base_path(Path(d)),p)

    def test_full_manifest_source_table_is_verified_after_reading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"trajectory.npz";p.write_bytes(b"source")
            table={"trajectory.npz":m.sha(p)};m.verify_source_table(root,table)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):m.verify_source_table(root,table)


if __name__=="__main__":unittest.main()
