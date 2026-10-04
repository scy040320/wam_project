"""Public group-source contract: exact pairing, immutable cache, no outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wam_reranking.shared_source import (
    array_sha, observation_sha, source_identity, write_source, read_source,
)

class SharedSourceTests(unittest.TestCase):
    def payload(self):
        return {'snapshot':{'sim_state':np.arange(7,dtype=np.float64)},
            'query_obs':{'primary_image':np.zeros((4,4,3),np.uint8),
                'wrist_image':np.ones((4,4,3),np.uint8),'proprio':np.arange(9,dtype=np.float64)},
            'source_result':{'actions':np.zeros((16,7),np.float64)}}

    def prepare(self,folder):
        p=self.payload()
        write_source(folder,p,source_identity(0,31,p,array_sha,observation_sha))
        return p

    def test_conditions_read_same_source_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'group';p=self.prepare(g);hashes=[]
            for _ in range(4):
                q,m=read_source(g,0,31,array_sha,observation_sha)
                np.testing.assert_array_equal(q['snapshot']['sim_state'],p['snapshot']['sim_state'])
                hashes.append(m['payload_sha256']);q['source_result']['actions'][0,0]=99
            self.assertEqual(len(set(hashes)),1)
            q,_=read_source(g,0,31,array_sha,observation_sha)
            self.assertEqual(q['source_result']['actions'][0,0],0)

    def test_cannot_overwrite_source(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'group';p=self.prepare(g)
            with self.assertRaises(FileExistsError):write_source(g,p,source_identity(0,31,p,array_sha,observation_sha))

    def test_wrong_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'group';self.prepare(g)
            for task,state in [(9,31),(0,32)]:
                with self.assertRaisesRegex(RuntimeError,'identity mismatch'):read_source(g,task,state,array_sha,observation_sha)

    def test_payload_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'group';self.prepare(g)
            p=g/'source.pkl';p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(RuntimeError,'payload hash mismatch'):read_source(g,0,31,array_sha,observation_sha)

    def test_changed_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'group';self.prepare(g)
            p=g/'source_manifest.json';m=json.loads(p.read_text());m['identity']['actions_sha256']='wrong';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):read_source(g,0,31,array_sha,observation_sha)

    def test_each_source_component_changes_identity(self):
        p=self.payload();i=source_identity(0,31,p,array_sha,observation_sha)
        for where,key in [('snapshot','sim_state'),('query_obs','wrist_image'),('source_result','actions')]:
            q=copy.deepcopy(p);q[where][key].flat[0]+=1
            self.assertNotEqual(source_identity(0,31,q,array_sha,observation_sha),i)

    def test_terminal_prefix_is_group_scoped(self):
        with tempfile.TemporaryDirectory() as td:
            g=Path(td)/'group';write_source(g,{'early_terminal':True,'prefix':[]},{'task':0,'state':31,'early_terminal':True})
            p,_=read_source(g,0,31,array_sha,observation_sha);self.assertTrue(p['early_terminal'])

    def test_shape_and_dtype_are_part_of_identity(self):
        a=np.zeros((16,7),np.float32)
        self.assertNotEqual(array_sha(a),array_sha(a.astype(np.float64)))
        self.assertNotEqual(array_sha(a),array_sha(a.reshape(7,16)))

if __name__=='__main__':unittest.main()
