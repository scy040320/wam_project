"""Immutable, once-per-(task,state) source. No candidate outcome selection."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np

def array_sha(value):
    a=np.ascontiguousarray(np.asarray(value));h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
    return h.hexdigest()

def observation_sha(obs):
    h=hashlib.sha256()
    for key in ('primary_image','wrist_image','proprio'):
        h.update(key.encode());h.update(array_sha(obs[key]).encode())
    return h.hexdigest()

def source_identity(task,state,payload,sha_array,obs_hash):
    return {'task':task,'state':state,
        'snapshot_sha256':sha_array(payload['snapshot']['sim_state']),
        'query_sha256':obs_hash(payload['query_obs']),
        'actions_sha256':sha_array(payload['source_result']['actions'])}

def write_source(folder,payload,identity):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    tmp=folder/'source.pkl.tmp'
    with tmp.open('xb') as f:pickle.dump(payload,f)
    tmp.replace(folder/'source.pkl')
    record={'identity':identity,'payload_sha256':hashlib.sha256((folder/'source.pkl').read_bytes()).hexdigest(),
        'early_terminal':payload.get('early_terminal',False),'source_created_once':True}
    (folder/'source_manifest.json').write_text(json.dumps(record,indent=2)+'\n')

def read_source(folder,task,state,sha_array,obs_hash):
    folder=Path(folder)
    record=json.loads((folder/'source_manifest.json').read_text())
    raw=(folder/'source.pkl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['payload_sha256']:
        raise RuntimeError('shared source payload hash mismatch')
    payload=pickle.loads(raw)
    if payload.get('early_terminal'):
        if record['identity']!={'task':task,'state':state,'early_terminal':True}:
            raise RuntimeError('terminal group identity mismatch')
    elif source_identity(task,state,payload,sha_array,obs_hash)!=record['identity']:
        raise RuntimeError('shared source identity mismatch')
    return payload,record
