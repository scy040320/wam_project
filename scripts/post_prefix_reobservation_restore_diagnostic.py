"""One restore-only diagnostic of the verified V2 prefix; ZERO env.step calls."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np

SOURCE="known_task_post_prefix_reobservation_auxiliary_20261007_v2"
SNAPSHOT_SHA="33ea49036d62526fc286ec5736d0a4131f1705f6085ba8eb812db2176d1dc778"
REFERENCE="candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"


def differences(a,b,path="runtime"):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):return [dict(path=path,kind="dict_structure_mismatch")]
        return [d for key in a for d in differences(a[key],b[key],path+"."+str(key))]
    if isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):return [dict(path=path,kind="list_structure_mismatch")]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+"."+str(i))]
    x,y=np.asarray(a),np.asarray(b)
    if np.array_equal(x,y):return []
    result=dict(path=path,kind="value_mismatch",old_shape=list(x.shape),restored_shape=list(y.shape),old_dtype=str(x.dtype),restored_dtype=str(y.dtype))
    if x.shape==y.shape and x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
        d=np.abs(x.astype(float)-y.astype(float));result.update(finite=bool(np.isfinite(d).all()),
            max_abs=float(d.max()) if d.size else 0.,p95=float(np.quantile(d,.95)) if d.size else 0.,
            changed_values=int(np.count_nonzero(d)),old_values=x.tolist(),restored_values=y.tolist())
    return [result]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out.exists() or out.parent!=Path(__file__).resolve().parent or out.name!="restore_only_diagnostic.json":
        raise ValueError("One exact new diagnostic output, no overwrite")
    code=root/"research_runs"/SOURCE
    launcher=load("postprefix_previous_launcher",code/"post_prefix_reobservation_launch.py")
    os.environ.update(launcher.execution_environment(root,code))
    collector=load("postprefix_previous_collector",code/"post_prefix_reobservation_collect.py")
    for name,digest in collector.DEPENDENCIES.items():
        if collector.sha(root/name)!=digest:raise ValueError("Frozen dependency changed")
    folder=root/"outputs"/SOURCE/"verified_prefixes"/"task57_state10_unknown"
    proof=json.loads((folder/"verified_prefix_proof.json").read_text())
    p=folder/"post_prefix_runtime_snapshot.pkl"
    if proof["new_runtime_snapshot_sha256"]!=SNAPSHOT_SHA or collector.sha(p)!=SNAPSHOT_SHA:
        raise ValueError("Verified existing full post-prefix snapshot changed")
    with p.open("rb") as stream:saved=pickle.load(stream)
    sys.path.insert(0,str(root/"research_runs"/REFERENCE))
    fork=load("postprefix_diagnostic_fork",root/"research_runs"/REFERENCE/"collect_snapshot_fork.py")
    intervention=load("postprefix_diagnostic_intervention",root/"research_runs"/REFERENCE/"base_collector_shift_calibrated.py")
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    env,_=get_libero_env(benchmark.get_benchmark_dict()["libero_90"]().get_task(57),"cosmos",resolution=256)
    try:
        env.reset();raw=fork.restore_runtime_snapshot(env,copy.deepcopy(saved));restored=fork.capture_runtime_snapshot(env)
        actual=collector.actual_observation(raw,intervention);before=np.load(folder/"proprio_actual_before.npy",allow_pickle=False)
        result=dict(schema="restore_only_runtime_difference_diagnostic_v1",source_snapshot_sha256=SNAPSHOT_SHA,
            env_step_calls=0,original_prefix_reconstruction_calls=0,AUX_step_calls=0,training_started=False,
            sim_state_exact=np.array_equal(saved["sim_state"],restored["sim_state"]),
            proprio_max_abs=float(np.abs(actual["proprio"]-before).max()),differences=differences(saved,restored),
            old_sources_and_output_not_modified=True,no_runtime_equality_assertion_or_observation_gate_modified=True)
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=="__main__":main()
