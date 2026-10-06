"""One scoped finite native probe launch; no restart or training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION="known_task_native_block_boundary_reobservation_20261007_v3_python310_compat"
REFERENCE="candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"
LIBRARY="known_task_recovery_observation_input_overlay_20261006_v2_namespace_scoped/lib"


def execution_environment(root,code):
    config=root/".libero"/"config.yaml"
    if not config.is_file() or not config.stat().st_size:raise ValueError("Existing LIBERO config required; no interactive rewrite")
    env=os.environ.copy();env.update(LIBERO_CONFIG_PATH=str(config.parent),MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",
        PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join((str(code),str(root/"research_runs"/LIBRARY),str(root/"research_runs"/REFERENCE),str(root))))
    return env


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);root=p.parse_args().root.resolve();code=Path(__file__).resolve().parent
    if code!=root/"research_runs"/VERSION or (root/"outputs"/VERSION).exists():raise ValueError("Exact new version only; no resume")
    manifest=json.loads((code/"native_block_reobservation_code_sha256.json").read_text())
    expected={"native_block_reobservation_collect.py","native_block_reobservation_supervision.py","native_block_reobservation_launch.py",
        "native_recovery_block_boundary_protocol_20261007_v1.json","native_recovery_block_boundary_protocol_20261007_v2.json","hand_contact_supervision.py"}
    if set(manifest)!=expected:raise ValueError("Exact finite code/protocol manifest required")
    for name,digest in manifest.items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=digest:raise ValueError("Frozen new code/protocol changed")
    env=execution_environment(root,code)
    descriptor=os.open(code/"launch.lock",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,"w") as stream:stream.write("one finite launch reserved\n")
    with (code/"collection.log").open("x") as output:
        proc=subprocess.Popen([sys.executable,str(code/"native_block_reobservation_collect.py"),"--root",str(root),"--collect"],
            cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(pid=proc.pid,version=VERSION,log=str(code/"collection.log"),status=str(root/"outputs"/VERSION/"status.json"),
        source_prefix_actions_cap=32,native_main_cap=8,native_QC_cap=8,Cosmos_queries_cap=8,all_action_steps_cap=288,
        source_gates_before_Cosmos_load=True,one_model_load=True,automatic_restart=False,training_started=False,
        environment={k:env[k] for k in ("LIBERO_CONFIG_PATH","PYTHONPATH","MUJOCO_GL","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    (code/"launch.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf8");print(json.dumps(record),flush=True)


if __name__=="__main__":main()
