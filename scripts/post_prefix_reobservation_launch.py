"""One-shot launcher for exactly the frozen post-prefix AUX collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION="known_task_post_prefix_reobservation_auxiliary_20261007_v3"
REFERENCE="candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"


def execution_environment(root,code):
    config=root/".libero"/"config.yaml"
    if not config.is_file() or config.stat().st_size==0:
        raise ValueError("Existing project LIBERO config.yaml required; no interactive setup or rewrite")
    env=os.environ.copy();env.update(MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",PYTHONUNBUFFERED="1",PYTHONDONTWRITEBYTECODE="1",
        LIBERO_CONFIG_PATH=str(config.parent),OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join((str(code),str(root/"research_runs"/REFERENCE),str(root))))
    return env


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    root=parser.parse_args().root.resolve();code=Path(__file__).resolve().parent
    if code!=root/"research_runs"/VERSION or (root/"outputs"/VERSION).exists():
        raise ValueError("Exact new version directory required; no resume or overwrite")
    manifest=json.loads((code/"post_prefix_reobservation_code_sha256.json").read_text())
    if set(manifest)!={"post_prefix_reobservation_collect.py","post_prefix_reobservation_supervision.py","post_prefix_reobservation_launch.py"}:
        raise ValueError("Exactly the three frozen files required")
    for name,digest in manifest.items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("Frozen new code changed: "+name)
    env=execution_environment(root,code)
    # Permanent one-shot lock, including failure: no unbounded restart logic.
    descriptor=os.open(code/"launch.lock",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,"w") as stream:stream.write("single bounded launch reserved\n")
    with (code/"collection.log").open("x") as output:
        process=subprocess.Popen([sys.executable,str(code/"post_prefix_reobservation_collect.py"),
            "--root",str(root),"--output",str(root/"outputs"/VERSION),"--collect"],
            cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True)
    value=dict(pid=process.pid,version=VERSION,log=str(code/"collection.log"),
        status=str(root/"outputs"/VERSION/"status.json"),training_started=False,
        source_prefix_pass_cap=2,auxiliary_main_cap=8,QC_cap=16,automatic_restart=False)
    value["fixed_runtime_environment"]={k:env[k] for k in ("LIBERO_CONFIG_PATH","PYTHONPATH","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MUJOCO_GL")}
    value["existing_libero_config_sha256"]=hashlib.sha256((root/".libero"/"config.yaml").read_bytes()).hexdigest()
    (code/"launch.json").write_text(json.dumps(value,indent=2)+"\n")
    print(json.dumps(value),flush=True)


if __name__=="__main__":main()
