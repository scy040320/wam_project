#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/wam_project/cosmos-policy
version=known_task_full_v8_command_predicate_training_20261008_v10_v3
mkdir -p logs
exec 9>"logs/$version.lock"
flock -n 9 || exit 75
test ! -e "outputs/${version}_training" || exit 76
printf '%s\n' "$$" > "logs/$version.pid"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${PWD}/research_runs/known_task_trusted_evidence_arbitration_20261005_v8/overlay:${PWD}/research_runs/${version}:${PWD}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME=/root/autodl-tmp/wam_project/.cache/huggingface HF_HUB_CACHE=/root/autodl-tmp/wam_project/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export LIBERO_CONFIG_PATH=${PWD}/.libero TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -B "research_runs/$version/train_command_predicate_utility.py" \
    --output "outputs/${version}_training"
