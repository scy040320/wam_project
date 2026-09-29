#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/wam_project/cosmos-policy
VERSION=candidate_reranking_task50_state_expansion_20260929_v1
ROOT=outputs/${VERSION}
CODE=research_runs/${VERSION}
PY=.venv/bin/python
BASE=${CODE}/base_collector_shift_calibrated.py
COLLECTOR=research_runs/D18_v9_机制修复与扩采_20260928/collect_snapshot_fork.py
ROLLOUT=research_runs/candidate_reranking_d19_d21_methodv5_coverage_run05_20260929/candidate_outcome_rollout.py
SOURCE_SNAPSHOT=outputs/candidate_reranking_pool_screen_s0_cached_20260929_v4_snapshot
SNAPSHOT=outputs/${VERSION}_snapshot
LOCK=logs/${VERSION}.lock
export LIBERO_CONFIG_PATH="$PWD/.libero"
export HF_HOME=/root/autodl-tmp/wam_project/.cache/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/wam_project/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

if [[ -e "$LOCK" || -e "$ROOT" || -e "$SNAPSHOT" ]]; then
  echo "refuse duplicate task50 state-expansion launch" >&2
  exit 2
fi
mkdir -p "$ROOT" logs
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT
cp "$CODE/candidate_pool_task50_state_expansion_v1.json" "$ROOT/protocol.json"
cp "$CODE/preflight.json" "$ROOT/preflight.json"
cp -al "$SOURCE_SNAPSHOT" "$SNAPSHOT"

tasks=(50 50 50 50 50 50 50 50)
states=(32 33 34 35 36 37 38 39)
targets=(alphabet_soup_1 alphabet_soup_1 alphabet_soup_1 alphabet_soup_1 alphabet_soup_1 alphabet_soup_1 alphabet_soup_1 alphabet_soup_1)
subgoals=(place_soup_in_basket place_soup_in_basket place_soup_in_basket place_soup_in_basket place_soup_in_basket place_soup_in_basket place_soup_in_basket place_soup_in_basket)

run_clean() {
  local i=$1 task=${tasks[$1]} state=${states[$1]} seed=$((65000 + ${tasks[$1]} * 10 + ${states[$1]}))
  local shard="$ROOT/clean_evidence/task${task}_state${state}_moment0"
  local outcome="$ROOT/clean_outcomes/task${task}_state${state}_moment0"
  "$PY" "$COLLECTOR" --base-collector "$BASE" --output-dir "$shard" \
    --states "$state" --moment-id 0 --task-id "$task" --benchmark-suite libero_90 \
    --target-object "${targets[$i]}" --target-intervention-kind rigid_translation \
    --active-subgoal "${subgoals[$i]}" --silver-schema candidate_pool_task50_state_expansion_v1 \
    --schema-parent d18_v18i_frozen --clean-replicates 1 --split val --base-seed "$seed" \
    --snapshot "$SNAPSHOT" --conditions clean
  "$PY" "$ROLLOUT" --snapshot "$SNAPSHOT" --shard "$shard" --output "$outcome" \
    --base-collector "$BASE" --snapshot-collector "$COLLECTOR" --task-id "$task" \
    --benchmark-suite libero_90 --state "$state" --base-seed "$seed" --k 4 --max-policy-steps 256
  printf 'clean\t%s\t%s\n' "$task" "$state" >> "$ROOT/completion.tsv"
}

run_stress() {
  local i=$1 task=${tasks[$1]} state=${states[$1]} seed=$((66000 + ${tasks[$1]} * 10 + ${states[$1]}))
  local shard="$ROOT/stress_evidence/task${task}_state${state}_moment0"
  local outcome="$ROOT/stress_outcomes/task${task}_state${state}_moment0"
  CFWAM_SHIFT_SCALE=1.0 "$PY" "$COLLECTOR" --base-collector "$BASE" --output-dir "$shard" \
    --states "$state" --moment-id 0 --task-id "$task" --benchmark-suite libero_90 \
    --target-object "${targets[$i]}" --target-intervention-kind rigid_translation \
    --active-subgoal "${subgoals[$i]}" --silver-schema candidate_pool_task50_state_expansion_v1 \
    --schema-parent d18_v18i_frozen --clean-replicates 1 --split val --base-seed "$seed" \
    --snapshot "$SNAPSHOT" --conditions object_shift
  CFWAM_SHIFT_SCALE=1.0 "$PY" "$ROLLOUT" --snapshot "$SNAPSHOT" --shard "$shard" --output "$outcome" \
    --base-collector "$BASE" --snapshot-collector "$COLLECTOR" --task-id "$task" \
    --benchmark-suite libero_90 --state "$state" --base-seed "$seed" --k 8 --max-policy-steps 256
  printf 'stress\t%s\t%s\n' "$task" "$state" >> "$ROOT/completion.tsv"
}

run_lane() {
  local phase=$1 lane=$2
  for i in "${!tasks[@]}"; do
    if (( i % 3 != lane )); then continue; fi
    if [[ "$phase" == clean ]]; then
      run_clean "$i"
    elif grep -qx "${tasks[$i]} ${states[$i]}" "$ROOT/eligible_cells.txt"; then
      run_stress "$i"
    fi
  done
}

printf '{"phase":"clean_competence","cells_total":8,"cells_complete":0,"k":4,"lanes":3}\n' > "$ROOT/status.json"
run_lane clean 0 > "logs/${VERSION}_clean_lane0.log" 2>&1 & p0=$!
run_lane clean 1 > "logs/${VERSION}_clean_lane1.log" 2>&1 & p1=$!
run_lane clean 2 > "logs/${VERSION}_clean_lane2.log" 2>&1 & p2=$!
wait "$p0"; wait "$p1"; wait "$p2"

"$PY" "$CODE/analyze_competence_conditioned_pool.py" --protocol "$ROOT/protocol.json" \
  --clean-outcomes "$ROOT/clean_outcomes" --output "$ROOT/clean_report.json"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("\n".join(f"{t} {s}" for t,s in d["new_eligible_cells"]+d["sentinel_eligible_cells"]))' \
  "$ROOT/clean_report.json" > "$ROOT/eligible_cells.txt"
eligible_count=$(wc -l < "$ROOT/eligible_cells.txt")
printf '{"phase":"fixed_stress","cells_total":%s,"cells_complete":0,"k":8,"lanes":3}\n' "$eligible_count" > "$ROOT/status.json"
if (( eligible_count > 0 )); then
  run_lane stress 0 > "logs/${VERSION}_stress_lane0.log" 2>&1 & p0=$!
  run_lane stress 1 > "logs/${VERSION}_stress_lane1.log" 2>&1 & p1=$!
  run_lane stress 2 > "logs/${VERSION}_stress_lane2.log" 2>&1 & p2=$!
  wait "$p0"; wait "$p1"; wait "$p2"
fi

"$PY" "$CODE/analyze_competence_conditioned_pool.py" --protocol "$ROOT/protocol.json" \
  --clean-outcomes "$ROOT/clean_outcomes" --stress-outcomes "$ROOT/stress_outcomes" \
  --output "$ROOT/final_report.json"
sha256sum "$ROOT/protocol.json" "$ROOT/preflight.json" "$ROOT/clean_report.json" \
  "$ROOT/final_report.json" > "$ROOT/SHA256SUMS_small.txt"
printf '{"phase":"complete","cells_total":8,"eligible_cells":%s,"analysis_complete":true}\n' "$eligible_count" > "$ROOT/status.json"
echo task50_state_expansion_complete
