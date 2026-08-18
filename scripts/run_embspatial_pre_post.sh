#!/usr/bin/env bash

set -euo pipefail

repo=/root/qwen35-08b-spatial-action-ft
python_bin="$repo/.venv-train/bin/python"
run_dir="$repo/outputs/qwen35-08b-cot-100k/v0-20260816-102046"
base_model=/model/ModelScope/Qwen/Qwen3.5-0.8B
after_model="$run_dir/checkpoint-11193"
dataset_dir="$repo/data/embspatial-bench"
dataset_file="$dataset_dir/embspatial_bench.json"
output_dir="$run_dir/ood-embspatial-bench"
status_file="$repo/run_logs/embspatial_pre_post.status"

cd "$repo"
mkdir -p "$dataset_dir" "$output_dir" run_logs

printf '%s\n' "starting started_at=$(date -Is)" >"$status_file"
finish() {
  exit_code=$?
  set +e
  if (( exit_code == 0 )); then
    printf '%s\n' "complete finished_at=$(date -Is)" >"$status_file"
  else
    printf '%s\n' "failed exit=$exit_code finished_at=$(date -Is)" >"$status_file"
  fi
}
trap finish EXIT

for required in "$python_bin" "$base_model/config.json" "$after_model/config.json"; do
  if [[ ! -e "$required" ]]; then
    printf '%s\n' "Missing required path: $required" >&2
    exit 2
  fi
done

if [[ ! -s "$dataset_file" ]]; then
  printf '%s\n' "downloading_dataset started_at=$(date -Is)" >"$status_file"
  "$python_bin" -c \
    'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="Phineas476/EmbSpatial-Bench", filename="embspatial_bench.json", repo_type="dataset", local_dir="data/embspatial-bench")'
fi

dataset_count="$($python_bin -c 'import json; print(len(json.load(open("data/embspatial-bench/embspatial_bench.json", encoding="utf-8"))))')"
if [[ "$dataset_count" -ne 3640 ]]; then
  printf '%s\n' "Unexpected EmbSpatial-Bench sample count: $dataset_count (expected 3640)" >&2
  exit 3
fi
sha256sum "$dataset_file" >"$output_dir/dataset.sha256"

exec 9>run_logs/gpu_eval.lock
printf '%s\n' "waiting_for_gpu_lock samples=$dataset_count" >"$status_file"
flock -x 9

evaluate() {
  label=$1
  checkpoint=$2
  predictions="$output_dir/${label}.jsonl"
  summary="$output_dir/${label}-summary.json"
  resume_args=()
  if [[ -s "$predictions" ]]; then
    resume_args+=(--resume)
  fi
  printf '%s\n' "evaluating_$label samples=$dataset_count started_at=$(date -Is)" >"$status_file"
  "$python_bin" -m training.evaluate_embspatial \
    --checkpoint "$checkpoint" \
    --dataset "$dataset_file" \
    --output "$predictions" \
    --summary "$summary" \
    --batch-size 16 \
    --max-new-tokens 8 \
    --device cuda \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    "${resume_args[@]}"
}

evaluate before "$base_model"
evaluate after "$after_model"

printf '%s\n' "comparing samples=$dataset_count started_at=$(date -Is)" >"$status_file"
"$python_bin" -m training.evaluate_embspatial \
  --compare-before "$output_dir/before.jsonl" \
  --compare-after "$output_dir/after.jsonl" \
  --summary "$output_dir/comparison.json"

