#!/usr/bin/env bash

set -euo pipefail

repo=/root/qwen35-08b-spatial-action-ft
python_bin="$repo/.venv-train/bin/python"
run_dir="$repo/outputs/qwen35-08b-cot-100k/v0-20260816-102046"
generation_dir="$run_dir/gpu-generation"
before_predictions="$generation_dir/val-all-step0.jsonl"
after_predictions="$generation_dir/val-all-step11193.jsonl"
comparison_dir="$run_dir/pre-post-comparison-full-val"
status_file="$repo/run_logs/cot100k_step0_full_comparison.status"
gpu_log="$repo/run_logs/cot100k_step0_full_comparison_gpu.csv"

cd "$repo"
mkdir -p "$generation_dir" "$comparison_dir" run_logs
expected="$($python_bin -c 'import json; print(json.load(open("training/prepared/cot-100k/manifest.json"))["validation_samples"])')"

printf '%s\n' "starting expected=$expected started_at=$(date -Is)" >"$status_file"
finish() {
  exit_code=$?
  set +e
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if (( exit_code == 0 )); then
    printf '%s\n' "complete samples=$expected finished_at=$(date -Is)" >"$status_file"
  else
    printf '%s\n' "failed exit=$exit_code finished_at=$(date -Is)" >"$status_file"
  fi
}
trap finish EXIT

after_count=0
if [[ -f "$after_predictions" ]]; then
  after_count="$(wc -l <"$after_predictions")"
fi
if [[ "$after_count" -ne "$expected" ]]; then
  printf '%s\n' "Training-after predictions are incomplete: $after_count/$expected" >&2
  exit 2
fi

before_count=0
if [[ -f "$before_predictions" ]]; then
  before_count="$(wc -l <"$before_predictions")"
fi
if [[ "$before_count" -ne "$expected" ]]; then
  temporary="${before_predictions}.tmp"
  rm -f "$temporary"
  printf '%s\n' "generating batch_size=16 dtype=bfloat16 attention=sdpa started_at=$(date -Is)" >"$status_file"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 5 >"$gpu_log" 2>&1 &
  monitor_pid=$!
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m training.generate_cpu_predictions \
      --checkpoint /model/ModelScope/Qwen/Qwen3.5-0.8B \
      --val-file training/prepared/cot-100k/val.jsonl \
      --raw-data exp0/new100k_clean_data/samples.jsonl \
      --manifest training/prepared/cot-100k/manifest.json \
      --output "$temporary" \
      --project-root "$repo" \
      --selection all \
      --image-mode correct \
      --max-new-tokens 256 \
      --batch-size 16 \
      --device cuda \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --overwrite
  generated_count="$(wc -l <"$temporary")"
  if [[ "$generated_count" -ne "$expected" ]]; then
    printf '%s\n' "Training-before predictions are incomplete: $generated_count/$expected" >&2
    exit 3
  fi
  mv -f "$temporary" "$before_predictions"
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  unset monitor_pid
fi

printf '%s\n' "scoring samples=$expected started_at=$(date -Is)" >"$status_file"
"$python_bin" -m training.compare_pre_post_metrics \
  --raw-data exp0/new100k_clean_data/samples.jsonl \
  --val-file training/prepared/cot-100k/val.jsonl \
  --manifest training/prepared/cot-100k/manifest.json \
  --before-predictions "$before_predictions" \
  --after-predictions "$after_predictions" \
  --output-dir "$comparison_dir"
