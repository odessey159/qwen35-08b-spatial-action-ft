#!/usr/bin/env bash

set -euo pipefail

repo=/root/qwen35-08b-spatial-action-ft
cd "$repo"

mkdir -p run_logs

status_file=run_logs/cot100k_format_only_eval_retry.status
pid_file=run_logs/cot100k_format_only_eval_retry.pid
log_file=run_logs/cot100k_format_only_eval_retry.log

printf '%s\n' "$$" > "$pid_file"
printf '%s\n' "waiting_for_training started_at=$(date -Is)" > "$status_file"

finish() {
  exit_code=$?
  set +e
  if (( exit_code == 0 )); then
    printf '%s\n' "complete finished_at=$(date -Is)" > "$status_file"
  else
    printf '%s\n' "failed exit=$exit_code finished_at=$(date -Is)" > "$status_file"
  fi
}
trap finish EXIT

while true; do
  training_status=$(cat run_logs/cot100k_format_only_retry.status 2>/dev/null || true)
  case "$training_status" in
    complete*) break ;;
    failed*)
      printf '%s\n' "format-only training failed: $training_status" >> "$log_file"
      exit 1
      ;;
  esac
  sleep 60
done

checkpoint=$(find outputs/qwen35-08b-cot-100k-format-only \
  -mindepth 2 -maxdepth 2 -type d -name checkpoint-11193 \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
if [[ -z "$checkpoint" ]]; then
  printf '%s\n' "No complete format-only checkpoint-11193 found." >> "$log_file"
  exit 1
fi
run_dir=${checkpoint%/checkpoint-11193}

expected=$(.venv-train/bin/python -c 'import json; print(json.load(open("training/prepared/cot-100k/manifest.json"))["validation_samples"])')
generation_dir="$run_dir/gpu-generation"
correct="$generation_dir/val-all-step11193.jsonl"
a_prime="$generation_dir/val-aprime-step11193.jsonl"
mkdir -p "$generation_dir"

generate() {
  image_mode=$1
  output=$2
  if [[ -f "$output" ]] && [[ $(wc -l < "$output") -eq $expected ]]; then
    printf '%s\n' "Skipping complete output: $output" >> "$log_file"
    return
  fi
  temporary="${output}.tmp"
  .venv-train/bin/python -m training.generate_cpu_predictions \
    --checkpoint "$checkpoint" \
    --val-file training/prepared/cot-100k/val.jsonl \
    --raw-data exp0/new100k_clean_data/samples.jsonl \
    --manifest training/prepared/cot-100k/manifest.json \
    --output "$temporary" \
    --project-root "$repo" \
    --selection all \
    --image-mode "$image_mode" \
    --max-new-tokens 256 \
    --batch-size 16 \
    --device cuda \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --overwrite >> "$log_file" 2>&1
  [[ $(wc -l < "$temporary") -eq $expected ]]
  mv -f "$temporary" "$output"
}

exec 9>run_logs/gpu_eval.lock
printf '%s\n' "waiting_for_gpu_lock run_dir=$run_dir" > "$status_file"
flock -x 9
printf '%s\n' "evaluating run_dir=$run_dir started_at=$(date -Is)" > "$status_file"

generate correct "$correct"
generate a-prime "$a_prime"

.venv-train/bin/python -m training.evaluate_in_domain_predictions \
  --raw-data exp0/new100k_clean_data/samples.jsonl \
  --val-file training/prepared/cot-100k/val.jsonl \
  --manifest training/prepared/cot-100k/manifest.json \
  --predictions "$correct" \
  --a-prime-predictions "$a_prime" \
  --oracle-raw-data exp0/new100k_clean_data/samples.jsonl \
  --oracle-manifest training/prepared/cot-100k/manifest.json \
  --output-dir "$run_dir/in-domain-full-val" >> "$log_file" 2>&1
