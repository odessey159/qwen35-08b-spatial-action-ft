#!/usr/bin/env bash

set -euo pipefail

repo=/root/qwen35-08b-spatial-action-ft
cd "$repo"

mkdir -p run_logs

status_file=run_logs/procthor_val_5k_eval_linux64.status
pid_file=run_logs/procthor_val_5k_eval_linux64.pid
log_file=run_logs/procthor_val_5k_eval_linux64.log

printf '%s\n' "$$" > "$pid_file"
printf '%s\n' "waiting_for_data started_at=$(date -Is)" > "$status_file"

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
  data_status=$(cat run_logs/procthor_val_5k_fast8.status 2>/dev/null || true)
  case "$data_status" in
    complete*) break ;;
    failed*)
      printf '%s\n' "ProcTHOR data generation failed: $data_status" >> "$log_file"
      exit 1
      ;;
  esac
  sleep 60
done

run_dir=outputs/qwen35-08b-cot-100k/v0-20260816-102046
checkpoint="$run_dir/checkpoint-11193"
expected=$(.venv-train/bin/python -c 'import json; print(json.load(open("training/prepared/procthor-val-5k-test/manifest.json"))["validation_samples"])')
generation_dir="$run_dir/gpu-generation"
correct="$generation_dir/test-procthor-val5k-all-step11193.jsonl"
a_prime="$generation_dir/test-procthor-val5k-aprime-step11193.jsonl"
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
    --val-file training/prepared/procthor-val-5k-test/val.jsonl \
    --raw-data exp0/procthor_val_5k_clean_data/samples.jsonl \
    --manifest training/prepared/procthor-val-5k-test/manifest.json \
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
printf '%s\n' "waiting_for_gpu_lock samples=$expected" > "$status_file"
flock -x 9
printf '%s\n' "evaluating samples=$expected started_at=$(date -Is)" > "$status_file"

generate correct "$correct"
generate a-prime "$a_prime"

.venv-train/bin/python -m training.evaluate_in_domain_predictions \
  --raw-data exp0/procthor_val_5k_clean_data/samples.jsonl \
  --val-file training/prepared/procthor-val-5k-test/val.jsonl \
  --manifest training/prepared/procthor-val-5k-test/manifest.json \
  --predictions "$correct" \
  --a-prime-predictions "$a_prime" \
  --oracle-raw-data exp0/new100k_clean_data/samples.jsonl \
  --oracle-manifest training/prepared/cot-100k/manifest.json \
  --output-dir "$run_dir/in-domain-heldout-procthor-val5k" >> "$log_file" 2>&1
