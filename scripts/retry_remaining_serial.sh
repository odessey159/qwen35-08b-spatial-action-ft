#!/usr/bin/env bash

set -euo pipefail

repo=/root/qwen35-08b-spatial-action-ft
cd "$repo"

mkdir -p run_logs

status_file=run_logs/retry_remaining_serial.status
pid_file=run_logs/retry_remaining_serial.pid
log_file=run_logs/retry_remaining_serial.log

printf '%s\n' "$$" > "$pid_file"
printf '%s\n' "waiting_for_fast8 started_at=$(date -Is)" > "$status_file"

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
      printf '%s\n' "Fast8 generation failed: $data_status" >> "$log_file"
      exit 1
      ;;
  esac
  sleep 30
done

printf '%s\n' "running_heldout_eval started_at=$(date -Is)" > "$status_file"
bash scripts/retry_procthor_val_5k_eval.sh >> "$log_file" 2>&1

printf '%s\n' "running_format_only_training started_at=$(date -Is)" > "$status_file"
bash scripts/retry_cot100k_format_only.sh >> "$log_file" 2>&1

printf '%s\n' "running_format_only_eval started_at=$(date -Is)" > "$status_file"
bash scripts/retry_cot100k_format_only_eval.sh >> "$log_file" 2>&1
