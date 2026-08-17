#!/usr/bin/env bash

set -euo pipefail

repo=/root/qwen35-08b-spatial-action-ft
cd "$repo"

mkdir -p run_logs

config=training/config.cot.format-only.100k.server.json
status_file=run_logs/cot100k_format_only_retry.status
pid_file=run_logs/cot100k_format_only_retry.pid
log_file=run_logs/cot100k_format_only_retry.log

printf '%s\n' "$$" > "$pid_file"
printf '%s\n' "running started_at=$(date -Is)" > "$status_file"

finish() {
  exit_code=$?
  set +e
  if (( exit_code == 0 )); then
    run_dir=$(find outputs/qwen35-08b-cot-100k-format-only -mindepth 1 -maxdepth 1 -type d -name 'v0-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    printf '%s\n' "complete run_dir=${run_dir:-unknown} finished_at=$(date -Is)" > "$status_file"
  else
    printf '%s\n' "failed exit=$exit_code finished_at=$(date -Is)" > "$status_file"
  fi
}
trap finish EXIT

if find outputs/qwen35-08b-cot-100k-format-only \
  -mindepth 2 -maxdepth 2 -type d -name checkpoint-11193 -print -quit 2>/dev/null \
  | grep -q .
then
  printf '%s\n' "A complete checkpoint-11193 already exists; skipping duplicate training." >> "$log_file"
  exit 0
fi

.venv-train/bin/python -m training.cli --config "$config" validate >> "$log_file" 2>&1
.venv-train/bin/python -m training.cli --config "$config" check-runtime >> "$log_file" 2>&1
.venv-train/bin/python -m training.cli --config "$config" show-command >> "$log_file" 2>&1
.venv-train/bin/python -m training.cli --config "$config" train >> "$log_file" 2>&1
