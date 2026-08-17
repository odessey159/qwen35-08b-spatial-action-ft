#!/usr/bin/env bash

set -u

repo=/root/qwen35-08b-spatial-action-ft
cd "$repo" || exit 1

mkdir -p run_logs

status_file=run_logs/procthor_val_5k_linux64.status
pid_file=run_logs/procthor_val_5k_linux64.pid
summary_log=run_logs/procthor_val_5k_linux64.log

printf '%s\n' "running started_at=$(date -Is)" > "$status_file"
printf '%s\n' "$$" > "$pid_file"

declare -a shard_pids=()
declare -a shard_indexes=()

for shard_index in 0 1 2 3; do
  report="exp0/procthor_val_5k_shard_data/shard_${shard_index}/generation_report.json"
  if .venv-train/bin/python -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if x.get("complete") and x.get("sample_count") == x.get("expected_sample_count") else 1)' \
    "$report"
  then
    printf '%s\n' "shard_${shard_index}=already_complete" >> "$summary_log"
    continue
  fi

  shard_log="run_logs/procthor_val_5k_linux64_shard_${shard_index}.log"
  (
    export AI2THOR_PLATFORM=Linux64
    exec xvfb-run -a -s '-screen 0 1024x768x24' \
      .venv/bin/python -m exp0.generate_data \
      --config exp0/generator_config.procthor_val_5k.json \
      --shard-index "$shard_index" \
      --shard-count 4
  ) > "$shard_log" 2>&1 &
  shard_pids+=("$!")
  shard_indexes+=("$shard_index")
  printf '%s\n' "shard_${shard_index}=started pid=$! log=$shard_log" >> "$summary_log"
done

failures=0
for array_index in "${!shard_pids[@]}"; do
  shard_pid="${shard_pids[$array_index]}"
  shard_index="${shard_indexes[$array_index]}"
  if wait "$shard_pid"; then
    printf '%s\n' "shard_${shard_index}=complete" >> "$summary_log"
  else
    exit_code=$?
    printf '%s\n' "shard_${shard_index}=failed exit=$exit_code" >> "$summary_log"
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  printf '%s\n' "failed generation_failures=$failures finished_at=$(date -Is)" > "$status_file"
  exit 1
fi

if ! .venv-train/bin/python -m training.clean_raw_shards \
  --shard-root exp0/procthor_val_5k_shard_data \
  --output-dir exp0/procthor_val_5k_clean_data \
  --shards 4 \
  --overwrite >> "$summary_log" 2>&1
then
  printf '%s\n' "failed cleaning_exit=1 finished_at=$(date -Is)" > "$status_file"
  exit 1
fi

if ! .venv-train/bin/python -m training.prepare_evaluation_dataset \
  --source exp0/procthor_val_5k_clean_data/samples.jsonl \
  --output-dir training/prepared/procthor-val-5k-test \
  --overwrite >> "$summary_log" 2>&1
then
  printf '%s\n' "failed prepare_exit=1 finished_at=$(date -Is)" > "$status_file"
  exit 1
fi

clean_lines=$(wc -l < exp0/procthor_val_5k_clean_data/samples.jsonl)
prepared_lines=$(wc -l < training/prepared/procthor-val-5k-test/val.jsonl)
printf '%s\n' \
  "complete clean_lines=$clean_lines prepared_lines=$prepared_lines finished_at=$(date -Is)" \
  > "$status_file"
