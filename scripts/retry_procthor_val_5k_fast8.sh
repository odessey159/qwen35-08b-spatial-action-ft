#!/usr/bin/env bash

set -u

repo=/root/qwen35-08b-spatial-action-ft
cd "$repo" || exit 1

mkdir -p run_logs

status_file=run_logs/procthor_val_5k_fast8.status
pid_file=run_logs/procthor_val_5k_fast8.pid
summary_log=run_logs/procthor_val_5k_fast8.log

printf '%s\n' "$$" > "$pid_file"
printf '%s\n' "running started_at=$(date -Is) workers=8 backend=Linux64-NVIDIA-Xorg" > "$status_file"

declare -a shard_pids=()
declare -a shard_indexes=()

for shard_index in 0 1 2 3 4 5 6 7; do
  report="exp0/procthor_val_5k_cloud8_shard_data/shard_${shard_index}/generation_report.json"
  if .venv-train/bin/python -c \
    'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()) if p.is_file() else {}; expected=x.get("expected_sample_count"); raise SystemExit(0 if p.is_file() and expected is not None and x.get("sample_count") == expected else 1)' \
    "$report"
  then
    printf '%s\n' "shard_${shard_index}=already_full" >> "$summary_log"
    continue
  fi

  shard_log="run_logs/procthor_val_5k_fast8_shard_${shard_index}.log"
  (
    export AI2THOR_PLATFORM=Linux64
    export DISPLAY=:99
    export XDG_RUNTIME_DIR=/tmp/runtime-root
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export LD_LIBRARY_PATH="/root/qwen35-08b-spatial-action-ft/nvidia-595.80-userspace:/tmp/xorg-headless-test/root/usr/lib/x86_64-linux-gnu"
    exec .venv/bin/python -m exp0.generate_data \
      --config exp0/generator_config.procthor_val_5k_cloud8.json \
      --shard-index "$shard_index" \
      --shard-count 8
  ) > "$shard_log" 2>&1 &
  shard_pids+=("$!")
  shard_indexes+=("$shard_index")
  printf '%s\n' "shard_${shard_index}=started pid=$! log=$shard_log" >> "$summary_log"
done

for array_index in "${!shard_pids[@]}"; do
  shard_pid="${shard_pids[$array_index]}"
  shard_index="${shard_indexes[$array_index]}"
  if wait "$shard_pid"; then
    printf '%s\n' "shard_${shard_index}=exit_0" >> "$summary_log"
  else
    exit_code=$?
    printf '%s\n' "shard_${shard_index}=exit_$exit_code; validating saved rows" >> "$summary_log"
  fi
done

incomplete=0
for shard_index in 0 1 2 3 4 5 6 7; do
  shard_dir="exp0/procthor_val_5k_cloud8_shard_data/shard_${shard_index}"
  if ! .venv-train/bin/python -c \
    'import json, pathlib, sys; d=pathlib.Path(sys.argv[1]); r=json.loads((d/"generation_report.json").read_text()); n=sum(1 for s in (d/"samples.jsonl").open() if s.strip()); expected=int(r["expected_sample_count"]); reported=int(r["sample_count"]); print(f"{d.name}: rows={n} reported={reported} expected={expected} complete={r.get(chr(99)+chr(111)+chr(109)+chr(112)+chr(108)+chr(101)+chr(116)+chr(101))}"); raise SystemExit(0 if n == expected and reported == expected else 1)' \
    "$shard_dir" >> "$summary_log" 2>&1
  then
    incomplete=$((incomplete + 1))
  fi
done

if (( incomplete > 0 )); then
  printf '%s\n' "failed incomplete_shards=$incomplete finished_at=$(date -Is)" > "$status_file"
  exit 1
fi

if ! .venv-train/bin/python -m training.clean_raw_shards \
  --shard-root exp0/procthor_val_5k_cloud8_shard_data \
  --output-dir exp0/procthor_val_5k_clean_data \
  --shards 8 \
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
