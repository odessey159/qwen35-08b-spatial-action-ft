set -euo pipefail
repo=/root/qwen35-08b-spatial-action-ft
python_bin="$repo/.venv-train/bin/python"
run_dir="$repo/outputs/qwen35-08b-cot-100k/v0-20260816-102046"
output_dir="$run_dir/section-loss-eval-gpu-bf16-n200-seed42"
status="$repo/run_logs/cot100k_n200_section.status"
cd "$repo"
mkdir -p run_logs
if [[ -e "$status" || -e "$output_dir" ]]; then
  printf '%s\n' "refusing_existing_status_or_output" >&2
  exit 2
fi
finish() {
  code=$?
  if (( code == 0 )); then
    printf '%s\n' "complete finished_at=$(date -Is)" >"$status"
  else
    printf '%s\n' "failed exit=$code finished_at=$(date -Is)" >"$status"
  fi
}
trap finish EXIT
printf '%s\n' "queued waiting_for_pid=16775 queued_at=$(date -Is)" >"$status"
while kill -0 16775 2>/dev/null; do
  sleep 15
done
printf '%s\n' "running started_at=$(date -Is)" >"$status"
"$python_bin" -m training.evaluate_section_losses \
  --run-dir "$run_dir" \
  --base-model-dir /model/ModelScope/Qwen/Qwen3.5-0.8B \
  --val-file training/prepared/cot-100k/val.jsonl \
  --checkpoint-steps 0 11193 \
  --output-dir "$output_dir" \
  --project-root "$repo" \
  --max-samples 200 \
  --sample-seed 42 \
  --device cuda \
  --dtype bfloat16 \
  --cpu-threads 8