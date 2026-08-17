from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CANDIDATES = [
    {"name": "b2_ga2_gc_w4", "batch": 2, "accum": 2, "gc": True, "workers": 4},
    {"name": "b4_ga1_gc_w4", "batch": 4, "accum": 1, "gc": True, "workers": 4},
    {"name": "b8_ga1_gc_w4", "batch": 8, "accum": 1, "gc": True, "workers": 4},
    {"name": "b12_ga1_gc_w4", "batch": 12, "accum": 1, "gc": True, "workers": 4},
    {"name": "b4_ga1_nogc_w4", "batch": 4, "accum": 1, "gc": False, "workers": 4},
    {"name": "b6_ga1_nogc_w4", "batch": 6, "accum": 1, "gc": False, "workers": 4},
    {"name": "b7_ga1_nogc_w4", "batch": 7, "accum": 1, "gc": False, "workers": 4},
    {"name": "b8_ga1_nogc_w4", "batch": 8, "accum": 1, "gc": False, "workers": 4},
    {
        "name": "b4_ga1_nogc_w8",
        "batch": 4,
        "accum": 1,
        "gc": False,
        "workers": 8,
        "persistent": True,
        "prefetch": 4,
    },
    {
        "name": "b4_ga1_nogc_w8_fused",
        "batch": 4,
        "accum": 1,
        "gc": False,
        "workers": 8,
        "persistent": True,
        "prefetch": 4,
        "optim": "adamw_torch_fused",
    },
    {
        "name": "b8_nogc_eval4",
        "batch": 8,
        "eval_batch": 4,
        "accum": 1,
        "gc": False,
        "workers": 4,
        "eval_strategy": "steps",
        "eval_steps": 1,
    },
    {
        "name": "b8_nogc_eval8",
        "batch": 8,
        "eval_batch": 8,
        "accum": 1,
        "gc": False,
        "workers": 4,
        "eval_strategy": "steps",
        "eval_steps": 1,
    },
    {
        "name": "b8_nogc_eval16",
        "batch": 8,
        "eval_batch": 16,
        "accum": 1,
        "gc": False,
        "workers": 4,
        "eval_strategy": "steps",
        "eval_steps": 1,
    },
    {
        "name": "b8_nogc_eval32",
        "batch": 8,
        "eval_batch": 32,
        "accum": 1,
        "gc": False,
        "workers": 4,
        "eval_strategy": "steps",
        "eval_steps": 1,
    },
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _gpu_sample() -> tuple[float, float, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout
        values = [float(value.strip()) for value in output.splitlines()[0].split(",")]
        return values[0], values[1], values[2]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _find_logging(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("logging.jsonl"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _parse_metrics(output_dir: Path) -> dict[str, Any]:
    path = _find_logging(output_dir)
    if path is None:
        return {}
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    final = next((row for row in reversed(rows) if "train_runtime" in row), {})
    step_rows = [row for row in rows if "loss" in row and "global_step/max_steps" in row]
    eval_rows = [row for row in rows if "eval_runtime" in row]
    last_eval = eval_rows[-1] if eval_rows else {}
    return {
        "logging_path": str(path),
        "train_runtime": final.get("train_runtime"),
        "train_samples_per_second": final.get("train_samples_per_second"),
        "train_steps_per_second": final.get("train_steps_per_second"),
        "train_loss": final.get("train_loss"),
        "reported_memory_gib": final.get("memory(GiB)", final.get("memory")),
        "last_step": step_rows[-1].get("global_step/max_steps") if step_rows else None,
        "last_train_speed_s_per_it": step_rows[-1].get("train_speed(s/it)") if step_rows else None,
        "eval_runtime": last_eval.get("eval_runtime"),
        "eval_samples_per_second": last_eval.get("eval_samples_per_second"),
        "eval_steps_per_second": last_eval.get("eval_steps_per_second"),
        "eval_loss": last_eval.get("eval_loss"),
    }


def _summarize_gpu(samples: list[tuple[float, float, float]]) -> dict[str, Any]:
    active = [sample for sample in samples if sample[1] >= 4096]
    values = active or samples
    if not values:
        return {"gpu_samples": 0}
    utilization = [sample[0] for sample in values]
    memory = [sample[1] for sample in values]
    power = [sample[2] for sample in values]
    return {
        "gpu_samples": len(values),
        "gpu_utilization_mean": round(statistics.fmean(utilization), 2),
        "gpu_utilization_max": max(utilization),
        "memory_used_mean_mib": round(statistics.fmean(memory), 1),
        "memory_used_max_mib": max(memory),
        "power_mean_w": round(statistics.fmean(power), 2),
        "power_max_w": max(power),
    }


def run_candidate(
    repository: Path,
    base_config: Path,
    prepared_dir: Path,
    work_dir: Path,
    candidate: dict[str, Any],
    max_steps: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    name = str(candidate["name"])
    output_dir = work_dir / "outputs" / name
    config_path = work_dir / "configs" / f"{name}.json"
    training: dict[str, Any] = {
        "max_steps": max_steps,
        "num_train_epochs": None,
        "per_device_train_batch_size": candidate["batch"],
        "per_device_eval_batch_size": candidate.get("eval_batch", candidate["batch"]),
        "gradient_accumulation_steps": candidate["accum"],
        "gradient_checkpointing": candidate["gc"],
        "dataloader_num_workers": candidate["workers"],
        "eval_on_start": False,
        "eval_strategy": candidate.get("eval_strategy", "no"),
        "save_strategy": "no",
        "save_only_model": True,
        "logging_steps": 1,
        "warmup_ratio": 0.0,
        "report_to": "none",
    }
    if candidate.get("eval_steps") is not None:
        training["eval_steps"] = candidate["eval_steps"]
    if candidate.get("persistent") is not None:
        training["dataloader_persistent_workers"] = candidate["persistent"]
    if candidate.get("prefetch") is not None:
        training["dataloader_prefetch_factor"] = candidate["prefetch"]
    if candidate.get("optim"):
        training["optim"] = candidate["optim"]
    with (prepared_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        prepared_manifest = json.load(handle)
    source_dataset = prepared_manifest.get("source_dataset")
    if not isinstance(source_dataset, str) or not Path(source_dataset).is_file():
        raise FileNotFoundError(f"Prepared manifest has no valid source_dataset: {source_dataset}")
    config = {
        "extends": str(base_config),
        "model": {"output_dir": str(output_dir)},
        "data": {
            "source_dataset": source_dataset,
            "prepared_dir": str(prepared_dir),
        },
        "training": training,
    }
    _write_json(config_path, config)
    log_path = work_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "training.cli",
        "--config",
        str(config_path),
        "train",
    ]
    started = time.monotonic()
    gpu_samples: list[tuple[float, float, float]] = []
    timed_out = False
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=repository,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        while process.poll() is None:
            if time.monotonic() - started > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            sample = _gpu_sample()
            if sample is not None:
                gpu_samples.append(sample)
            time.sleep(0.5)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    result = {
        **candidate,
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_time_seconds": round(elapsed, 2),
        "oom": "out of memory" in log_tail.casefold(),
        "log_path": str(log_path),
        **_parse_metrics(output_dir),
        **_summarize_gpu(gpu_samples),
    }
    if return_code != 0:
        result["error_tail"] = log_tail[-2000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full-finetuning throughput configurations")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--names", nargs="*", help="Optional candidate names to run")
    args = parser.parse_args()
    repository = args.repository.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    candidates = CANDIDATES
    if args.names:
        requested = set(args.names)
        candidates = [candidate for candidate in CANDIDATES if candidate["name"] in requested]
        missing = requested - {candidate["name"] for candidate in candidates}
        if missing:
            raise ValueError(f"Unknown benchmark candidate names: {sorted(missing)}")
    for candidate in candidates:
        print(f"BENCHMARK_START {candidate['name']}", flush=True)
        result = run_candidate(
            repository,
            args.base_config.resolve(),
            args.prepared_dir.resolve(),
            work_dir,
            candidate,
            args.max_steps,
            args.timeout_seconds,
        )
        results.append(result)
        _write_json(work_dir / "results.json", {"max_steps": args.max_steps, "results": results})
        print("BENCHMARK_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    successful = [
        result
        for result in results
        if result["return_code"] == 0 and result.get("train_samples_per_second") is not None
    ]
    summary = {
        "max_steps": args.max_steps,
        "results": results,
        "fastest": max(successful, key=lambda item: item["train_samples_per_second"])["name"]
        if successful
        else None,
    }
    _write_json(work_dir / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
