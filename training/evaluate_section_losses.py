from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SECTIONS = ("state", "plan", "action")
REMOTE_PROJECT_PREFIX = "/root/qwen35-08b-spatial-action-ft/"


@dataclass(frozen=True)
class TokenSpan:
    section: str
    weight: float
    supervised_start: int
    start: int
    body_start: int
    body_end: int
    end: int

    def validate(self, sequence_length: int) -> None:
        if not (
            1
            <= self.supervised_start
            <= self.start
            <= self.body_start
            < self.body_end
            <= self.end
            <= sequence_length
        ):
            raise ValueError(f"Invalid token span: {self}, sequence_length={sequence_length}")


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    correct_tokens: int = 0
    token_count: int = 0
    exact_sequences: int = 0
    sequence_count: int = 0

    def update(self, losses: Iterable[float], correct: Iterable[bool]) -> None:
        loss_values = list(losses)
        correct_values = list(correct)
        if len(loss_values) != len(correct_values) or not loss_values:
            raise ValueError("losses and correct must have the same non-zero length")
        self.loss_sum += sum(float(value) for value in loss_values)
        self.correct_tokens += sum(bool(value) for value in correct_values)
        self.token_count += len(loss_values)
        self.exact_sequences += int(all(correct_values))
        self.sequence_count += 1

    @property
    def loss(self) -> float:
        return self.loss_sum / self.token_count if self.token_count else math.nan

    @property
    def perplexity(self) -> float:
        return math.exp(min(self.loss, 50.0)) if self.token_count else math.nan

    @property
    def token_accuracy(self) -> float:
        return self.correct_tokens / self.token_count if self.token_count else math.nan

    @property
    def exact_match(self) -> float:
        return self.exact_sequences / self.sequence_count if self.sequence_count else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_sum": self.loss_sum,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "correct_tokens": self.correct_tokens,
            "token_count": self.token_count,
            "token_accuracy": self.token_accuracy,
            "exact_sequences": self.exact_sequences,
            "sequence_count": self.sequence_count,
            "exact_match": self.exact_match,
        }


def section_body_char_span(section: str, content: str) -> tuple[int, int]:
    stripped = content.strip()
    opening = f"<{section}>"
    closing = f"</{section}>"
    if not stripped.startswith(opening) or closing not in stripped:
        raise ValueError(f"Malformed <{section}> section")
    start = len(opening)
    end = stripped.rindex(closing)
    while start < end and stripped[start].isspace():
        start += 1
    while end > start and stripped[end - 1].isspace():
        end -= 1
    if start == end:
        raise ValueError(f"Empty <{section}> body")
    return start, end


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def select_rows(
    rows: list[dict[str, Any]], max_samples: int | None, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(rows):
        return list(enumerate(rows))
    indices = sorted(random.Random(seed).sample(range(len(rows)), max_samples))
    return [(index, rows[index]) for index in indices]


def _checkpoint_file_target(name: str) -> str | None:
    suffix = ".codex_sync_part"
    target = name[: -len(suffix)] if name.endswith(suffix) else name
    keep = (
        target.endswith(".json")
        or target.endswith(".jinja")
        or target.endswith(".safetensors")
        or target.endswith(".model")
        or target.endswith(".txt")
    )
    if target in {"trainer_state.json", "args.json"}:
        keep = False
    return target if keep else None


def materialize_checkpoint_view(checkpoint: Path, view: Path) -> Path:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    sources: dict[str, Path] = {}
    has_temporary_names = False
    for source in checkpoint.iterdir():
        if not source.is_file():
            continue
        target_name = _checkpoint_file_target(source.name)
        if target_name is None:
            continue
        sources[target_name] = source
        has_temporary_names |= source.name.endswith(".codex_sync_part")
    required = {"config.json", "tokenizer.json"}
    missing = sorted(required - sources.keys())
    if missing:
        raise FileNotFoundError(f"{checkpoint} is missing {missing}")
    has_single_weight = "model.safetensors" in sources
    has_sharded_weight = (
        "model.safetensors.index.json" in sources
        and any(
            name.startswith("model.safetensors-") and name.endswith(".safetensors")
            for name in sources
        )
    )
    if not has_single_weight and not has_sharded_weight:
        raise FileNotFoundError(
            f"{checkpoint} has neither model.safetensors nor a sharded safetensors index"
        )
    if not has_temporary_names and all((checkpoint / name).is_file() for name in required):
        return checkpoint
    view.mkdir(parents=True, exist_ok=True)
    for target_name, source in sources.items():
        target = view / target_name
        if target.is_file() and target.stat().st_size == source.stat().st_size:
            continue
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return view


def localize_row(
    row: dict[str, Any], project_root: Path
) -> tuple[list[dict[str, Any]], Path]:
    raw_messages = row.get("messages")
    images = row.get("images")
    if not isinstance(raw_messages, list) or len(raw_messages) != 5:
        raise ValueError("Expected system, user, and three assistant messages")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Expected exactly one image")
    image_value = str(images[0])
    if image_value.startswith(REMOTE_PROJECT_PREFIX):
        image_path = project_root / image_value[len(REMOTE_PROJECT_PREFIX) :]
    else:
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = project_root / image_path
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    messages = [dict(message) for message in raw_messages]
    user_text = str(messages[1].get("content", ""))
    if "<image>" not in user_text:
        raise ValueError("User message has no <image> placeholder")
    before, after = user_text.split("<image>", 1)
    user_content: list[dict[str, str]] = []
    if before:
        user_content.append({"type": "text", "text": before})
    user_content.append({"type": "image", "image": str(image_path)})
    if after:
        user_content.append({"type": "text", "text": after})
    messages[1]["content"] = user_content

    for section, message in zip(SECTIONS, messages[2:]):
        if message.get("role") != "assistant":
            raise ValueError(f"{section}: expected assistant role")
        section_body_char_span(section, str(message.get("content", "")))
        try:
            float(message["loss_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{section}: invalid loss_scale") from exc
    return messages, image_path


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _prefix_length(tokenizer: Any, prefix: str, full_text_ids: list[int]) -> int:
    prefix_ids = _token_ids(tokenizer, prefix)
    if full_text_ids[: len(prefix_ids)] != prefix_ids:
        common = 0
        for left, right in zip(prefix_ids, full_text_ids):
            if left != right:
                break
            common += 1
        raise ValueError(
            f"Tokenizer boundary is not prefix-stable: prefix={len(prefix_ids)}, common={common}"
        )
    return len(prefix_ids)


def encode_row(
    processor: Any,
    row: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], list[TokenSpan]]:
    from PIL import Image

    messages, image_path = localize_row(row, project_root)
    template_kwargs = {"enable_thinking": False}
    assistant_contents = [str(message["content"]) for message in messages[2:]]
    merged_content = "".join(assistant_contents)
    render_messages = messages[:2] + [
        {"role": "assistant", "content": merged_content}
    ]
    full_text = processor.apply_chat_template(
        render_messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        encoded = processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )
    input_ids = encoded["input_ids"][0].tolist()
    full_text_ids = _token_ids(processor.tokenizer, full_text)
    expansion = len(input_ids) - len(full_text_ids)
    if expansion < 0:
        raise ValueError("Processor produced fewer tokens than the tokenizer")

    # ms-swift merges consecutive assistant messages before applying the chat
    # template. The separate messages only carry per-section loss scales; they
    # are one generated response and therefore share one assistant/think prefix.
    assistant_prefix = processor.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    empty_think = "<think>\n\n</think>\n\n"
    if not assistant_prefix.endswith(empty_think):
        raise ValueError("Expected one empty non-thinking prefix")
    if not full_text.startswith(assistant_prefix + merged_content):
        raise ValueError("Merged assistant rendering does not prefix the conversation")

    spans: list[TokenSpan] = []
    content_offset = 0
    for offset, section in enumerate(SECTIONS):
        message_index = 2 + offset
        content = assistant_contents[offset]
        leading_whitespace = len(content) - len(content.lstrip())
        stripped_body_start, stripped_body_end = section_body_char_span(
            section, content
        )
        body_start_char = leading_whitespace + stripped_body_start
        body_end_char = leading_whitespace + stripped_body_end
        start_text = assistant_prefix + merged_content[:content_offset]
        body_start_text = (
            assistant_prefix
            + merged_content[: content_offset + body_start_char]
        )
        body_end_text = (
            assistant_prefix
            + merged_content[: content_offset + body_end_char]
        )
        content_offset += len(content)
        end_text = (
            full_text
            if offset == len(SECTIONS) - 1
            else assistant_prefix + merged_content[:content_offset]
        )
        supervised_start_text = (
            assistant_prefix[: -len(empty_think)] if offset == 0 else start_text
        )
        for prefix_name, prefix in {
            "supervised_start": supervised_start_text,
            "start": start_text,
            "body_start": body_start_text,
            "body_end": body_end_text,
            "end": end_text,
        }.items():
            if not full_text.startswith(prefix):
                raise ValueError(f"{section} {prefix_name} rendering does not prefix the full conversation")
        span = TokenSpan(
            section=section,
            weight=float(messages[message_index]["loss_scale"]),
            supervised_start=(
                _prefix_length(processor.tokenizer, supervised_start_text, full_text_ids)
                + expansion
            ),
            start=_prefix_length(processor.tokenizer, start_text, full_text_ids) + expansion,
            body_start=_prefix_length(processor.tokenizer, body_start_text, full_text_ids) + expansion,
            body_end=_prefix_length(processor.tokenizer, body_end_text, full_text_ids) + expansion,
            end=_prefix_length(processor.tokenizer, end_text, full_text_ids) + expansion,
        )
        span.validate(len(input_ids))
        spans.append(span)
    return dict(encoded), spans


def _load_logged_eval_loss(checkpoint: Path, step: int) -> float | None:
    candidates = [checkpoint / "trainer_state.json", checkpoint / "trainer_state.json.codex_sync_part"]
    state_path = next((path for path in candidates if path.is_file()), None)
    if state_path is None:
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for row in reversed(state.get("log_history", [])):
        if int(row.get("step", -1)) == step and "eval_loss" in row:
            return float(row["eval_loss"])
    return None


def load_run_logged_eval_loss(run_dir: Path, step: int) -> float | None:
    direct = _load_logged_eval_loss(run_dir, step)
    if direct is not None:
        return direct
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for checkpoint in checkpoints:
        value = _load_logged_eval_loss(checkpoint, step)
        if value is not None:
            return value
    return None


def evaluate_checkpoint(
    checkpoint: Path,
    checkpoint_step: int,
    view: Path,
    rows: list[tuple[int, dict[str, Any]]],
    project_root: Path,
    device_name: str,
    dtype_name: str,
    cpu_threads: int,
    logged_eval_loss: float | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    model_path = materialize_checkpoint_view(checkpoint, view)
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device_name)
    model.eval()

    accumulators = {
        section: {
            "full": MetricAccumulator(),
            "body": MetricAccumulator(),
            "format": MetricAccumulator(),
        }
        for section in SECTIONS
    }
    weights: dict[str, float] = {}
    sample_seconds: list[float] = []
    sample_indices: list[int] = []
    empty_think_tokens = 0
    all_causal_tokens = 0

    for position, (source_index, row) in enumerate(rows, start=1):
        started = time.monotonic()
        encoded, spans = encode_row(processor, row, project_root)
        sequence_length = int(encoded["input_ids"].shape[1])
        all_causal_tokens += sequence_length - 1
        first_target = min(span.start for span in spans)
        logits_to_keep = sequence_length - (first_target - 1)
        model_inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            if key in {"labels", "loss_scale"}:
                continue
            if hasattr(value, "to"):
                value = value.to(device_name)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(dtype=dtype)
            model_inputs[key] = value
        with torch.inference_mode():
            outputs = model(**model_inputs, logits_to_keep=logits_to_keep, use_cache=False)
        logits = outputs.logits[0]
        logit_start = sequence_length - int(logits.shape[0])
        targets = encoded["input_ids"][0].to(device_name)

        for span in spans:
            weights[span.section] = span.weight
            empty_think_tokens += span.start - span.supervised_start
            ranges = {
                "full": [(span.start, span.end)],
                "body": [(span.body_start, span.body_end)],
                "format": [(span.start, span.body_start), (span.body_end, span.end)],
            }
            for category, token_ranges in ranges.items():
                positions = [
                    token_index
                    for start, end in token_ranges
                    for token_index in range(start, end)
                ]
                logit_indices = torch.tensor(
                    [token_index - 1 - logit_start for token_index in positions],
                    dtype=torch.long,
                    device=device_name,
                )
                if int(logit_indices.min()) < 0 or int(logit_indices.max()) >= logits.shape[0]:
                    raise ValueError(
                        f"{span.section}/{category}: logits_to_keep alignment failed"
                    )
                target_values = targets[
                    torch.tensor(positions, dtype=torch.long, device=device_name)
                ]
                selected_logits = logits.index_select(0, logit_indices).float()
                losses = functional.cross_entropy(
                    selected_logits,
                    target_values,
                    reduction="none",
                )
                predictions = selected_logits.argmax(dim=-1)
                accumulators[span.section][category].update(
                    losses.tolist(),
                    predictions.eq(target_values).tolist(),
                )
        elapsed = time.monotonic() - started
        sample_seconds.append(elapsed)
        sample_indices.append(source_index)
        print(
            f"checkpoint={checkpoint_step} sample={position}/{len(rows)} "
            f"source_index={source_index} seconds={elapsed:.2f}",
            flush=True,
        )
        del outputs, logits, encoded, model_inputs
        gc.collect()

    section_rows: list[dict[str, Any]] = []
    total_section_tokens = 0
    raw_loss_sum = 0.0
    weighted_loss_sum = 0.0
    for section in SECTIONS:
        full = accumulators[section]["full"]
        total_section_tokens += full.token_count
        raw_loss_sum += full.loss_sum
        weighted_loss_sum += weights[section] * full.loss_sum
        section_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "section": section,
                "weight": weights[section],
                **{f"full_{key}": value for key, value in full.to_dict().items()},
                **{
                    f"body_{key}": value
                    for key, value in accumulators[section]["body"].to_dict().items()
                },
                **{
                    f"format_{key}": value
                    for key, value in accumulators[section]["format"].to_dict().items()
                },
            }
        )
    result = {
        "checkpoint_step": checkpoint_step,
        "checkpoint": str(checkpoint),
        "model_view": str(model_path),
        "sample_count": len(rows),
        "sample_indices": sample_indices,
        "device": device_name,
        "dtype": dtype_name,
        "cpu_threads": cpu_threads,
        "seconds": sum(sample_seconds),
        "seconds_per_sample": sum(sample_seconds) / len(sample_seconds),
        "logged_full_validation_loss": (
            logged_eval_loss
            if logged_eval_loss is not None
            else _load_logged_eval_loss(checkpoint, checkpoint_step)
        ),
        "raw_micro_loss": raw_loss_sum / total_section_tokens,
        "weighted_swift_loss_approx": weighted_loss_sum / all_causal_tokens,
        "response_only_weighted_loss": weighted_loss_sum
        / (total_section_tokens + empty_think_tokens),
        "total_section_tokens": total_section_tokens,
        "empty_think_tokens": empty_think_tokens,
        "zero_weight_context_tokens": all_causal_tokens - total_section_tokens,
        "all_causal_tokens": all_causal_tokens,
        "sections": section_rows,
    }
    del model, processor
    if device_name == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    number = float(value)
    return "—" if math.isnan(number) else f"{number:.{digits}f}"


def write_outputs(results: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    section_rows = [row for result in results for row in result["sections"]]
    with (output_dir / "section_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(section_rows[0]))
        writer.writeheader()
        writer.writerows(section_rows)
    summary_rows = [
        {key: value for key, value in result.items() if key not in {"sections", "sample_indices"}}
        for result in results
    ]
    with (output_dir / "checkpoint_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {"metadata": metadata, "results": results}
    (output_dir / "section_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# 三段 CoT Checkpoint EVA",
        "",
        f"- 验证样本：{metadata['sample_count']} / {metadata['validation_size']}",
        f"- 抽样 seed：{metadata['sample_seed']}",
        f"- 设备 / dtype：{metadata['device']} / {metadata['dtype']}",
        "- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。",
        "- `weighted approx` 按 ms-swift 的实际公式 `Σ(weight × token CE) / 全部 causal token 数` 计算；system/user/image/空 think 的权重为 0，但仍进入分母。",
        "",
        "## Checkpoint 汇总",
        "",
        "| step | logged eval loss | weighted approx | raw micro loss | seconds/sample |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {checkpoint_step} | {logged} | {weighted} | {raw} | {seconds} |".format(
                checkpoint_step=result["checkpoint_step"],
                logged=_fmt(result["logged_full_validation_loss"]),
                weighted=_fmt(result["weighted_swift_loss_approx"]),
                raw=_fmt(result["raw_micro_loss"]),
                seconds=_fmt(result["seconds_per_sample"], 2),
            )
        )
    lines.extend(
        [
            "",
            "## 分段详细指标",
            "",
            "| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in section_rows:
        values = dict(row)
        values.update(
            {
                "full_loss": _fmt(row["full_loss"]),
                "body_loss": _fmt(row["body_loss"]),
                "format_loss": _fmt(row["format_loss"]),
                "full_perplexity": _fmt(row["full_perplexity"], 4),
                "full_token_accuracy": _fmt(row["full_token_accuracy"], 4),
                "body_token_accuracy": _fmt(row["body_token_accuracy"], 4),
                "full_exact_match": _fmt(row["full_exact_match"], 4),
            }
        )
        lines.append(
            "| {checkpoint_step} | {section} | {weight:.1f} | {full_token_count} | "
            "{full_loss} | {body_loss} | {format_loss} | {full_perplexity} | "
            "{full_token_accuracy} | {body_token_accuracy} | {full_exact_match} |".format(
                **values,
            )
        )
    (output_dir / "EVA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate state/plan/action teacher-forced losses by checkpoint"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        help="Original pretrained model directory; required when checkpoint step 0 is requested",
    )
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    val_file = args.val_file.resolve()
    output_dir = args.output_dir.resolve()
    all_rows = load_jsonl(val_file)
    selected = select_rows(all_rows, args.max_samples, args.sample_seed)
    results: list[dict[str, Any]] = []
    for step in args.checkpoint_steps:
        if step == 0:
            if args.base_model_dir is None:
                raise ValueError("--base-model-dir is required for checkpoint step 0")
            checkpoint = args.base_model_dir.resolve()
            view = run_dir / ".section_loss_views" / "base-model-step-0"
        else:
            checkpoint = run_dir / f"checkpoint-{step}"
            view = run_dir / ".section_loss_views" / f"checkpoint-{step}"
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            checkpoint_step=step,
            view=view,
            rows=selected,
            project_root=project_root,
            device_name=args.device,
            dtype_name=args.dtype,
            cpu_threads=args.cpu_threads,
            logged_eval_loss=load_run_logged_eval_loss(run_dir, step),
        )
        results.append(result)
        metadata = {
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "base_model_dir": str(args.base_model_dir.resolve()) if args.base_model_dir else None,
            "val_file": str(val_file),
            "validation_size": len(all_rows),
            "sample_count": len(selected),
            "sample_seed": args.sample_seed,
            "checkpoint_steps": args.checkpoint_steps,
            "device": result["device"],
            "dtype": args.dtype,
            "cpu_threads": args.cpu_threads,
        }
        write_outputs(results, output_dir, metadata)
    print(output_dir / "EVA.md", flush=True)


if __name__ == "__main__":
    main()
