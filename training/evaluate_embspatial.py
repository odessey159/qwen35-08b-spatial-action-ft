from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from training.evaluate_section_losses import materialize_checkpoint_view


CHOICE_LABELS = ("A", "B", "C", "D")
PROMPT_VERSIONS = {
    "direct": "embspatial-generation-direct-v1",
    "cot": "embspatial-generation-cot-final-answer-v2",
}
DIRECT_SYSTEM_PROMPT = (
    "You are evaluating spatial understanding. Answer the multiple-choice "
    "question using only one letter: A, B, C, or D."
)
COT_SYSTEM_PROMPT = (
    "You are evaluating spatial understanding. Analyze the image and the answer "
    "choices step by step. Keep the reasoning concise. End the response with exactly "
    "one line in the form 'Final answer: X', where X is A, B, C, or D. Do not use XML tags."
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty JSON array: {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at dataset index {index}")
        options = row.get("answer_options")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError(f"Dataset index {index} does not have four answer options")
        if not isinstance(row.get("image"), str) or not row["image"].strip():
            raise ValueError(f"Dataset index {index} has no base64 image")
        rows.append(row)
    return rows


def answer_label(answer: Any) -> str:
    if isinstance(answer, bool):
        raise ValueError(f"Invalid boolean answer: {answer!r}")
    if isinstance(answer, int) and 0 <= answer < len(CHOICE_LABELS):
        return CHOICE_LABELS[answer]
    text = str(answer).strip().upper()
    if text in CHOICE_LABELS:
        return text
    if text.isdigit() and 0 <= int(text) < len(CHOICE_LABELS):
        return CHOICE_LABELS[int(text)]
    raise ValueError(f"Unsupported answer value: {answer!r}")


def format_question(
    question: Any, options: Iterable[Any], prompt_style: str = "direct"
) -> str:
    option_values = [str(value).strip() for value in options]
    if len(option_values) != len(CHOICE_LABELS) or any(not value for value in option_values):
        raise ValueError("Exactly four non-empty answer options are required")
    lines = [str(question).strip(), ""]
    lines.extend(f"{label}. {value}" for label, value in zip(CHOICE_LABELS, option_values))
    if prompt_style == "direct":
        lines.extend(["", "Answer with only A, B, C, or D."])
    elif prompt_style == "cot":
        lines.extend(
            [
                "",
                "First reason about the relevant objects and their spatial relationship.",
                "Then respond in exactly this format:",
                "Reasoning: Your concise step-by-step reasoning",
                "Final answer: X",
                "Replace X with exactly one option letter: A, B, C, or D.",
                "Do not use XML tags. Your response must end with the Final answer line.",
            ]
        )
    else:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    return "\n".join(lines)


def parse_choice(prediction: Any, options: Iterable[Any] | None = None) -> str | None:
    text = str(prediction).strip()
    if not text:
        return None
    tagged = re.findall(r"(?is)<answer>\s*([A-D])(?:\s*</answer>)?", text)
    if tagged:
        return tagged[-1].upper()
    final = re.findall(
        r"(?im)^\s*final\s*(?:answer|choice)?\s*[:=]\s*[\(\[]?([A-D])\b",
        text,
    )
    if final:
        return final[-1].upper()
    leading = re.match(r"^[\s\(\[]*([A-Da-d])(?:[\s\)\]\.,:;]|$)", text)
    if leading:
        return leading.group(1).upper()
    explicit = re.findall(
        r"(?i)\b(?:answer|option|choice)\s*(?:is|=|:)?\s*[\(\[]?([A-D])\b",
        text,
    )
    if explicit:
        return explicit[-1].upper()
    if options is not None:
        normalized = re.sub(r"\s+", " ", text).strip(" .\t\r\n").casefold()
        matches = [
            label
            for label, option in zip(CHOICE_LABELS, options)
            if normalized == re.sub(r"\s+", " ", str(option)).strip(" .").casefold()
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    if not values:
        raise ValueError("No prediction records to summarize")
    slices: dict[str, dict[str, list[int]]] = {
        "relation": defaultdict(list),
        "data_source": defaultdict(list),
    }
    correct = 0
    valid = 0
    for row in values:
        is_valid = row.get("predicted_label") in CHOICE_LABELS
        is_correct = bool(row.get("correct"))
        valid += int(is_valid)
        correct += int(is_correct)
        for field in slices:
            slices[field][str(row.get(field, "unknown"))].append(int(is_correct))
    return {
        "samples": len(values),
        "correct": correct,
        "accuracy": correct / len(values),
        "valid_predictions": valid,
        "valid_prediction_rate": valid / len(values),
        "by_relation": {
            key: {"samples": len(items), "accuracy": sum(items) / len(items)}
            for key, items in sorted(slices["relation"].items())
        },
        "by_data_source": {
            key: {"samples": len(items), "accuracy": sum(items) / len(items)}
            for key, items in sorted(slices["data_source"].items())
        },
    }


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            reparsed = parse_choice(value.get("prediction", ""))
            if reparsed is not None:
                value["predicted_label"] = reparsed
                if value.get("gold_label") in CHOICE_LABELS:
                    value["correct"] = reparsed == value["gold_label"]
            rows.append(value)
    return rows


def compare_predictions(before: Path, after: Path) -> dict[str, Any]:
    before_rows = load_prediction_records(before)
    after_rows = load_prediction_records(after)
    before_by_id = {str(row["question_id"]): row for row in before_rows}
    after_by_id = {str(row["question_id"]): row for row in after_rows}
    if set(before_by_id) != set(after_by_id):
        missing_before = sorted(set(after_by_id) - set(before_by_id))[:10]
        missing_after = sorted(set(before_by_id) - set(after_by_id))[:10]
        raise ValueError(
            f"Prediction IDs differ: missing_before={missing_before}, missing_after={missing_after}"
        )
    ordered_ids = sorted(before_by_id)
    before_summary = summarize(before_by_id[key] for key in ordered_ids)
    after_summary = summarize(after_by_id[key] for key in ordered_ids)
    improved = regressed = unchanged = 0
    for key in ordered_ids:
        left = int(bool(before_by_id[key].get("correct")))
        right = int(bool(after_by_id[key].get("correct")))
        improved += int(left == 0 and right == 1)
        regressed += int(left == 1 and right == 0)
        unchanged += int(left == right)
    return {
        "samples": len(ordered_ids),
        "before": before_summary,
        "after": after_summary,
        "accuracy_delta": after_summary["accuracy"] - before_summary["accuracy"],
        "paired": {"improved": improved, "regressed": regressed, "unchanged": unchanged},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 on EmbSpatial-Bench")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--compare-before", type=Path)
    parser.add_argument("--compare-after", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt-style", choices=["direct", "cot"], default="direct")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_comparison(args: argparse.Namespace) -> None:
    if not args.compare_before or not args.compare_after or not args.summary:
        raise ValueError("Comparison mode requires --compare-before, --compare-after, and --summary")
    result = compare_predictions(args.compare_before.resolve(), args.compare_after.resolve())
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def run_evaluation(args: argparse.Namespace) -> None:
    if not args.checkpoint or not args.dataset or not args.output or not args.summary:
        raise ValueError("Evaluation mode requires --checkpoint, --dataset, --output, and --summary")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("--batch-size and --max-new-tokens must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(args.cpu_threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    checkpoint = args.checkpoint.resolve()
    view = checkpoint.parent / ".embspatial_views" / checkpoint.name
    model_path = materialize_checkpoint_view(checkpoint, view)
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    rows = load_rows(args.dataset.resolve())
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_prediction_records(output) if args.resume else []
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"Output exists (use --resume or --overwrite): {output}")
    if args.overwrite:
        existing = []
    completed = {str(row["question_id"]) for row in existing}
    pending = [row for row in rows if str(row.get("question_id")) not in completed]
    mode = "a" if args.resume and output.exists() else "w"
    prompt_version = PROMPT_VERSIONS[args.prompt_style]
    system_prompt = (
        DIRECT_SYSTEM_PROMPT if args.prompt_style == "direct" else COT_SYSTEM_PROMPT
    )

    with output.open(mode, encoding="utf-8", newline="\n") as handle:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            prompts: list[str] = []
            images: list[Any] = []
            started = time.monotonic()
            for row in batch:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "embspatial-inline-image"},
                            {
                                "type": "text",
                                "text": format_question(
                                    row.get("question"),
                                    row["answer_options"],
                                    prompt_style=args.prompt_style,
                                ),
                            },
                        ],
                    },
                ]
                prompts.append(
                    processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                image_bytes = base64.b64decode(row["image"], validate=False)
                with Image.open(io.BytesIO(image_bytes)) as source:
                    images.append(source.convert("RGB"))
            encoded = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {
                key: value.to(device, dtype=dtype) if value.is_floating_point() else value.to(device)
                for key, value in encoded.items()
            }
            input_length = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            elapsed = time.monotonic() - started
            for batch_index, row in enumerate(batch):
                new_ids = generated[batch_index, input_length:].tolist()
                prediction = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                predicted_label = parse_choice(prediction, row["answer_options"])
                gold_label = answer_label(row.get("answer"))
                record = {
                    "question_id": str(row.get("question_id")),
                    "relation": str(row.get("relation", "unknown")),
                    "data_source": str(row.get("data_source", "unknown")),
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == gold_label,
                    "prediction": prediction,
                    "generated_tokens": len(new_ids),
                    "seconds": elapsed / len(batch),
                    "prompt_version": prompt_version,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done = len(completed) + batch_start + len(batch)
            print(
                f"samples={done}/{len(rows)} batch_seconds={elapsed:.2f} "
                f"seconds_per_sample={elapsed / len(batch):.2f}",
                flush=True,
            )

    records = load_prediction_records(output)
    summary = summarize(records)
    summary.update(
        {
            "checkpoint": str(checkpoint),
            "dataset": str(args.dataset.resolve()),
            "prompt_version": prompt_version,
            "device": device,
            "dtype": args.dtype,
        }
    )
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.compare_before or args.compare_after:
        run_comparison(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
