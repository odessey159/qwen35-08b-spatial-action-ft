from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.evaluate_section_losses import (
    load_jsonl,
    localize_row,
    materialize_checkpoint_view,
)


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("meta", row.get("metadata", row.get("_meta", {})))
    return value if isinstance(value, dict) else {}


def select_counterfactual_rows(
    prepared_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    max_pairs: int,
) -> list[tuple[int, str, str, dict[str, Any]]]:
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids", [])]
    if len(validation_ids) != len(prepared_rows):
        raise ValueError("validation_sample_ids do not align with the prepared val rows")
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    groups: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for index, (sample_id, prepared) in enumerate(zip(validation_ids, prepared_rows)):
        raw = raw_by_id.get(sample_id)
        if raw is None:
            raise ValueError(f"Raw validation sample is missing: {sample_id}")
        group = _metadata(raw).get("counterfactual_group")
        if group not in (None, ""):
            groups[str(group)].append((index, sample_id, prepared))
    complete = [(group, values) for group, values in sorted(groups.items()) if len(values) == 2]
    if max_pairs > 0:
        complete = complete[:max_pairs]
    return [
        (index, sample_id, group, prepared)
        for group, values in complete
        for index, sample_id, prepared in values
    ]


def select_all_rows(
    prepared_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    max_samples: int,
) -> list[tuple[int, str, str, dict[str, Any]]]:
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids", [])]
    if len(validation_ids) != len(prepared_rows):
        raise ValueError("validation_sample_ids do not align with the prepared val rows")
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    selected: list[tuple[int, str, str, dict[str, Any]]] = []
    for index, (sample_id, prepared) in enumerate(zip(validation_ids, prepared_rows)):
        raw = raw_by_id.get(sample_id)
        if raw is None:
            raise ValueError(f"Raw validation sample is missing: {sample_id}")
        group = _metadata(raw).get("counterfactual_group")
        selected.append((index, sample_id, "" if group in (None, "") else str(group), prepared))
    return selected[:max_samples] if max_samples > 0 else selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate full-CoT predictions for validation rows"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--selection", choices=["counterfactual-pairs", "all"], default="counterfactual-pairs"
    )
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output}")
    if args.selection == "counterfactual-pairs" and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    except RuntimeError:
        pass
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
    project_root = args.project_root.resolve()
    checkpoint = args.checkpoint.resolve()
    view = checkpoint.parent / ".section_loss_views" / checkpoint.name
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

    prepared_rows = load_jsonl(args.val_file.resolve())
    raw_rows = load_jsonl(args.raw_data.resolve())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.selection == "counterfactual-pairs":
        selected = select_counterfactual_rows(
            prepared_rows, raw_rows, manifest, args.max_pairs
        )
    else:
        selected = select_all_rows(
            prepared_rows, raw_rows, manifest, args.max_samples
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for batch_start in range(0, len(selected), args.batch_size):
            batch = selected[batch_start : batch_start + args.batch_size]
            started = time.monotonic()
            prompts: list[str] = []
            images: list[Any] = []
            for _, _, _, row in batch:
                messages, image_path = localize_row(row, project_root)
                prompts.append(
                    processor.apply_chat_template(
                        messages[:2],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
                with Image.open(image_path) as source_image:
                    images.append(source_image.convert("RGB"))
            encoded = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device)
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
            for batch_index, (source_index, sample_id, group, _) in enumerate(batch):
                new_ids = generated[batch_index, input_length:].tolist()
                prediction = processor.tokenizer.decode(
                    new_ids, skip_special_tokens=True
                ).strip()
                result = {
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "counterfactual_group": group,
                    "prediction": prediction,
                    "generated_tokens": len(new_ids),
                    "device": device,
                    "dtype": args.dtype,
                    "batch_size": len(batch),
                    "seconds": elapsed / len(batch),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"samples={batch_start + 1}-{batch_start + len(batch)}/{len(selected)} "
                f"batch_seconds={elapsed:.2f} seconds_per_sample={elapsed / len(batch):.2f}",
                flush=True,
            )
    print(output, flush=True)


if __name__ == "__main__":
    main()
