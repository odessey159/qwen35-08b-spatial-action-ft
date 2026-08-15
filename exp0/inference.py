from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .prompts import build_messages, build_prompt_cases
from .schema import append_jsonl, parse_plan, read_jsonl


def _load_model(config: dict[str, Any]):
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install requirements-exp0.txt first."
        ) from exc

    model_config = config["model"]
    dtype_name = model_config.get("dtype", "bfloat16")
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    load_kwargs: dict[str, Any] = {
        "dtype": getattr(torch, dtype_name),
        "device_map": model_config.get("device_map", "auto"),
    }
    attn_impl = model_config.get("attn_implementation")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    processor = AutoProcessor.from_pretrained(config["model_id"])
    model = AutoModelForMultimodalLM.from_pretrained(
        config["model_id"],
        **load_kwargs,
    )
    model.eval()
    torch.manual_seed(int(model_config.get("seed", 42)))
    return torch, processor, model


def _generate_one(
    torch_module: Any,
    processor: Any,
    model: Any,
    messages: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 512)),
        "do_sample": bool(generation_config.get("do_sample", False)),
    }
    if generate_kwargs["do_sample"]:
        generate_kwargs["temperature"] = float(generation_config.get("temperature", 1.0))

    with torch_module.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference(
    config: dict[str, Any],
    dataset_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    samples = read_jsonl(dataset_path)
    allowed_actions = list(config["allowed_actions"])

    completed: set[tuple[str, str]] = set()
    if output_path.exists() and not overwrite:
        for row in read_jsonl(output_path):
            completed.add((str(row["sample_id"]), str(row["condition"])))
    elif output_path.exists() and overwrite:
        output_path.unlink()

    torch_module, processor, model = _load_model(config)
    for sample in samples:
        cases = build_prompt_cases(sample, dataset_path.parent, allowed_actions)
        for case in cases:
            key = (sample["sample_id"], case.condition)
            if key in completed:
                continue

            messages = build_messages(case.prompt, case.image_path)
            started = time.perf_counter()
            raw_output = _generate_one(
                torch_module,
                processor,
                model,
                messages,
                config["model"],
            )
            elapsed = time.perf_counter() - started
            parsed = parse_plan(raw_output)
            append_jsonl(
                output_path,
                {
                    "sample_id": sample["sample_id"],
                    "condition": case.condition,
                    "scene_graph_format": case.scene_graph_format,
                    "image": str(case.image_path) if case.image_path else None,
                    "prompt": case.prompt,
                    "raw_output": raw_output,
                    "pred_actions": parsed.actions,
                    "elapsed_seconds": round(elapsed, 4),
                },
            )

