from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是一个室内家务动作规划助手。
请观察输入图片，并根据目标指令，用自然语言给出完整、可执行且顺序明确的操作规划。
规划中的每一步都必须明确指出要操作的物品或位置，并根据图片中容器当前的开启或关闭状态决定是否需要打开、关闭。
只输出简洁的中文编号步骤。"""


def generate(
    checkpoint: Path,
    image_path: Path,
    instruction: str,
    max_new_tokens: int,
) -> tuple[str, float, int, str]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to("cpu")
    model.eval()

    user_text = f"目标指令：{instruction}\n请直接给出完成任务的中文自然语言步骤。"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    encoded = processor(text=[prompt], images=[image], return_tensors="pt")
    input_length = int(encoded["input_ids"].shape[1])
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.monotonic() - started
    new_ids = generated[0, input_length:].tolist()
    prediction = processor.tokenizer.decode(
        new_ids, skip_special_tokens=True
    ).strip()

    del generated, encoded, image, model, processor
    gc.collect()
    return prediction, elapsed, len(new_ids), user_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare before/after natural-language plans on CPU"
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, args.cpu_threads)))
    except RuntimeError:
        pass

    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    runs = (
        ("before", args.base_model.resolve()),
        ("after", args.trained_model.resolve()),
    )
    results: list[dict[str, Any]] = []
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    for label, checkpoint in runs:
        print(f"loading={label} checkpoint={checkpoint}", flush=True)
        prediction, seconds, tokens, user_text = generate(
            checkpoint,
            image_path,
            args.instruction,
            args.max_new_tokens,
        )
        results.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "sample_id": args.sample_id,
                "image": str(image_path),
                "system_prompt": SYSTEM_PROMPT,
                "user_text": user_text,
                "prediction": prediction,
                "generated_tokens": tokens,
                "seconds": seconds,
                "device": "cpu",
                "dtype": "float32",
                "decoding": "greedy",
            }
        )
        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"finished={label} seconds={seconds:.2f} tokens={tokens}", flush=True)

    print(output, flush=True)


if __name__ == "__main__":
    main()
