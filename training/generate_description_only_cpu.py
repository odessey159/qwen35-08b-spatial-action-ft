from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是室内家务动作规划助手。
请根据输入图像和目标指令，用中文自然语言按执行顺序说明完成任务所需的动作。
回答必须且只能包含 <action_description>...</action_description>。
不要输出 <state>、<plan> 或 <action>，不要输出动作代码，也不要添加其他段落。
描述必须明确说明前往哪里、操作什么物体，以及需要打开或关闭的容器。"""


def generate(
    checkpoint: Path,
    image_path: Path,
    instruction: str,
    max_new_tokens: int,
) -> tuple[str, float, int]:
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

    user_content: list[dict[str, str]] = [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": f"目标指令：{instruction}"},
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
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
    output = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    del generated, encoded, image, model, processor
    gc.collect()
    return output, elapsed, len(new_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate description-only plans from base and trained checkpoints on CPU"
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
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    checkpoints = (
        ("before", args.base_model.resolve()),
        ("after", args.trained_model.resolve()),
    )
    for label, checkpoint in checkpoints:
        print(f"loading={label} checkpoint={checkpoint}", flush=True)
        prediction, seconds, tokens = generate(
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
                "instruction": args.instruction,
                "image": str(image_path),
                "system_prompt": SYSTEM_PROMPT,
                "prediction": prediction,
                "generated_tokens": tokens,
                "seconds": seconds,
                "device": "cpu",
                "dtype": "float32",
                "decoding": "greedy",
            }
        )
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"finished={label} seconds={seconds:.2f} tokens={tokens}", flush=True)

    print(output_path, flush=True)


if __name__ == "__main__":
    main()
