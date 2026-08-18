from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是动作序列逐行翻译器，不是动作规划器。
用户只会提供 <action>，你必须按照原顺序把每一行翻译成中文，不得纠正、补全、重排或删除。
不得猜测 action 中没有写出的参数。GotoLocation 没有括号参数时写“前往目标未指定的位置”；PickupObject 没有括号参数时写“拿起物体未指定”；PutObject 没有括号参数时写“放置的物体和目标位置均未指定”。
输出必须是 <action_description> 标签，标签内使用中文编号列表，每个输入动作对应且只能对应一项。"""


def load_prediction(path: Path, sample_id: str) -> str:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("sample_id")) == sample_id:
                return str(row["prediction"])
    raise ValueError(f"{sample_id} was not found in {path}")


def extract_action(prediction: str) -> str:
    match = re.search(r"<action>\s*(.*?)\s*</action>", prediction, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("Prediction has no complete <action> block")
    return "<action>\n" + match.group(1).strip() + "\n</action>"


def generate_description(
    checkpoint: Path,
    prediction: str,
    instruction: str,
    max_new_tokens: int,
) -> tuple[str, float, int]:
    import torch
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

    action = extract_action(prediction)
    user_prompt = f"""请逐行翻译下面的动作序列：

{action}

只翻译这些动作。不要复述任务，不要生成 state 或 plan，不要推测缺失参数。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = processor(text=[prompt], return_tensors="pt")
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
    description = processor.tokenizer.decode(
        new_ids, skip_special_tokens=True
    ).strip()

    del generated, encoded, model, processor
    gc.collect()
    return description, elapsed, len(new_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate model-authored Chinese explanations for before/after actions"
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--trained-model", type=Path, required=True)
    parser.add_argument("--before-predictions", type=Path, required=True)
    parser.add_argument("--after-predictions", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
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

    before = load_prediction(args.before_predictions.resolve(), args.sample_id)
    after = load_prediction(args.after_predictions.resolve(), args.sample_id)
    inputs = (
        ("before", args.base_model.resolve(), before),
        ("after", args.trained_model.resolve(), after),
    )
    results: list[dict[str, Any]] = []
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    for label, checkpoint, prediction in inputs:
        print(f"loading={label} checkpoint={checkpoint}", flush=True)
        description, seconds, generated_tokens = generate_description(
            checkpoint,
            prediction,
            args.instruction,
            args.max_new_tokens,
        )
        results.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "sample_id": args.sample_id,
                "instruction": args.instruction,
                "original_prediction": prediction,
                "action_given_for_description": extract_action(prediction),
                "action_description": description,
                "generated_tokens": generated_tokens,
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
        print(
            f"finished={label} seconds={seconds:.2f} tokens={generated_tokens}",
            flush=True,
        )

    print(output, flush=True)


if __name__ == "__main__":
    main()
