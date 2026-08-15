from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import PLAN_BLOCK_PATTERN, normalize_action, parse_action, parse_plan, read_json, read_jsonl, write_json


SPACE_ACTION_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)[。.]?$")
LEADING_MARKER_PATTERN = re.compile(r"^(?:[-*]|\d+[.)])\s*")


def lenient_actions(text: str, allowed_actions: list[str]) -> list[str]:
    block_match = PLAN_BLOCK_PATTERN.search(text)
    block = block_match.group(1) if block_match else text
    canonical_names = {name.casefold(): name for name in allowed_actions}
    actions: list[str] = []
    for raw_line in block.splitlines():
        line = LEADING_MARKER_PATTERN.sub("", raw_line.strip()).strip("` ")
        if not line:
            continue
        parsed = parse_action(line)
        if parsed is None:
            space_match = SPACE_ACTION_PATTERN.match(line)
            if space_match is None:
                continue
            raw_name, raw_arg = space_match.groups()
            name = canonical_names.get(raw_name.casefold())
            if name is None:
                continue
            actions.append(f"{name}({raw_arg})")
            continue
        raw_name, args = parsed
        name = canonical_names.get(raw_name.casefold())
        if name is None:
            continue
        actions.append(f"{name}({','.join(args)})")
    return actions


def score_actions(predicted: list[str], gold: list[str]) -> tuple[float, float]:
    denominator = max(len(predicted), len(gold), 1)
    positional = sum(a == b for a, b in zip(predicted, gold))
    return float(predicted == gold), positional / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplemental Exp 0 output-format analysis")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    base_dir = config_path.parent
    samples = {
        row["sample_id"]: row
        for row in read_jsonl((base_dir / config["dataset_path"]).resolve())
    }
    output_dir = (base_dir / config["output_dir"]).resolve()
    predictions = read_jsonl(output_dir / "predictions.jsonl")
    allowed_actions = list(config["allowed_actions"])

    by_condition: dict[str, list[dict[str, float]]] = defaultdict(list)
    invalid_names: dict[str, Counter[str]] = defaultdict(Counter)
    unparseable_lines: Counter[str] = Counter()
    keys: set[tuple[str, str]] = set()

    for prediction in predictions:
        sample_id = str(prediction["sample_id"])
        condition = str(prediction["condition"])
        key = (sample_id, condition)
        if key in keys:
            raise RuntimeError(f"duplicate prediction: {key}")
        keys.add(key)

        raw_output = str(prediction.get("raw_output", ""))
        strict = parse_plan(raw_output)
        gold = [normalize_action(action) for action in samples[sample_id]["gold"]["plan_actions"]]
        relaxed = lenient_actions(raw_output, allowed_actions)
        strict_em, strict_step = score_actions(strict.actions, gold)
        relaxed_em, relaxed_step = score_actions(relaxed, gold)
        by_condition[condition].append(
            {
                "strict_em": strict_em,
                "strict_step": strict_step,
                "strict_structure": float(strict.structure_valid),
                "strict_nonempty": float(bool(strict.actions)),
                "lenient_em": relaxed_em,
                "lenient_step": relaxed_step,
                "lenient_nonempty": float(bool(relaxed)),
                "elapsed_seconds": float(prediction.get("elapsed_seconds", 0.0)),
            }
        )

        invalid_names[condition].update(
            name for name in strict.action_names if name not in allowed_actions
        )
        block_match = PLAN_BLOCK_PATTERN.search(raw_output)
        block = block_match.group(1) if block_match else raw_output
        for line in (line.strip() for line in block.splitlines() if line.strip()):
            if parse_action(line) is None:
                unparseable_lines[line] += 1

    summary: dict[str, Any] = {}
    for condition, rows in sorted(by_condition.items()):
        summary[condition] = {
            "count": len(rows),
            **{
                key: mean(row[key] for row in rows)
                for key in (
                    "strict_em",
                    "strict_step",
                    "strict_structure",
                    "strict_nonempty",
                    "lenient_em",
                    "lenient_step",
                    "lenient_nonempty",
                    "elapsed_seconds",
                )
            },
            "top_invalid_action_names": invalid_names[condition].most_common(10),
        }

    result = {
        "prediction_count": len(predictions),
        "unique_keys": len(keys),
        "conditions": summary,
        "top_unparseable_plan_lines": unparseable_lines.most_common(20),
        "note": "Lenient metrics are supplemental diagnostics only; official Exp 0 metrics remain unchanged.",
    }
    write_json(output_dir / "format_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
