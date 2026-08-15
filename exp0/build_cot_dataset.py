from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .schema import parse_action, read_jsonl


def _nonempty_text(value: Any, field: str, sample_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{sample_id}: '{field}' must be a non-empty string")
    return value.strip()


def _nonempty_string_list(value: Any, field: str, sample_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{sample_id}: '{field}' must be a non-empty list")
    result = [_nonempty_text(item, f"{field}[]", sample_id) for item in value]
    return result


def _relative_image(source_path: Path, output_path: Path, image: str) -> str:
    source_image = Path(image).expanduser()
    if not source_image.is_absolute():
        source_image = source_path.parent / source_image
    source_image = source_image.resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Image does not exist: {source_image}")
    return Path(os.path.relpath(source_image, output_path.parent)).as_posix()


def build_cot_row(row: dict[str, Any], source_path: Path, output_path: Path) -> dict[str, Any]:
    sample_id = _nonempty_text(row.get("sample_id"), "sample_id", "<unknown>")
    instruction = _nonempty_text(
        row.get("instruction", row.get("prompt")), "instruction/prompt", sample_id
    )
    image = _relative_image(
        source_path,
        output_path,
        _nonempty_text(row.get("image"), "image", sample_id),
    )
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else row
    actions = _nonempty_string_list(gold.get("plan_actions"), "gold.plan_actions", sample_id)
    for action in actions:
        if parse_action(action) is None:
            raise ValueError(f"{sample_id}: invalid action syntax: {action}")
    plan_nl = _nonempty_text(gold.get("plan_nl"), "gold.plan_nl", sample_id)
    spatial_facts = _nonempty_string_list(
        row.get("spatial_facts"), "spatial_facts", sample_id
    )
    planner_subgoals = _nonempty_string_list(row.get("subgoals"), "subgoals", sample_id)

    metadata = row.get("meta", row.get("_meta", {}))
    if not isinstance(metadata, dict):
        raise ValueError(f"{sample_id}: 'meta/_meta' must be an object")
    if metadata.get("sim_verified") is not True:
        raise ValueError(f"{sample_id}: simulator verification is required")

    return {
        "sample_id": sample_id,
        "image": image,
        "instruction": instruction,
        "gold": {
            "spatial_state": "\n".join(spatial_facts),
            "subgoals": "\n".join(
                f"{index}. {subgoal}"
                for index, subgoal in enumerate(planner_subgoals, start=1)
            ),
            "plan_actions": actions,
            "plan_nl": plan_nl,
        },
        "_meta": dict(metadata),
    }


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_cot_dataset(source_path: Path, output_path: Path, overwrite: bool = False) -> int:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset does not exist: {source_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output_path}")
    source_rows = read_jsonl(source_path)
    output_rows = [build_cot_row(row, source_path, output_path) for row in source_rows]
    if not output_rows:
        raise ValueError(f"Source dataset is empty: {source_path}")
    _write_jsonl_atomic(output_path, output_rows)
    return len(output_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build simulator-supervised state/subgoal/action SFT records"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("data") / "samples.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data_cot") / "train.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = build_cot_dataset(args.input, args.output, overwrite=bool(args.overwrite))
    print(f"Built {count} CoT samples: {args.output.resolve()}")


if __name__ == "__main__":
    main()
