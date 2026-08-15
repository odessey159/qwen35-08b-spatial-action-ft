from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
PLAN_BLOCK_PATTERN = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedPlan:
    actions: list[str]
    action_names: list[str]
    arguments: list[list[str]]
    has_plan_tags: bool
    all_lines_parseable: bool

    @property
    def structure_valid(self) -> bool:
        return self.has_plan_tags and self.all_lines_parseable and bool(self.actions)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_action(action: str) -> str:
    match = ACTION_PATTERN.match(action.strip())
    if not match:
        return " ".join(action.strip().split())
    name, raw_args = match.groups()
    args = [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    return f"{name}({','.join(args)})"


def parse_action(action: str) -> tuple[str, list[str]] | None:
    match = ACTION_PATTERN.match(action.strip())
    if not match:
        return None
    name, raw_args = match.groups()
    args = [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    return name, args


def parse_plan(text: str) -> ParsedPlan:
    block_match = PLAN_BLOCK_PATTERN.search(text)
    has_tags = block_match is not None
    block = block_match.group(1) if block_match else text
    candidate_lines = [line.strip() for line in block.splitlines() if line.strip()]

    actions: list[str] = []
    names: list[str] = []
    arguments: list[list[str]] = []
    all_parseable = bool(candidate_lines)
    for line in candidate_lines:
        parsed = parse_action(line)
        if parsed is None:
            all_parseable = False
            continue
        name, args = parsed
        actions.append(normalize_action(line))
        names.append(name)
        arguments.append(args)

    return ParsedPlan(
        actions=actions,
        action_names=names,
        arguments=arguments,
        has_plan_tags=has_tags,
        all_lines_parseable=all_parseable,
    )


def _require_type(
    sample_id: str,
    row: dict[str, Any],
    key: str,
    expected_type: type,
    errors: list[str],
) -> Any:
    value = row.get(key)
    if not isinstance(value, expected_type):
        errors.append(f"{sample_id}: '{key}' must be {expected_type.__name__}")
    return value


def validate_samples(
    samples: list[dict[str, Any]],
    dataset_dir: Path,
    allowed_actions: set[str],
    allowed_objects: set[str],
    min_samples: int,
    max_samples: int,
) -> list[str]:
    errors: list[str] = []
    if not min_samples <= len(samples) <= max_samples:
        errors.append(
            f"Dataset must contain {min_samples}-{max_samples} rows; found {len(samples)}"
        )

    seen_ids: set[str] = set()
    for index, row in enumerate(samples, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            sample_id = f"row_{index}"
            errors.append(f"{sample_id}: missing non-empty 'sample_id'")
        elif sample_id in seen_ids:
            errors.append(f"{sample_id}: duplicate sample_id")
        seen_ids.add(sample_id)

        for key in ("image", "wrong_image", "instruction"):
            value = _require_type(sample_id, row, key, str, errors)
            if key in {"image", "wrong_image"} and isinstance(value, str):
                image_path = (dataset_dir / value).resolve()
                if not image_path.is_file():
                    errors.append(f"{sample_id}: image does not exist: {image_path}")

        scene_graph = _require_type(sample_id, row, "scene_graph", dict, errors)
        if isinstance(scene_graph, dict):
            if not isinstance(scene_graph.get("objects"), list):
                errors.append(f"{sample_id}: scene_graph.objects must be a list")
            if not isinstance(scene_graph.get("relations"), list):
                errors.append(f"{sample_id}: scene_graph.relations must be a list")

        spatial_facts = _require_type(sample_id, row, "spatial_facts", list, errors)
        subgoals = _require_type(sample_id, row, "subgoals", list, errors)
        if isinstance(spatial_facts, list) and not all(isinstance(x, str) for x in spatial_facts):
            errors.append(f"{sample_id}: every spatial fact must be a string")
        if isinstance(subgoals, list) and not all(isinstance(x, str) for x in subgoals):
            errors.append(f"{sample_id}: every subgoal must be a string")

        gold = _require_type(sample_id, row, "gold", dict, errors)
        gold_actions: list[str] = []
        if isinstance(gold, dict):
            raw_actions = gold.get("plan_actions")
            if not isinstance(raw_actions, list) or not raw_actions:
                errors.append(f"{sample_id}: gold.plan_actions must be a non-empty list")
            elif not all(isinstance(action, str) for action in raw_actions):
                errors.append(f"{sample_id}: every gold action must be a string")
            else:
                gold_actions = raw_actions
                for action in gold_actions:
                    parsed = parse_action(action)
                    if parsed is None:
                        errors.append(f"{sample_id}: invalid gold action syntax: {action}")
                        continue
                    name, args = parsed
                    if name not in allowed_actions:
                        errors.append(f"{sample_id}: action outside vocabulary: {name}")
                    if allowed_objects:
                        for arg in args:
                            if arg not in allowed_objects:
                                errors.append(f"{sample_id}: object outside vocabulary: {arg}")
            if not isinstance(gold.get("plan_nl"), str):
                errors.append(f"{sample_id}: gold.plan_nl must be a string")

        meta = _require_type(sample_id, row, "meta", dict, errors)
        if isinstance(meta, dict):
            if meta.get("sim_verified") is not True:
                errors.append(f"{sample_id}: meta.sim_verified must be true")
            if gold_actions and meta.get("plan_length") != len(gold_actions):
                errors.append(
                    f"{sample_id}: meta.plan_length must equal len(gold.plan_actions)"
                )
    return errors

