from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from .subgoal_abstraction import (
    abstract_subgoals,
    parse_primitive_action,
    validate_subgoal_abstraction,
)


VARIANTS = ("cot", "plan-only", "action-only")
SECTION_PATTERNS = {
    name: re.compile(rf"<{name}>\s*(.*?)\s*(?:</{name}>|$)", re.I | re.S)
    for name in ("state", "plan", "action")
}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\s\w]", re.UNICODE)
RELATION_TEXT = {
    "in": "{s} is inside {o}.",
    "inside": "{s} is inside {o}.",
    "on": "{s} is on {o}.",
    "contains": "{o} is inside {s}.",
    "left_of": "{s} is left of {o}.",
    "right_of": "{s} is right of {o}.",
    "above": "{s} is above {o}.",
    "below": "{s} is below {o}.",
    "near": "{s} is near {o}.",
    "close": "{s} is near {o}.",
    "far": "{s} is far from {o}.",
    "next_to": "{s} is next to {o}.",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _actions(sample: dict[str, Any]) -> list[str]:
    gold = sample.get("gold") if isinstance(sample.get("gold"), dict) else sample
    value = gold.get("plan_actions", sample.get("actions"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{sample.get('sample_id', '<unknown>')}: missing actions")
    result = [str(action).strip() for action in value]
    for action in result:
        parse_primitive_action(action)
    return result


def _label(obj: dict[str, Any]) -> str:
    return str(
        obj.get("id")
        or obj.get("objectId")
        or obj.get("name")
        or obj.get("type")
        or "Unknown"
    )


def _type(obj: dict[str, Any]) -> str:
    return str(obj.get("type") or _label(obj))


def _attribute_facts(label: str, attributes: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    if isinstance(attributes.get("is_open"), bool):
        facts.append(f"{label} is {'open' if attributes['is_open'] else 'closed'}.")
    if isinstance(attributes.get("is_dirty"), bool):
        facts.append(f"{label} is {'dirty' if attributes['is_dirty'] else 'clean'}.")
    if isinstance(attributes.get("is_toggled"), bool):
        facts.append(f"{label} is switched {'on' if attributes['is_toggled'] else 'off'}.")
    if attributes.get("is_picked_up") is True:
        facts.append(f"{label} is held by Agent.")
    return facts


def extract_task_relevant_state(sample: dict[str, Any], max_facts: int = 12) -> list[str]:
    if max_facts <= 0:
        raise ValueError("max_facts must be positive")
    actions = _actions(sample)
    requested = {
        argument.casefold()
        for action in actions
        for argument in parse_primitive_action(action).arguments
    }
    instruction = str(sample.get("instruction", sample.get("prompt", ""))).casefold()
    graph = sample.get("scene_graph", {})
    objects_list = graph.get("objects", []) if isinstance(graph, dict) else []
    relations = graph.get("relations", []) if isinstance(graph, dict) else []
    objects = {_label(obj): obj for obj in objects_list if isinstance(obj, dict)}
    meta = sample.get("meta", sample.get("_meta", {}))
    if not isinstance(meta, dict):
        meta = {}
    required_object_ids = {
        str(value).casefold()
        for value in meta.get("required_object_ids", [])
        if str(value).strip()
    }
    requested.update(
        _type(obj).casefold()
        for obj in objects.values()
        if _type(obj).casefold() != "agent" and _type(obj).casefold() in instruction
    )

    relation_entities: set[str] = set()
    for relation in relations:
        if isinstance(relation, dict):
            relation_entities.update(
                {str(relation.get("subject", "")), str(relation.get("object", ""))}
            )
    for relation in meta.get("spatial_relations", []):
        if isinstance(relation, (list, tuple)) and len(relation) == 3:
            relation_entities.update({str(relation[0]), str(relation[2])})

    direct: set[str] = set()
    for entity in requested:
        matches = [
            label
            for label, obj in objects.items()
            if entity in {label.casefold(), _type(obj).casefold()}
        ]
        exact = [label for label in matches if label.casefold() == entity]
        required = [
            label
            for label in matches
            if any(
                str(objects[label].get(key, "")).casefold() in required_object_ids
                for key in ("id", "objectId", "name")
            )
        ]
        related = [label for label in matches if label in relation_entities]
        direct.update(required or exact or related or (matches if len(matches) == 1 else []))

    parents: set[str] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        subject, obj = str(relation.get("subject", "")), str(relation.get("object", ""))
        name = str(relation.get("relation", "")).casefold()
        if name in {"in", "inside", "on", "contains"} and (subject in direct or obj in direct):
            parents.update({subject, obj} - direct)

    candidates: list[tuple[int, int, str]] = []
    order = 0
    for raw in sorted(direct | parents):
        if raw not in objects:
            continue
        obj = objects[raw]
        attributes = obj.get("attributes", {})
        if isinstance(attributes, dict):
            for text in _attribute_facts(_type(obj), attributes):
                candidates.append((0, order, text))
                order += 1
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        subject, obj = str(relation.get("subject", "")), str(relation.get("object", ""))
        name = str(relation.get("relation", "")).casefold()
        if subject not in direct and obj not in direct:
            continue
        s = _type(objects[subject]) if subject in objects else subject
        o = _type(objects[obj]) if obj in objects else obj
        text = RELATION_TEXT.get(name, "{s} is related to {o} by " + name.replace("_", " ") + ".")
        priority = 1 if name in {"in", "inside", "on", "contains"} else (2 if "Agent" in {s, o} else 3)
        candidates.append((priority, order, text.format(s=s, o=o)))
        order += 1
    if not candidates:
        spatial = sample.get("spatial_facts", [])
        if isinstance(spatial, list):
            for fact in spatial:
                text = str(fact).strip()
                if text and any(entity in text.casefold() for entity in requested):
                    candidates.append((4, order, text))
                    order += 1
    if not candidates:
        for raw in sorted(direct):
            if objects.get(raw, {}).get("attributes", {}).get("visible") is True:
                candidates.append((5, order, f"{_type(objects[raw])} is visible."))
                order += 1
    result: list[str] = []
    for _, _, text in sorted(candidates):
        if text not in result:
            result.append(text)
        if len(result) >= max_facts:
            break
    if not result:
        raise ValueError(f"{sample.get('sample_id', '<unknown>')}: no relevant state")
    return result


def render_response(
    state: Sequence[str], plan: Sequence[str], actions: Sequence[str], variant: str = "cot"
) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    blocks: list[str] = []
    if variant == "cot":
        blocks.append("<state>\n" + "\n".join(state) + "\n</state>")
    if variant in {"cot", "plan-only"}:
        numbered = [f"{index}. {value}" for index, value in enumerate(plan, start=1)]
        blocks.append("<plan>\n" + "\n".join(numbered) + "\n</plan>")
    blocks.append("<action>\n" + "\n".join(actions) + "\n</action>")
    return "\n\n".join(blocks)


def parse_response_sections(content: str) -> dict[str, list[str]]:
    result = {name: [] for name in SECTION_PATTERNS}
    for name, pattern in SECTION_PATTERNS.items():
        match = pattern.search(content)
        if match is None:
            continue
        lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        if name == "plan":
            lines = [re.sub(r"^\d+\s*[.、)]\s*", "", line) for line in lines]
        result[name] = lines
    return result


def build_cot_sample(
    sample: dict[str, Any], image: str | None = None, variant: str = "cot", max_state_facts: int = 12
) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id") or "")
    instruction = str(sample.get("instruction", sample.get("prompt", ""))).strip()
    if not sample_id or not instruction:
        raise ValueError("sample_id and instruction are required")
    actions = _actions(sample)
    state = extract_task_relevant_state(sample, max_state_facts)
    plan = abstract_subgoals(actions)
    meta = sample.get("meta", sample.get("_meta", {}))
    if not isinstance(meta, dict) or meta.get("sim_verified") is not True:
        raise ValueError(f"{sample_id}: simulator verification is required")
    if meta.get("target_visible") is False:
        raise ValueError(f"{sample_id}: target_visible must not be false")
    image_value = image or str(sample.get("image", ""))
    if not image_value.strip():
        raise ValueError(f"{sample_id}: image is required")
    scene_id = meta.get("scene_id")
    metadata = {
        "source": "ProcTHOR" if str(scene_id).casefold().startswith("procthor") else "AI2THOR",
        "verified": True,
        "sim_verified": True,
        "source_sample_id": sample_id,
        **{key: meta[key] for key in ("scene_id", "counterfactual_group", "task_group", "target_visible") if key in meta},
    }
    return {
        "sample_id": sample_id,
        "image": image_value,
        "instruction": instruction,
        "conversations": [
            {"role": "user", "content": [{"type": "image", "image": image_value}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": render_response(state, plan, actions, variant)},
        ],
        "oracle": {"state": state, "plan": plan, "actions": actions},
        "metadata": metadata,
    }


def validate_cot_sample(row: dict[str, Any], variant: str = "cot") -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    sample_id = str(row.get("sample_id", "<unknown>"))
    image = row.get("image")
    instruction = row.get("instruction")
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f"{sample_id}: image is required")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{sample_id}: instruction is required")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{sample_id}: invalid conversations")
    if [turn.get("role") for turn in conversations if isinstance(turn, dict)] != [
        "user",
        "assistant",
    ]:
        raise ValueError(f"{sample_id}: conversations must be user then assistant")
    user_content = conversations[0].get("content")
    if not isinstance(user_content, list) or len(user_content) != 2:
        raise ValueError(f"{sample_id}: invalid user content")
    if user_content[0] != {"type": "image", "image": image}:
        raise ValueError(f"{sample_id}: conversation image differs from image")
    if user_content[1] != {"type": "text", "text": instruction}:
        raise ValueError(f"{sample_id}: conversation instruction differs from instruction")
    content = conversations[1].get("content")
    if not isinstance(content, str):
        raise ValueError(f"{sample_id}: assistant content must be text")
    sections = parse_response_sections(content)
    required = {"action"} | ({"plan"} if variant != "action-only" else set()) | ({"state"} if variant == "cot" else set())
    if any(not sections[name] for name in required):
        raise ValueError(f"{sample_id}: missing required section")
    unexpected = {"state", "plan", "action"} - required
    if any(sections[name] for name in unexpected):
        raise ValueError(f"{sample_id}: unexpected response section")
    oracle = row.get("oracle", {})
    if not isinstance(oracle, dict):
        raise ValueError(f"{sample_id}: oracle must be an object")
    for name, key in (("state", "state"), ("plan", "plan"), ("action", "actions")):
        if name in required and sections[name] != oracle.get(key):
            raise ValueError(f"{sample_id}: {name} differs from oracle")
    for action in sections["action"]:
        parse_primitive_action(action)
    if sections["plan"]:
        validate_subgoal_abstraction(sections["action"], sections["plan"])
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{sample_id}: metadata must be an object")
    if metadata.get("verified") is not True or metadata.get("sim_verified") is not True:
        raise ValueError(f"{sample_id}: simulator verification must be true")


def estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def length_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    counts = {name: [] for name in SECTION_PATTERNS}
    for row in rows:
        sections = parse_response_sections(row["conversations"][1]["content"])
        for name in counts:
            counts[name].append(estimate_token_count("\n".join(sections[name])))
    return {f"average_{name}_tokens": round(mean(values), 3) for name, values in counts.items()}


def validate_cot_dataset(path: Path, variant: str = "cot") -> dict[str, Any]:
    path = path.resolve()
    rows = _read_jsonl(path)
    ids: set[str] = set()
    for row in rows:
        validate_cot_sample(row, variant)
        sample_id = str(row["sample_id"])
        if sample_id in ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        image = Path(str(row["image"]))
        if not image.is_absolute():
            image = path.parent / image
        if not image.resolve().is_file():
            raise FileNotFoundError(f"{sample_id}: image does not exist: {image}")
    return {
        "dataset": str(path),
        "variant": variant,
        "sample_count": len(rows),
        "all_required_sections_present": True,
        "all_actions_parseable": True,
        "all_plans_action_consistent": True,
        "all_images_present": True,
        "all_simulator_verified": True,
        "token_count_method": "deterministic_unicode_estimate",
        **length_statistics(rows),
    }


def _default_output(variant: str) -> Path:
    name = {"cot": "samples_cot.jsonl", "plan-only": "samples_plan.jsonl", "action-only": "samples_action.jsonl"}[variant]
    return Path(__file__).with_name("data_cot") / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simulator-supervised embodied CoT")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("data") / "samples.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="cot")
    parser.add_argument("--max-state-facts", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.input.resolve()
    output = (args.output or _default_output(args.variant)).resolve()
    if args.validate_only:
        print(json.dumps(validate_cot_dataset(output, args.variant), ensure_ascii=False, indent=2))
        return
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {output}")
    rows: list[dict[str, Any]] = []
    for source_row in _read_jsonl(source):
        image = Path(str(source_row.get("image", "")))
        if not image.is_absolute():
            image = source.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Image does not exist: {image}")
        relative = Path(os.path.relpath(image, output.parent)).as_posix()
        row = build_cot_sample(source_row, relative, args.variant, args.max_state_facts)
        validate_cot_sample(row, args.variant)
        rows.append(row)
    _write_jsonl(output, rows)
    report = {
        "source_dataset": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_dataset": str(output),
        "variant": args.variant,
        "sample_count": len(rows),
        "max_state_facts": args.max_state_facts,
        "all_simulator_verified": True,
        "token_count_method": "deterministic_unicode_estimate",
        **length_statistics(rows),
    }
    _write_json(output.with_name("generation_report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
