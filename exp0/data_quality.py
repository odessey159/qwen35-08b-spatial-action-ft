from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable


ATOMIC_GROUPS = frozenset(
    {"pickup", "clean", "heat", "toggle", "slice", "open_close"}
)
ACTION_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\((.*)\)\s*$")


def parse_action(action: str) -> tuple[str, tuple[str, ...]] | None:
    match = ACTION_PATTERN.fullmatch(str(action))
    if match is None:
        return None
    raw_arguments = match.group(2).strip()
    arguments = (
        tuple(argument.strip() for argument in raw_arguments.split(","))
        if raw_arguments
        else ()
    )
    return match.group(1), arguments


def instruction_plan_collisions(
    rows: Iterable[dict[str, Any]],
    image_identity: Callable[[dict[str, Any]], str] | None = None,
) -> dict[tuple[str, str], set[tuple[str, ...]]]:
    """Return inputs that map to more than one gold plan.

    The image content hash is preferred over its path when available. This also
    catches duplicate screenshots saved under different sample names, while
    counterfactual pairs with genuinely different state images remain valid.
    """

    plans_by_input: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        if image_identity is None:
            image_key = str(
                row.get("meta", {}).get("image_sha256") or row.get("image", "")
            )
        else:
            image_key = str(image_identity(row))
        key = (image_key, str(row.get("instruction", "")).strip())
        actions = row.get("gold", {}).get("plan_actions", [])
        plans_by_input[key].add(tuple(str(action) for action in actions))
    return {
        key: plans for key, plans in plans_by_input.items() if len(plans) > 1
    }


def generation_quality_summary(
    rows: Iterable[dict[str, Any]],
    image_identity: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    materialized = list(rows)
    action_distribution: Counter[str] = Counter()
    goto_schema: Counter[str] = Counter()

    for row in materialized:
        actions = [str(action) for action in row.get("gold", {}).get("plan_actions", [])]
        parsed_actions = [parse_action(action) for action in actions]
        for parsed in parsed_actions:
            if parsed is not None and parsed[0] != "GotoLocation":
                action_distribution[parsed[0]] += 1

        group = str(row.get("meta", {}).get("task_group", ""))
        if group not in ATOMIC_GROUPS or len(parsed_actions) < 2:
            continue
        goto = parsed_actions[0]
        target_action = parsed_actions[-1]
        if (
            goto is None
            or target_action is None
            or goto[0] != "GotoLocation"
            or not goto[1]
            or not target_action[1]
        ):
            continue
        key = "object_location" if goto[1][0] == target_action[1][0] else "parent_location"
        goto_schema[key] += 1

    collisions = instruction_plan_collisions(materialized, image_identity=image_identity)
    return {
        "instruction_collision": len(collisions),
        "action_distribution": dict(sorted(action_distribution.items())),
        "goto_schema": {
            "parent_location": goto_schema["parent_location"],
            "object_location": goto_schema["object_location"],
        },
    }
