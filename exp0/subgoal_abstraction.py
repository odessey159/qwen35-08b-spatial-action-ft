from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


ACTION_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


@dataclass(frozen=True)
class PrimitiveAction:
    name: str
    arguments: tuple[str, ...]


def parse_primitive_action(value: str) -> PrimitiveAction:
    match = ACTION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Invalid primitive action: {value!r}")
    name, raw = match.groups()
    arguments = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not arguments:
        raise ValueError(f"Primitive action has no object argument: {value!r}")
    return PrimitiveAction(name, arguments)


def abstract_subgoals(actions: Sequence[str] | Iterable[str]) -> list[str]:
    parsed = [parse_primitive_action(str(value)) for value in actions]
    if not parsed:
        raise ValueError("Cannot abstract an empty action sequence")
    # Navigation and container access are implementation details of a larger
    # manipulation goal. Keep Open/Close only when changing that container is
    # itself the task; otherwise an Open -> Pickup trace should become one
    # "Acquire" subgoal rather than an action-by-action paraphrase.
    has_primary_manipulation = any(
        action.name
        in {
            "PickupObject",
            "PutObject",
            "ToggleObject",
            "CleanObject",
            "HeatObject",
            "SliceObject",
        }
        for action in parsed
    )
    result: list[str] = []

    def add(text: str) -> None:
        if text not in result:
            result.append(text)

    for action in parsed:
        obj = action.arguments[0]
        if action.name == "GotoLocation":
            continue
        if action.name == "PickupObject":
            add(f"Acquire {obj}.")
        elif action.name == "PutObject":
            if len(action.arguments) < 2:
                raise ValueError("PutObject requires object and destination")
            target = action.arguments[1]
            add(f"Transport {obj} to {target}.")
            add(f"Place {obj} at {target}.")
        elif action.name in {"OpenObject", "CloseObject"} and has_primary_manipulation:
            continue
        elif action.name == "OpenObject":
            add(f"Make {obj} accessible.")
        elif action.name == "CloseObject":
            add(f"Secure {obj}.")
        elif action.name == "ToggleObject":
            add(f"Change the operating state of {obj}.")
        elif action.name == "CleanObject":
            add(f"Restore {obj} to a clean state.")
        elif action.name == "HeatObject":
            add(f"Bring {obj} to the required heated state.")
        elif action.name == "SliceObject":
            add(f"Prepare {obj} in sliced form.")
        else:
            add(f"Complete the required manipulation of {obj}.")
    if not result:
        for action in parsed:
            if action.name == "GotoLocation":
                add(f"Reach {action.arguments[0]} for the task.")
    return result


def validate_subgoal_abstraction(actions: Sequence[str], subgoals: Sequence[str]) -> None:
    expected = abstract_subgoals(actions)
    actual = [str(value).strip() for value in subgoals if str(value).strip()]
    if actual != expected:
        raise ValueError(
            f"High-level plan is inconsistent with actions: expected={expected!r}, actual={actual!r}"
        )
