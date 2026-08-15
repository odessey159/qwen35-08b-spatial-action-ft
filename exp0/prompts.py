from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


B_FORMATS = ("natural", "json", "triples")

# The format example deliberately uses object types that do NOT appear anywhere in
# the diagnostic set, and a 3-step plan length that no gold plan has (gold lengths
# are 1 / 2 / 4 / 6). Both properties make example copying detectable instead of
# silently inflating scores -- see `example_echo_rate` in evaluation.py.
FORMAT_EXAMPLE_OBJECTS = ("BathtubBasin", "WateringCan")
FORMAT_EXAMPLE_PLAN = (
    "GotoLocation(BathtubBasin)",
    "PickupObject(WateringCan)",
    "CleanObject(WateringCan)",
)
FORMAT_EXAMPLE_SUMMARY = "走到 BathtubBasin 前，拿起 WateringCan，把它清洗干净。"

_EXAMPLE_BLOCK = (
    "<plan>\n"
    + "\n".join(FORMAT_EXAMPLE_PLAN)
    + "\n</plan>\n<summary>\n"
    + FORMAT_EXAMPLE_SUMMARY
    + "\n</summary>"
)

_ACTION_EXAMPLE_BLOCK = (
    "<action>\n" + "\n".join(FORMAT_EXAMPLE_PLAN) + "\n</action>"
)


@dataclass(frozen=True)
class PromptCase:
    condition: str
    scene_graph_format: str | None
    prompt: str
    image_path: Path | None


def _object_label(obj: dict[str, Any]) -> str:
    return str(obj.get("id") or obj.get("name") or obj.get("type") or "Unknown")


def serialize_scene_graph(scene_graph: dict[str, Any], format_name: str) -> str:
    if format_name == "json":
        return json.dumps(scene_graph, ensure_ascii=False, sort_keys=True, indent=2)

    objects = scene_graph.get("objects", [])
    relations = scene_graph.get("relations", [])

    if format_name == "natural":
        lines = ["场景中的物体："]
        for obj in objects:
            label = _object_label(obj)
            obj_type = obj.get("type", label)
            attributes = obj.get("attributes", {})
            attr_text = "，".join(
                f"{key}={json.dumps(value, ensure_ascii=False)}"
                for key, value in sorted(attributes.items())
            )
            suffix = f"，属性：{attr_text}" if attr_text else ""
            lines.append(f"- {label}，类型：{obj_type}{suffix}")
        lines.append("场景关系：")
        for relation in relations:
            lines.append(
                "- "
                f"{relation.get('subject', 'Unknown')} "
                f"{relation.get('relation', 'related_to')} "
                f"{relation.get('object', 'Unknown')}"
            )
        return "\n".join(lines)

    if format_name == "triples":
        triples: list[str] = []
        for obj in objects:
            label = _object_label(obj)
            triples.append(f"({label}, type, {obj.get('type', label)})")
            for key, value in sorted(obj.get("attributes", {}).items()):
                triples.append(
                    f"({label}, {key}, {json.dumps(value, ensure_ascii=False)})"
                )
        for relation in relations:
            triples.append(
                f"({relation.get('subject', 'Unknown')}, "
                f"{relation.get('relation', 'related_to')}, "
                f"{relation.get('object', 'Unknown')})"
            )
        return "\n".join(triples)

    raise ValueError(f"Unsupported scene graph format: {format_name}")


def system_prompt(
    allowed_actions: list[str], inline_example: bool, response_format: str = "legacy"
) -> str:
    """Format rules live here, not in the user turn.

    Everything the model is *told to do* is kept out of the user turn so that a
    verbatim copy of the rules cannot be mistaken for an answer. The previous
    contract embedded the literal placeholder `ActionName(Object)` plus the
    sentence "随后用一句自然语言描述相同步骤。" inside the user turn; 0.8B copied
    both back (61% and 91% of outputs respectively).
    """
    if response_format not in {"legacy", "action"}:
        raise ValueError("response_format must be 'legacy' or 'action'")
    action_text = "、".join(allowed_actions)
    if response_format == "action":
        lines = [
            "你是一个室内家务动作规划助手。",
            "",
            "回答必须且只能由以下一段组成，不要写思考过程、解释或复述题目：",
            "",
            "<action>",
            "（每行一个动作，写成 动作名(物体名) 的形式；多个参数用英文逗号分隔）",
            "</action>",
            "",
            f"动作名只能从这 9 个里选：{action_text}。",
            "物体名一律使用英文原名，例如 Fridge、Plate。",
            "步骤数由任务本身决定，可以是一步，也可以是多步。",
        ]
        if inline_example:
            lines += [
                "",
                "下面是一个格式示例。它与你要回答的任务无关，不要照抄：",
                "",
                _ACTION_EXAMPLE_BLOCK,
            ]
        return "\n".join(lines)

    lines = [
        "你是一个室内家务动作规划助手。",
        "",
        "回答必须且只能由以下两段组成，不要写任何思考过程、解释或复述题目：",
        "",
        "<plan>",
        "（每行一个动作，写成 动作名(物体名) 的形式；多个参数用英文逗号分隔）",
        "</plan>",
        "<summary>",
        "（一句中文，把上面这些步骤连起来说清楚）",
        "</summary>",
        "",
        f"动作名只能从这 9 个里选：{action_text}。",
        "物体名一律使用英文原名，例如 Fridge、Plate。",
        "步骤数由任务本身决定，可以是一步，也可以是多步。",
        "<summary> 部分是最终要交付的内容，必须完整、通顺、与 <plan> 一致。",
    ]
    if inline_example:
        lines += [
            "",
            "下面是一个格式示例。它与你要回答的任务无关，只用来说明排版，"
            "不要照抄里面的动作和物体：",
            "",
            _EXAMPLE_BLOCK,
        ]
    return "\n".join(lines)


def format_demo_turns(response_format: str = "legacy") -> list[dict[str, Any]]:
    """One-shot format demo as a real user/assistant exchange.

    Enabled by `model.format_demo_as_turns` in the config. Small models follow a
    demonstrated assistant turn far more reliably than an inline template. The
    demo carries no scene, so it teaches layout only, and it is identical across
    all seven conditions -- it cannot shift the between-condition comparison.
    """
    example = _ACTION_EXAMPLE_BLOCK if response_format == "action" else _EXAMPLE_BLOCK
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "目标指令：清洗 WateringCan。（格式示例，与后面的题目无关）"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": example}]},
    ]


def build_prompt_cases(
    sample: dict[str, Any],
    dataset_dir: Path,
    allowed_actions: list[str],
) -> list[PromptCase]:
    instruction = sample["instruction"]
    scene_graph = sample["scene_graph"]
    correct_image = (dataset_dir / sample["image"]).resolve()
    wrong_image = (dataset_dir / sample["wrong_image"]).resolve()

    cases = [
        PromptCase(
            condition="D",
            scene_graph_format=None,
            image_path=correct_image,
            prompt=(
                f"目标指令：{instruction}\n\n"
                "已知的正确子目标：\n- "
                + "\n- ".join(sample["subgoals"])
            ),
        ),
        PromptCase(
            condition="A_prime",
            scene_graph_format=None,
            image_path=wrong_image,
            prompt=f"目标指令：{instruction}",
        ),
        PromptCase(
            condition="A",
            scene_graph_format=None,
            image_path=correct_image,
            prompt=f"目标指令：{instruction}",
        ),
    ]

    for format_name in B_FORMATS:
        cases.append(
            PromptCase(
                condition=f"B_{format_name}",
                scene_graph_format=format_name,
                image_path=None,
                prompt=(
                    f"目标指令：{instruction}\n\n"
                    f"场景图（{format_name}）：\n"
                    f"{serialize_scene_graph(scene_graph, format_name)}"
                ),
            )
        )

    cases.append(
        PromptCase(
            condition="C",
            scene_graph_format=None,
            image_path=correct_image,
            prompt=(
                f"目标指令：{instruction}\n\n"
                "已知的正确空间事实：\n- "
                + "\n- ".join(sample["spatial_facts"])
            ),
        )
    )
    return cases


def build_messages(
    prompt: str,
    image_path: Path | None,
    allowed_actions: list[str],
    format_demo_as_turns: bool = False,
    response_format: str = "legacy",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt(
                        allowed_actions,
                        inline_example=not format_demo_as_turns,
                        response_format=response_format,
                    ),
                }
            ],
        }
    ]
    if format_demo_as_turns:
        messages.extend(format_demo_turns(response_format=response_format))

    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image", "path": str(image_path)})
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages
