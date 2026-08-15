from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


B_FORMATS = ("natural", "json", "triples")


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


def _output_contract(allowed_actions: list[str]) -> str:
    action_text = " ".join(allowed_actions)
    return f"""可用动作仅限：{action_text}
请严格输出以下两部分，不要输出分析过程：
<plan>
ActionName(Object)
...</plan>
随后用一句自然语言描述相同步骤。
动作序列必须放在最前面，每行一个动作。"""


def build_prompt_cases(
    sample: dict[str, Any],
    dataset_dir: Path,
    allowed_actions: list[str],
) -> list[PromptCase]:
    instruction = sample["instruction"]
    scene_graph = sample["scene_graph"]
    contract = _output_contract(allowed_actions)
    correct_image = (dataset_dir / sample["image"]).resolve()
    wrong_image = (dataset_dir / sample["wrong_image"]).resolve()

    cases = [
        PromptCase(
            condition="D",
            scene_graph_format=None,
            image_path=correct_image,
            prompt=(
                f"目标指令：{instruction}\n\n"
                "正确子目标：\n- "
                + "\n- ".join(sample["subgoals"])
                + f"\n\n{contract}"
            ),
        ),
        PromptCase(
            condition="A_prime",
            scene_graph_format=None,
            image_path=wrong_image,
            prompt=f"目标指令：{instruction}\n\n请根据图像生成计划。\n\n{contract}",
        ),
        PromptCase(
            condition="A",
            scene_graph_format=None,
            image_path=correct_image,
            prompt=f"目标指令：{instruction}\n\n请根据图像生成计划。\n\n{contract}",
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
                    f"正确场景图（{format_name}）：\n"
                    f"{serialize_scene_graph(scene_graph, format_name)}\n\n"
                    f"请根据场景图生成计划。\n\n{contract}"
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
                "正确空间事实：\n- "
                + "\n- ".join(sample["spatial_facts"])
                + f"\n\n请结合图像和空间事实生成计划。不要假设未提供的子目标。\n\n{contract}"
            ),
        )
    )
    return cases


def build_messages(prompt: str, image_path: Path | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image", "path": str(image_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
