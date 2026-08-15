from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from .data_quality import (
    ATOMIC_GROUPS,
    generation_quality_summary,
    instruction_plan_collisions,
    parse_action,
)


EXPECTED_GROUPS = {
    "counterfactual_put": 80,
    "pickup": 32,
    "clean": 32,
    "heat": 24,
    "toggle": 24,
    "slice": 24,
    "open_close": 24,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep integrity checks for generated Exp 0 data"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=240)
    parser.add_argument(
        "--expected-group",
        action="append",
        default=[],
        metavar="NAME=COUNT",
        help="Expected task-group count; repeat for custom dataset compositions",
    )
    parser.add_argument("--expected-counterfactual-pairs", type=int)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=672)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    rows = read_jsonl(dataset_dir / "samples.jsonl")
    require(len(rows) == args.expected_count, f"expected {args.expected_count} rows")

    sample_ids = [str(row["sample_id"]) for row in rows]
    image_names = [str(row["image"]) for row in rows]
    require(len(set(sample_ids)) == len(rows), "sample_id values are not unique")
    require(len(set(image_names)) == len(rows), "image paths are not unique")

    image_to_scene = {
        str(row["image"]): str(row["meta"]["scene_id"]) for row in rows
    }
    group_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    plan_lengths: Counter[int] = Counter()
    counterfactual_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    image_hashes: dict[str, str] = {}

    for row in rows:
        meta = row["meta"]
        group = str(meta["task_group"])
        scene_id = str(meta["scene_id"])
        group_counts[group] += 1
        scene_counts[scene_id] += 1
        plan_lengths[int(meta["plan_length"])] += 1

        require(bool(meta.get("sim_verified")), f"{row['sample_id']} is not simulator verified")
        require(bool(meta.get("target_visible")), f"{row['sample_id']} target is not visible")
        require(bool(meta.get("required_object_ids")), f"{row['sample_id']} has no required object")
        require(
            len(row["gold"]["plan_actions"]) == int(meta["plan_length"]),
            f"{row['sample_id']} plan_length mismatch",
        )

        actions = [parse_action(action) for action in row["gold"]["plan_actions"]]
        if group in ATOMIC_GROUPS:
            require(
                len(actions) >= 2 and actions[0] is not None and actions[-1] is not None,
                f"{row['sample_id']} has malformed atomic plan",
            )
            goto = actions[0]
            target_action = actions[-1]
            assert goto is not None and target_action is not None
            require(
                goto[0] == "GotoLocation" and bool(goto[1]) and bool(target_action[1]),
                f"{row['sample_id']} has malformed atomic action arguments",
            )

            scene_objects = row["scene_graph"]["objects"]
            type_by_alias = {
                str(obj["id"]): str(obj["type"])
                for obj in scene_objects
                if obj.get("id") != "Agent"
            }
            goto_type = goto[1][0]
            target_type = target_action[1][0]
            require(
                goto_type in set(type_by_alias.values()),
                f"{row['sample_id']} GotoLocation target is missing from scene graph",
            )
            visible_parent_types = {
                type_by_alias[str(relation["object"])]
                for relation in row["scene_graph"]["relations"]
                if relation.get("relation") in {"in", "on"}
                and type_by_alias.get(str(relation.get("subject"))) == target_type
                and str(relation.get("object")) in type_by_alias
            }
            if visible_parent_types:
                require(
                    goto_type in visible_parent_types,
                    f"{row['sample_id']} must navigate to visible parent; "
                    f"got {goto_type}, expected one of {sorted(visible_parent_types)}",
                )
            else:
                require(
                    goto_type == target_type,
                    f"{row['sample_id']} has no visible parent and must navigate to target",
                )

            instruction = str(row["instruction"])
            if target_action[0] == "ToggleObject":
                require(
                    "打开" not in instruction,
                    f"{row['sample_id']} uses ambiguous 打开 for ToggleObject",
                )
            if target_action[0] in {"OpenObject", "CloseObject"}:
                require(
                    "开启" not in instruction,
                    f"{row['sample_id']} uses device verb 开启 for open/close action",
                )

        image_path = dataset_dir / str(row["image"])
        require(image_path.is_file(), f"missing image: {image_path}")
        image_hashes[str(row["image"])] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        with Image.open(image_path) as image:
            require(
                image.size == (args.width, args.height),
                f"unexpected image size for {image_path}: {image.size}",
            )

        wrong_image = str(row["wrong_image"])
        require(wrong_image in image_to_scene, f"unknown wrong_image: {wrong_image}")
        require(wrong_image != row["image"], f"{row['sample_id']} uses itself as wrong_image")
        require(
            image_to_scene[wrong_image] != scene_id,
            f"{row['sample_id']} wrong_image is from the same scene",
        )

        pair_id = meta.get("counterfactual_group")
        if pair_id:
            counterfactual_groups[str(pair_id)].append(row)

    collisions = instruction_plan_collisions(
        rows, image_identity=lambda row: image_hashes[str(row["image"])]
    )
    if collisions:
        (image_hash, instruction), plans = next(iter(collisions.items()))
        rendered_plans = sorted(" -> ".join(plan) for plan in plans)
        require(
            False,
            "ambiguous (image, instruction) maps to multiple gold plans: "
            f"image_sha256={image_hash!r}, instruction={instruction!r}, "
            f"plans={rendered_plans}",
        )

    expected_groups = EXPECTED_GROUPS
    if args.expected_group:
        expected_groups = {}
        for item in args.expected_group:
            name, separator, count = item.partition("=")
            require(bool(separator and name and count), f"invalid --expected-group: {item}")
            expected_groups[name] = int(count)
    require(dict(group_counts) == expected_groups, f"unexpected groups: {dict(group_counts)}")
    expected_pairs = (
        args.expected_counterfactual_pairs
        if args.expected_counterfactual_pairs is not None
        else 40
    )
    require(
        len(counterfactual_groups) == expected_pairs,
        f"expected {expected_pairs} counterfactual pairs",
    )
    for pair_id, pair in counterfactual_groups.items():
        require(len(pair) == 2, f"{pair_id} does not contain exactly two samples")
        require(len({row["instruction"] for row in pair}) == 1, f"{pair_id} instruction differs")
        require(
            len({row["meta"]["scene_id"] for row in pair}) == 1,
            f"{pair_id} spans multiple scenes",
        )
        require(
            len({row["meta"]["receptacle_state"] for row in pair}) == 2,
            f"{pair_id} does not encode two states",
        )
        require(
            len({tuple(row["gold"]["plan_actions"]) for row in pair}) == 2,
            f"{pair_id} plans are identical",
        )

    actual_images = list((dataset_dir / "images").glob("*.png"))
    require(len(actual_images) == len(rows), "image directory count does not match rows")

    summary = {
        "sample_count": len(rows),
        "scene_count": len(scene_counts),
        "image_count": len(actual_images),
        "group_counts": dict(group_counts),
        "plan_length_counts": dict(sorted(plan_lengths.items())),
        "counterfactual_pairs": len(counterfactual_groups),
        "all_sim_verified": True,
        "all_wrong_images_cross_scene": True,
        **generation_quality_summary(
            rows, image_identity=lambda row: image_hashes[str(row["image"])]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
