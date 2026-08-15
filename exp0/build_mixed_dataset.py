from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_old_half(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["meta"]["task_group"])].append(row)

    selected: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_group.items()):
        if len(group_rows) % 2:
            raise ValueError(f"Old task group {group!r} has odd size {len(group_rows)}")
        target = len(group_rows) // 2
        if group != "counterfactual_put":
            selected.extend(rng.sample(group_rows, target))
            continue

        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            pair_id = str(row["meta"].get("counterfactual_group") or "")
            if not pair_id:
                raise ValueError("counterfactual_put row is missing counterfactual_group")
            by_pair[pair_id].append(row)
        bad_pairs = {key: len(value) for key, value in by_pair.items() if len(value) != 2}
        if bad_pairs:
            raise ValueError(f"Counterfactual groups are not pairs: {bad_pairs}")
        pair_ids = sorted(by_pair)
        for pair_id in rng.sample(pair_ids, target // 2):
            selected.extend(by_pair[pair_id])

    if len(selected) * 2 != len(rows):
        raise ValueError(f"Expected exactly half of old data; selected {len(selected)}/{len(rows)}")
    return selected


def add_partition(
    rows: list[dict[str, Any]],
    partition: str,
    source_dir: Path,
    output_images: Path,
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for index, original in enumerate(rows, start=1):
        row = copy.deepcopy(original)
        old_id = str(row["sample_id"])
        new_id = f"{partition}_{index:04d}"
        source_image = source_dir / str(row["image"])
        suffix = source_image.suffix.lower()
        image_name = f"{new_id}{suffix}"
        shutil.copy2(source_image, output_images / image_name)
        row["sample_id"] = new_id
        row["image"] = f"images/{image_name}"
        row["wrong_image"] = ""
        meta = row["meta"]
        meta["source_partition"] = partition
        meta["source_sample_id"] = old_id
        if meta.get("counterfactual_group"):
            meta["counterfactual_group"] = f"{partition}_{meta['counterfactual_group']}"
        copied.append(row)
    return copied


def assign_wrong_images(rows: list[dict[str, Any]], rng: random.Random) -> None:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["meta"]["scene_id"])].append(row)
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        raise ValueError("A_prime requires at least two scenes")
    for row in rows:
        own_scene = str(row["meta"]["scene_id"])
        wrong_scene = rng.choice([scene for scene in scenes if scene != own_scene])
        row["wrong_image"] = str(rng.choice(by_scene[wrong_scene])["image"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build new + 120-old Exp 0 dataset")
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    new_dir = args.new_dir.resolve()
    old_dir = args.old_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    new_rows = read_jsonl(new_dir / "samples.jsonl")
    old_rows = read_jsonl(old_dir / "samples.jsonl")
    if not new_rows or len(old_rows) != 240:
        raise ValueError(f"Expected non-empty new data and 240 old rows; got {len(new_rows)} and {len(old_rows)}")

    rng = random.Random(args.seed)
    old_selected = select_old_half(old_rows, rng)
    new_partition = f"new{len(new_rows)}"
    mixed = add_partition(new_rows, new_partition, new_dir, images_dir)
    mixed.extend(add_partition(old_selected, "old120", old_dir, images_dir))
    assign_wrong_images(mixed, rng)

    write_jsonl(output_dir / "samples.jsonl", mixed)
    write_json(
        output_dir / "selection_report.json",
        {
            "seed": args.seed,
            "sample_count": len(mixed),
            "partition_counts": dict(Counter(row["meta"]["source_partition"] for row in mixed)),
            "task_group_counts": dict(Counter(row["meta"]["task_group"] for row in mixed)),
            "old_selected_ids": [row["meta"]["source_sample_id"] for row in mixed if row["meta"]["source_partition"] == "old120"],
        },
    )
    print(f"mixed={len(mixed)} new={len(new_rows)} old_selected={len(old_selected)}")


if __name__ == "__main__":
    main()
