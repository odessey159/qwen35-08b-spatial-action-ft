from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .data_quality import generation_quality_summary, instruction_plan_collisions
except ImportError:  # Support `python exp0/merge_shards.py` from the repository root.
    from data_quality import generation_quality_summary, instruction_plan_collisions


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint ProcTHOR generation shards")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_root = args.shard_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    shard_reports: list[dict[str, Any]] = []

    for shard_index in range(args.shards):
        shard_dir = shard_root / f"shard_{shard_index}"
        report = read_json(shard_dir / "generation_report.json")
        shard_rows = read_jsonl(shard_dir / "samples.jsonl")
        if int(report["sample_count"]) != len(shard_rows):
            raise RuntimeError(f"Shard {shard_index} report/sample mismatch")
        shard_reports.append(report)
        rejection_counts.update(report.get("rejections", {}))

        for original in shard_rows:
            row = copy.deepcopy(original)
            old_id = str(row["sample_id"])
            new_id = f"exp0_{len(rows) + 1:04d}"
            old_image = Path(str(row["image"]))
            suffix = old_image.stem[len(old_id) :] if old_image.stem.startswith(old_id) else ""
            new_name = f"{new_id}{suffix}{old_image.suffix}"
            new_image_path = images_dir / new_name
            shutil.copy2(shard_dir / old_image, new_image_path)
            row["sample_id"] = new_id
            row["image"] = f"images/{new_name}"
            row["wrong_image"] = ""
            row["meta"]["image_sha256"] = hashlib.sha256(
                new_image_path.read_bytes()
            ).hexdigest()
            counterfactual_group = row["meta"].get("counterfactual_group")
            if counterfactual_group:
                row["meta"]["counterfactual_group"] = (
                    f"shard_{shard_index}_{counterfactual_group}"
                )
            rows.append(row)
            group_counts[row["meta"]["task_group"]] += 1
            scene_counts[row["meta"]["scene_id"]] += 1

    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"Merged sample count mismatch: expected {args.expected_count}, got {len(rows)}"
        )
    if len(scene_counts) < 2:
        raise RuntimeError("A_prime requires at least two scenes")

    collisions = instruction_plan_collisions(rows)
    if collisions:
        image, instruction = next(iter(collisions))
        raise RuntimeError(
            "Ambiguous merged input maps to multiple gold plans: "
            f"image={image!r}, instruction={instruction!r}"
        )

    rng = random.Random(args.seed)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[row["meta"]["scene_id"]].append(row)
    scene_ids = sorted(by_scene)
    for row in rows:
        own_scene = row["meta"]["scene_id"]
        wrong_scene = rng.choice([scene for scene in scene_ids if scene != own_scene])
        row["wrong_image"] = rng.choice(by_scene[wrong_scene])["image"]

    sources = sorted({str(report.get("source", "unknown")) for report in shard_reports})
    write_jsonl(output_dir / "samples.jsonl", rows)
    write_json(
        output_dir / "generation_report.json",
        {
            "seed": args.seed,
            "source": sources[0] if len(sources) == 1 else "+".join(sources),
            "sample_count": len(rows),
            "scene_count": len(scene_counts),
            "group_counts": dict(group_counts),
            "scene_counts": dict(scene_counts),
            "counterfactual_ratio": group_counts["counterfactual_put"] / len(rows),
            "rejections": dict(rejection_counts),
            "shard_reports": shard_reports,
            **generation_quality_summary(rows),
        },
    )
    print(
        f"merged={len(rows)} scenes={len(scene_counts)} groups={dict(group_counts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
