from __future__ import annotations

import argparse
import random
from collections import Counter

import prior


COOKABLE_TYPES = {"Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato"}


def walk(objects: list[dict]) -> list[dict]:
    result: list[dict] = []
    for obj in objects:
        result.append(obj)
        result.extend(walk(obj.get("children", [])))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", type=int, default=30)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    split = prior.load_dataset("procthor-10k", revision=args.revision)["train"]
    indices = list(range(len(split)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[: args.scenes]
    base, remainder = divmod(len(selected), args.shards)
    sizes = [base + (1 if index < remainder else 0) for index in range(args.shards)]
    offset = 0
    for shard_index, size in enumerate(sizes):
        shard_indices = selected[offset : offset + size]
        offset += size
        counts: Counter[str] = Counter()
        scenes_with_food = 0
        for scene_index in shard_indices:
            types = [obj["id"].split("|", 1)[0] for obj in walk(split[scene_index]["objects"])]
            found = Counter(item for item in types if item in COOKABLE_TYPES)
            counts.update(found)
            scenes_with_food += bool(found)
        print(
            f"shard={shard_index} scenes={shard_indices} "
            f"scenes_with_cookable={scenes_with_food} cookables={dict(counts)}"
        )


if __name__ == "__main__":
    main()
