from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
from pathlib import Path
from typing import Any


def _reservoir(path: Path, count: int, seed: int) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    selected: list[tuple[int, str]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            seen += 1
            item = (index, line)
            if len(selected) < count:
                selected.append(item)
            else:
                replacement = rng.randrange(seen)
                if replacement < count:
                    selected[replacement] = item
    if seen < count:
        raise ValueError(f"Requested {count} rows from {path}, but only found {seen}")
    return sorted(selected)


def _longest(path: Path, count: int) -> list[tuple[int, str]]:
    selected: list[tuple[int, int, str]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            seen += 1
            item = (len(line), index, line)
            if len(selected) < count:
                heapq.heappush(selected, item)
            elif item[0] > selected[0][0]:
                heapq.heapreplace(selected, item)
    if seen < count:
        raise ValueError(f"Requested {count} rows from {path}, but only found {seen}")
    return sorted((index, line) for _, index, line in selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_lines(path: Path, selected: list[tuple[int, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for _, line in selected:
            handle.write(line.rstrip("\r\n") + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def select_prepared_subset(
    source_dir: Path,
    output_dir: Path,
    train_count: int,
    val_count: int,
    seed: int,
    overwrite: bool,
    strategy: str = "random",
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_count and val_count must be positive")
    source_manifest_path = source_dir / "manifest.json"
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {source_manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / name for name in ("train.jsonl", "val.jsonl", "manifest.json")]
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError(f"Subset outputs already exist; pass --overwrite: {output_dir}")

    if strategy not in {"random", "longest"}:
        raise ValueError("strategy must be 'random' or 'longest'")
    if strategy == "longest":
        train = _longest(source_dir / "train.jsonl", train_count)
        val = _longest(source_dir / "val.jsonl", val_count)
    else:
        train = _reservoir(source_dir / "train.jsonl", train_count, seed)
        val = _reservoir(source_dir / "val.jsonl", val_count, seed + 1)
    _write_lines(outputs[0], train)
    _write_lines(outputs[1], val)
    result = dict(manifest)
    train_ids = manifest.get("train_sample_ids", [])
    val_ids = manifest.get("validation_sample_ids", [])
    result.update(
        {
            "total_samples": train_count + val_count,
            "train_samples": train_count,
            "validation_samples": val_count,
            "train_sample_ids": [train_ids[index] for index, _ in train] if train_ids else [],
            "validation_sample_ids": [val_ids[index] for index, _ in val] if val_ids else [],
            "benchmark_subset": {
                "source_prepared_dir": str(source_dir),
                "seed": seed,
                "strategy": strategy,
                "train_count": train_count,
                "val_count": val_count,
            },
        }
    )
    _write_json(outputs[2], result)
    report = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "strategy": strategy,
        "train_count": train_count,
        "val_count": val_count,
        "train_sha256": _sha256(outputs[0]),
        "val_sha256": _sha256(outputs[1]),
    }
    _write_json(output_dir / "subset_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a deterministic prepared-data benchmark subset")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--strategy", choices=("random", "longest"), default="random")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            select_prepared_subset(
                args.source_dir,
                args.output_dir,
                args.train_count,
                args.val_count,
                args.seed,
                bool(args.overwrite),
                args.strategy,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
