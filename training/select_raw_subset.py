from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import TrainingConfigError


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TrainingConfigError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingConfigError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TrainingConfigError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _whole_scene_subset(
    by_scene: dict[str, list[dict[str, Any]]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Select exactly count rows without taking a partial scene."""
    groups = list(by_scene.values())
    random.Random(seed).shuffle(groups)
    reachable = 1
    mask = (1 << (count + 1)) - 1
    history: list[tuple[list[dict[str, Any]], int]] = []
    for group in groups:
        before = reachable
        reachable = (reachable | (reachable << len(group))) & mask
        history.append((group, before))
        if (reachable >> count) & 1:
            break
    if not (reachable >> count) & 1:
        raise TrainingConfigError(
            f"Cannot select exactly {count} rows while preserving whole scenes"
        )

    remaining = count
    selected: list[dict[str, Any]] = []
    for group, before in reversed(history):
        if (before >> remaining) & 1:
            continue
        selected.extend(group)
        remaining -= len(group)
    if remaining or len(selected) != count:
        raise RuntimeError("Internal whole-scene subset reconstruction error")
    return selected


def select_raw_subset(
    shard_root: Path,
    shard_count: int,
    output_dir: Path,
    sample_count: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    shard_root = shard_root.resolve()
    output_dir = output_dir.resolve()
    if shard_count <= 0:
        raise TrainingConfigError("shard_count must be positive")
    if sample_count <= 0:
        raise TrainingConfigError("sample_count must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output is not empty; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_dir = shard_root / f"shard_{shard_index}"
        source_path = shard_dir / "samples.jsonl"
        report_path = shard_dir / "generation_report.json"
        if not source_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing samples/report for shard_{shard_index}")
        rows = _read_jsonl(source_path)
        report = _read_json(report_path)
        reported_samples = int(report.get("sample_count", -1))
        if reported_samples < 0 or reported_samples > len(rows):
            raise TrainingConfigError(
                f"shard_{shard_index} report/sample mismatch: "
                f"{reported_samples} reported, {len(rows)} readable"
            )
        counterfactual_counts = Counter(
            str(row.get("meta", {}).get("counterfactual_group"))
            for row in rows
            if isinstance(row.get("meta"), dict)
            and row["meta"].get("counterfactual_group") not in (None, "")
        )
        incomplete_pairs = {
            name: pair_count
            for name, pair_count in counterfactual_counts.items()
            if pair_count != 2
        }
        if incomplete_pairs:
            example = next(iter(incomplete_pairs.items()))
            raise TrainingConfigError(
                f"shard_{shard_index} has an incomplete counterfactual group: "
                f"{example[0]} has {example[1]} rows"
            )
        source_records.append(
            {
                "shard": shard_index,
                "path": str(source_path),
                "samples": len(rows),
                "reported_samples": reported_samples,
                "unreported_tail_samples": len(rows) - reported_samples,
                "sha256": _sha256(source_path),
            }
        )
        for source_index, original in enumerate(rows):
            row = copy.deepcopy(original)
            old_id = str(row.get("sample_id") or f"row_{source_index + 1:07d}")
            row["sample_id"] = f"shard_{shard_index}_{old_id}"
            image_value = row.get("image")
            if not isinstance(image_value, str) or not image_value.strip():
                raise TrainingConfigError(f"{row['sample_id']}: image is required")
            image_path = Path(image_value).expanduser()
            if not image_path.is_absolute():
                image_path = shard_dir / image_path
            image_path = image_path.resolve()
            if not image_path.is_file():
                raise TrainingConfigError(f"{row['sample_id']}: missing image {image_path}")
            row["image"] = str(image_path)
            metadata = row.get("meta")
            if not isinstance(metadata, dict):
                raise TrainingConfigError(f"{row['sample_id']}: meta must be an object")
            scene_id = metadata.get("scene_id")
            if not isinstance(scene_id, str) or not scene_id:
                raise TrainingConfigError(f"{row['sample_id']}: meta.scene_id is required")
            counterfactual_group = metadata.get("counterfactual_group")
            if counterfactual_group not in (None, ""):
                metadata["counterfactual_group"] = (
                    f"shard_{shard_index}_{counterfactual_group}"
                )
            row["_selection_source"] = {
                "shard": shard_index,
                "source_index": source_index,
            }
            all_rows.append(row)

    if sample_count > len(all_rows):
        raise TrainingConfigError(
            f"Requested {sample_count} samples, but only {len(all_rows)} are available"
        )
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_scene[str(row["meta"]["scene_id"])].append(row)
    selected = _whole_scene_subset(by_scene, sample_count, seed)
    selected.sort(
        key=lambda row: (
            int(row["_selection_source"]["shard"]),
            int(row["_selection_source"]["source_index"]),
        )
    )
    if len({row["sample_id"] for row in selected}) != sample_count:
        raise TrainingConfigError("Selected sample_id values are not unique")

    output_path = output_dir / "samples.jsonl"
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            row.pop("_selection_source", None)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    shard_counts = Counter()
    task_counts = Counter()
    scene_ids = set()
    counterfactual_groups = set()
    for row in selected:
        shard_counts[row["sample_id"].split("_", 2)[1]] += 1
        metadata = row["meta"]
        task_counts[str(metadata.get("task_group", "unknown"))] += 1
        scene_ids.add(str(metadata["scene_id"]))
        if metadata.get("counterfactual_group"):
            counterfactual_groups.add(str(metadata["counterfactual_group"]))
    manifest = {
        "seed": seed,
        "available_samples": len(all_rows),
        "selected_samples": len(selected),
        "whole_scene_selection": True,
        "selected_scenes": len(scene_ids),
        "selected_counterfactual_groups": len(counterfactual_groups),
        "shard_counts": dict(sorted(shard_counts.items())),
        "task_group_counts": dict(sorted(task_counts.items())),
        "sources": source_records,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    manifest_path = output_dir / "selection_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a deterministic whole-scene subset from raw generation shards"
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = select_raw_subset(
        args.shard_root,
        args.shards,
        args.output_dir,
        args.count,
        args.seed,
        bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
