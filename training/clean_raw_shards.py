from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import TrainingConfigError
from .cot_data import _normalize_raw_simulator_sample


FAILURE_STATUSES = {"failed", "failure", "error", "invalid", "rejected"}


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
            row["_clean_source"] = {"path": str(path), "line_number": line_number}
            rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _explicit_failure_reason(row: dict[str, Any]) -> str | None:
    containers = [("row", row)]
    metadata = row.get("meta")
    if isinstance(metadata, dict):
        containers.append(("meta", metadata))
    for prefix, value in containers:
        status = value.get("status")
        if isinstance(status, str) and status.strip().casefold() in FAILURE_STATUSES:
            return f"{prefix}.status={status.strip()}"
        if value.get("failed") is True:
            return f"{prefix}.failed=true"
        if value.get("success") is False:
            return f"{prefix}.success=false"
    if not isinstance(metadata, dict):
        return "meta_missing"
    if metadata.get("sim_verified") is not True and metadata.get("verified") is not True:
        return "simulator_not_verified"
    if metadata.get("target_visible") is False:
        return "target_not_visible"
    return None


def _structure_failure_reason(row: dict[str, Any], shard_dir: Path) -> str | None:
    instruction = row.get("instruction", row.get("prompt"))
    if not isinstance(instruction, str) or not instruction.strip():
        return "instruction_missing"
    gold = row.get("gold")
    actions = gold.get("plan_actions") if isinstance(gold, dict) else None
    if (
        not isinstance(actions, list)
        or not actions
        or any(not isinstance(action, str) or not action.strip() for action in actions)
    ):
        return "gold_plan_actions_invalid"
    image_value = row.get("image")
    if not isinstance(image_value, str) or not image_value.strip():
        return "image_missing"
    image = Path(image_value).expanduser()
    if not image.is_absolute():
        image = shard_dir / image
    if not image.is_file():
        return "image_file_missing"
    return None


def _preprocess_failure_reason(
    row: dict[str, Any], line_number: int, shard_dir: Path
) -> str | None:
    try:
        _normalize_raw_simulator_sample(
            row,
            line_number,
            shard_dir,
            require_sim_verified=True,
            max_state_facts=12,
        )
    except TrainingConfigError as exc:
        message = str(exc).casefold()
        if "no relevant state" in message:
            return "preprocess_no_relevant_state"
        if "plan_actions" in message:
            return "preprocess_invalid_plan_actions"
        return "preprocess_invalid"
    return None


def _collision_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("meta")
    image_hash = metadata.get("image_sha256") if isinstance(metadata, dict) else None
    image_identity = str(image_hash or row.get("image", ""))
    instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
    return image_identity, instruction


def _plan_signature(row: dict[str, Any]) -> tuple[str, ...]:
    gold = row.get("gold")
    actions = gold.get("plan_actions", []) if isinstance(gold, dict) else []
    return tuple(str(action).strip() for action in actions)


def clean_raw_shards(
    shard_root: Path,
    output_dir: Path,
    shard_count: int,
    overwrite: bool,
) -> dict[str, Any]:
    shard_root = shard_root.resolve()
    output_dir = output_dir.resolve()
    if shard_count <= 0:
        raise TrainingConfigError("shard_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        output_dir / "samples.jsonl",
        output_dir / "failed_samples.jsonl",
        output_dir / "cleaning_report.json",
    ]
    if not overwrite and any(path.exists() for path in output_files):
        raise FileExistsError(f"Cleaned outputs already exist; pass --overwrite: {output_dir}")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for shard_index in range(shard_count):
        shard_dir = shard_root / f"shard_{shard_index}"
        samples_path = shard_dir / "samples.jsonl"
        report_path = shard_dir / "generation_report.json"
        if not samples_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing samples/report in {shard_dir}")
        report = _read_json(report_path)
        rows = _read_jsonl(samples_path)
        reported_samples = int(report.get("sample_count", -1))
        expected_samples = int(report.get("expected_sample_count", reported_samples))
        if (
            reported_samples < 0
            or reported_samples > len(rows)
            or len(rows) not in {reported_samples, expected_samples}
        ):
            raise TrainingConfigError(
                f"shard_{shard_index} report/sample mismatch: "
                f"{reported_samples} reported, {expected_samples} expected, "
                f"{len(rows)} readable"
            )
        rejection_counts.update(
            {str(name): int(count) for name, count in report.get("rejections", {}).items()}
        )
        source_reports.append(
            {
                "shard": shard_index,
                "samples": len(rows),
                "reported_samples": reported_samples,
                "expected_samples": expected_samples,
                "unreported_tail_samples": len(rows) - reported_samples,
                "reported_instruction_collisions": int(report.get("instruction_collision", 0)),
                "complete": bool(report.get("complete", False)),
                "samples_sha256": _sha256(samples_path),
            }
        )
        for row in rows:
            source = row.pop("_clean_source")
            original_id = str(row.get("sample_id") or f"row_{source['line_number']:07d}")
            sample_id = f"shard_{shard_index}_{original_id}"
            if sample_id in seen_ids:
                raise TrainingConfigError(f"Duplicate namespaced sample_id: {sample_id}")
            seen_ids.add(sample_id)
            failure_reason = (
                _explicit_failure_reason(row)
                or _structure_failure_reason(row, shard_dir)
                or _preprocess_failure_reason(row, source["line_number"], shard_dir)
            )
            if failure_reason:
                failures.append(
                    {
                        "status": "failed",
                        "reason": failure_reason,
                        "shard": shard_index,
                        "line_number": source["line_number"],
                        "sample_id": original_id,
                    }
                )
                continue
            normalized = copy.deepcopy(row)
            normalized["sample_id"] = sample_id
            image = Path(str(normalized["image"])).expanduser()
            if not image.is_absolute():
                image = shard_dir / image
            normalized["image"] = str(image.resolve())
            metadata = normalized["meta"]
            counterfactual_group = metadata.get("counterfactual_group")
            if counterfactual_group not in (None, ""):
                metadata["counterfactual_group"] = (
                    f"shard_{shard_index}_{counterfactual_group}"
                )
            records.append(
                {
                    "row": normalized,
                    "shard": shard_index,
                    "line_number": source["line_number"],
                    "original_id": original_id,
                }
            )

    plans_by_input: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for record in records:
        plans_by_input[_collision_key(record["row"])].add(_plan_signature(record["row"]))
    collision_keys = {
        key for key, plan_signatures in plans_by_input.items() if len(plan_signatures) > 1
    }
    collision_ids = {
        key: f"ambiguous_input_{index:06d}"
        for index, key in enumerate(sorted(collision_keys), start=1)
    }

    cleaned: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter(item["reason"] for item in failures)
    for record in records:
        row = record["row"]
        key = _collision_key(row)
        if key in collision_keys:
            reason = "same_image_and_instruction_map_to_multiple_gold_plans"
            failures.append(
                {
                    "status": "failed",
                    "reason": reason,
                    "collision_id": collision_ids[key],
                    "shard": record["shard"],
                    "line_number": record["line_number"],
                    "sample_id": record["original_id"],
                    "image_identity": key[0],
                    "instruction": key[1],
                    "gold_plan_actions": list(_plan_signature(row)),
                }
            )
            failure_counts[reason] += 1
            continue
        cleaned.append(row)

    counterfactual_counts = Counter(
        str(row["meta"].get("counterfactual_group"))
        for row in cleaned
        if row["meta"].get("counterfactual_group") not in (None, "")
    )
    incomplete_counterfactual_groups = {
        name: count for name, count in counterfactual_counts.items() if count != 2
    }
    _write_jsonl(output_files[0], cleaned)
    _write_jsonl(output_files[1], failures)
    report = {
        "shard_root": str(shard_root),
        "shard_count": shard_count,
        "input_samples": len(records) + sum(failure_counts.values()) - failure_counts[
            "same_image_and_instruction_map_to_multiple_gold_plans"
        ],
        "clean_samples": len(cleaned),
        "failed_samples": len(failures),
        "failure_counts": dict(sorted(failure_counts.items())),
        "collision_inputs": len(collision_keys),
        "counterfactual_groups": len(counterfactual_counts),
        "incomplete_counterfactual_groups": incomplete_counterfactual_groups,
        "generation_rejections_not_present_in_samples": dict(sorted(rejection_counts.items())),
        "source_reports": source_reports,
        "samples_path": str(output_files[0]),
        "samples_sha256": _sha256(output_files[0]),
    }
    _write_json(output_files[2], report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge generated shards while excluding failed and ambiguous samples"
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = clean_raw_shards(
        args.shard_root,
        args.output_dir,
        args.shards,
        bool(args.overwrite),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
