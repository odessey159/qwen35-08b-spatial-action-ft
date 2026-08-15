from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import TrainingConfigError, require_mapping, resolve_path


ACTION_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\([^\n()]*\)$")


@dataclass(frozen=True)
class NormalizedSample:
    sample_id: str
    image_path: Path
    instruction: str
    plan_actions: tuple[str, ...]
    plan_nl: str
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TrainingConfigError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TrainingConfigError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise TrainingConfigError(f"Dataset is empty: {path}")
    return rows


def _nonempty_text(value: Any, field: str, sample_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigError(f"{sample_id}: '{field}' must be a non-empty string")
    return value.strip()


def _normalize_sample(
    row: dict[str, Any],
    index: int,
    source_dir: Path,
    require_sim_verified: bool,
) -> NormalizedSample:
    sample_id = str(row.get("sample_id") or f"row_{index:07d}")
    image_value = _nonempty_text(row.get("image"), "image", sample_id)
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = source_dir / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise TrainingConfigError(f"{sample_id}: image does not exist: {image_path}")

    instruction = _nonempty_text(
        row.get("instruction", row.get("prompt")), "instruction/prompt", sample_id
    )
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else row
    actions_value = gold.get("plan_actions")
    if not isinstance(actions_value, list) or not actions_value:
        raise TrainingConfigError(f"{sample_id}: plan_actions must be a non-empty list")
    actions: list[str] = []
    for action in actions_value:
        normalized = _nonempty_text(action, "plan_actions[]", sample_id)
        if not ACTION_PATTERN.fullmatch(normalized):
            raise TrainingConfigError(f"{sample_id}: invalid action syntax: {normalized}")
        actions.append(normalized)
    plan_nl = _nonempty_text(gold.get("plan_nl"), "plan_nl", sample_id)

    metadata_value = row.get("_meta", row.get("meta", {}))
    if not isinstance(metadata_value, dict):
        raise TrainingConfigError(f"{sample_id}: '_meta/meta' must be a JSON object")
    metadata = dict(metadata_value)
    if require_sim_verified and metadata.get("sim_verified") is not True:
        raise TrainingConfigError(f"{sample_id}: sim_verified must be true")
    if metadata.get("target_visible") is False:
        raise TrainingConfigError(f"{sample_id}: target_visible must not be false")

    return NormalizedSample(
        sample_id=sample_id,
        image_path=image_path,
        instruction=instruction,
        plan_actions=tuple(actions),
        plan_nl=plan_nl,
        metadata=metadata,
    )


def _output_contract(allowed_actions: Iterable[str]) -> str:
    action_text = " ".join(allowed_actions)
    return f"""可用动作仅限：{action_text}
请严格输出以下两部分，不要输出分析过程：
<plan>
ActionName(Object)
...</plan>
随后用一句自然语言描述相同步骤。
动作序列必须放在最前面，每行一个动作。"""


def _swift_row(sample: NormalizedSample, allowed_actions: list[str]) -> dict[str, Any]:
    prompt = (
        f"<image>\n目标指令：{sample.instruction}\n\n"
        f"请根据图像生成计划。\n\n{_output_contract(allowed_actions)}"
    )
    response = "<plan>\n" + "\n".join(sample.plan_actions) + f"\n</plan>\n{sample.plan_nl}"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "images": [str(sample.image_path)],
    }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _split_components(
    samples: list[NormalizedSample], group_fields: list[str]
) -> list[list[int]]:
    groups = _DisjointSet(len(samples))
    first_seen: dict[tuple[str, str], int] = {}
    for index, sample in enumerate(samples):
        for field in group_fields:
            value = sample.metadata.get(field)
            if value is None or value == "":
                continue
            key = (field, json.dumps(value, ensure_ascii=False, sort_keys=True))
            if key in first_seen:
                groups.union(index, first_seen[key])
            else:
                first_seen[key] = index
    components: dict[int, list[int]] = {}
    for index in range(len(samples)):
        components.setdefault(groups.find(index), []).append(index)
    return list(components.values())


def _choose_validation_indices(
    samples: list[NormalizedSample],
    validation_ratio: float,
    group_fields: list[str],
    seed: int,
) -> set[int]:
    if not 0 <= validation_ratio < 1:
        raise TrainingConfigError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0:
        return set()
    components = _split_components(samples, group_fields)
    if len(components) < 2:
        raise TrainingConfigError(
            "A non-zero validation split needs at least two independent scene/counterfactual groups"
        )
    rng = random.Random(seed)
    rng.shuffle(components)
    target = max(1, round(len(samples) * validation_ratio))
    selected: list[list[int]] = []
    selected_count = 0
    for component in components:
        without = abs(target - selected_count)
        with_component = abs(target - (selected_count + len(component)))
        if with_component < without:
            selected.append(component)
            selected_count += len(component)
    if not selected:
        selected = [min(components, key=lambda component: abs(target - len(component)))]
    if len(selected) == len(components):
        selected.remove(
            min(
                selected,
                key=lambda component: abs(target - (selected_count - len(component))),
            )
        )
    return {index for component in selected for index in component}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def prepare_dataset(config: dict[str, Any], base_dir: Path, overwrite: bool) -> dict[str, Any]:
    data_config = require_mapping(config.get("data"), "data")
    source_path = resolve_path(base_dir, str(data_config.get("source_dataset", "")))
    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset does not exist: {source_path}")
    output_dir = resolve_path(base_dir, str(data_config.get("prepared_dir", "prepared")))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [path for path in (train_path, val_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Prepared files already exist (use --overwrite): {names}")

    allowed_actions_value = config.get("allowed_actions")
    if not isinstance(allowed_actions_value, list) or not allowed_actions_value:
        raise TrainingConfigError("'allowed_actions' must be a non-empty list")
    allowed_actions = [str(value) for value in allowed_actions_value]
    allowed_set = set(allowed_actions)
    source_rows = _read_jsonl(source_path)
    samples = [
        _normalize_sample(
            row,
            index,
            source_path.parent,
            bool(data_config.get("require_sim_verified", True)),
        )
        for index, row in enumerate(source_rows, start=1)
    ]
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise TrainingConfigError("sample_id values must be unique")
    for sample in samples:
        unknown = {
            action.split("(", 1)[0]
            for action in sample.plan_actions
            if action.split("(", 1)[0] not in allowed_set
        }
        if unknown:
            raise TrainingConfigError(
                f"{sample.sample_id}: actions outside allowed_actions: {sorted(unknown)}"
            )

    group_fields_value = data_config.get(
        "split_group_fields", ["scene_id", "counterfactual_group"]
    )
    if not isinstance(group_fields_value, list) or not all(
        isinstance(value, str) and value for value in group_fields_value
    ):
        raise TrainingConfigError("split_group_fields must be a list of non-empty strings")
    seed = int(data_config.get("seed", 42))
    validation_ratio = float(data_config.get("validation_ratio", 0.1))
    val_indices = _choose_validation_indices(
        samples, validation_ratio, list(group_fields_value), seed
    )
    train_indices = [index for index in range(len(samples)) if index not in val_indices]
    ordered_val_indices = [index for index in range(len(samples)) if index in val_indices]
    if not train_indices:
        raise TrainingConfigError("Training split is empty")

    _write_jsonl(train_path, (_swift_row(samples[index], allowed_actions) for index in train_indices))
    _write_jsonl(
        val_path, (_swift_row(samples[index], allowed_actions) for index in ordered_val_indices)
    )
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest = {
        "source_dataset": str(source_path),
        "source_sha256": source_hash,
        "seed": seed,
        "validation_ratio_requested": validation_ratio,
        "split_group_fields": group_fields_value,
        "total_samples": len(samples),
        "train_samples": len(train_indices),
        "validation_samples": len(ordered_val_indices),
        "train_sample_ids": [samples[index].sample_id for index in train_indices],
        "validation_sample_ids": [samples[index].sample_id for index in ordered_val_indices],
    }
    _write_json(manifest_path, manifest)
    return manifest


def validate_prepared_dataset(config: dict[str, Any], base_dir: Path) -> dict[str, int]:
    data_config = require_mapping(config.get("data"), "data")
    output_dir = resolve_path(base_dir, str(data_config.get("prepared_dir", "prepared")))
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        path = output_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared split does not exist: {path}")
        rows = _read_jsonl(path) if path.stat().st_size else []
        for index, row in enumerate(rows, start=1):
            messages = row.get("messages")
            images = row.get("images")
            if not isinstance(messages, list) or len(messages) != 2:
                raise TrainingConfigError(f"{path}:{index}: expected exactly two messages")
            if not isinstance(images, list) or len(images) != 1:
                raise TrainingConfigError(f"{path}:{index}: expected exactly one image")
            if not Path(str(images[0])).is_file():
                raise TrainingConfigError(f"{path}:{index}: image does not exist: {images[0]}")
            if messages[0].get("role") != "user" or "<image>" not in str(
                messages[0].get("content", "")
            ):
                raise TrainingConfigError(f"{path}:{index}: invalid multimodal user message")
            if messages[1].get("role") != "assistant" or "<plan>" not in str(
                messages[1].get("content", "")
            ):
                raise TrainingConfigError(f"{path}:{index}: invalid assistant response")
        counts[split] = len(rows)
    if counts["train"] == 0:
        raise TrainingConfigError("Prepared training split is empty")
    return counts
