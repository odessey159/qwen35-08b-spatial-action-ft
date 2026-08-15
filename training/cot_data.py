from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from exp0.generate_cot_data import (
    extract_task_relevant_state,
    parse_response_sections,
    render_response,
)
from exp0.prompts import system_prompt
from exp0.subgoal_abstraction import (
    abstract_subgoals,
    parse_primitive_action,
    validate_subgoal_abstraction,
)

from .common import TrainingConfigError, require_mapping, resolve_path
from .data import _choose_validation_indices, _read_jsonl, _write_json, _write_jsonl


FORMATS = {"action", "plan_action", "cot"}
SOURCE_FORMATS = {"cot", "raw_simulator"}


@dataclass(frozen=True)
class CotSample:
    sample_id: str
    image_path: Path
    instruction: str
    state: tuple[str, ...]
    plan: tuple[str, ...]
    actions: tuple[str, ...]
    metadata: dict[str, Any]


def uses_embodied_format(config: dict[str, Any]) -> bool:
    data = config.get("data")
    return isinstance(data, dict) and data.get("response_format") is not None


def _source_format(data: dict[str, Any]) -> str:
    value = data.get("source_format", "cot")
    if not isinstance(value, str) or value not in SOURCE_FORMATS:
        raise TrainingConfigError(
            f"data.source_format must be one of {sorted(SOURCE_FORMATS)}"
        )
    return value


def _expected_source_samples(data: dict[str, Any]) -> int | None:
    value = data.get("expected_source_samples")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingConfigError("data.expected_source_samples must be a positive integer")
    return value


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_image_metadata(
    row: dict[str, Any], index: int, source_dir: Path
) -> tuple[str, Path, str, dict[str, Any]]:
    sample_id = str(row.get("sample_id") or f"row_{index:07d}")
    image_value = row.get("image")
    instruction = row.get("instruction", row.get("prompt"))
    if not isinstance(image_value, str) or not image_value.strip():
        raise TrainingConfigError(f"{sample_id}: image is required")
    if not isinstance(instruction, str) or not instruction.strip():
        raise TrainingConfigError(f"{sample_id}: instruction is required")
    image = Path(image_value).expanduser()
    if not image.is_absolute():
        image = source_dir / image
    image = image.resolve()
    if not image.is_file():
        raise TrainingConfigError(f"{sample_id}: image does not exist: {image}")
    metadata_value = row.get("_meta", row.get("meta", row.get("metadata", {})))
    if not isinstance(metadata_value, dict):
        raise TrainingConfigError(f"{sample_id}: metadata must be an object")
    return sample_id, image, instruction.strip(), dict(metadata_value)


def _validate_simulator_metadata(
    sample_id: str, metadata: dict[str, Any], require_sim_verified: bool
) -> None:
    is_verified = metadata.get("sim_verified") is True or metadata.get("verified") is True
    if require_sim_verified and not is_verified:
        raise TrainingConfigError(f"{sample_id}: simulator verification is required")
    if metadata.get("target_visible") is False:
        raise TrainingConfigError(f"{sample_id}: target_visible must not be false")


def _text_list(value: Any, name: str, sample_id: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise TrainingConfigError(f"{sample_id}: {name} must be a string list")


def _sections(row: dict[str, Any], sample_id: str) -> dict[str, list[str]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return {"state": [], "plan": [], "action": []}
    assistants = [
        turn
        for turn in conversations
        if isinstance(turn, dict) and turn.get("role") == "assistant"
    ]
    if len(assistants) != 1 or not isinstance(assistants[0].get("content"), str):
        raise TrainingConfigError(f"{sample_id}: expected one assistant text turn")
    return parse_response_sections(assistants[0]["content"])


def _normalize(
    row: dict[str, Any],
    index: int,
    source_dir: Path,
    verified: bool,
    response_format: str,
) -> CotSample:
    sample_id, image, instruction, metadata = _identity_image_metadata(
        row, index, source_dir
    )

    sections = _sections(row, sample_id)
    oracle = row.get("oracle") if isinstance(row.get("oracle"), dict) else {}
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else row
    actions_value = gold.get("plan_actions", oracle.get("actions", sections["action"]))
    if not isinstance(actions_value, list) or not actions_value:
        raise TrainingConfigError(f"{sample_id}: actions are required")
    actions = [str(value).strip() for value in actions_value]
    for action in actions:
        try:
            parse_primitive_action(action)
        except ValueError as exc:
            raise TrainingConfigError(f"{sample_id}: {exc}") from exc
    state = _text_list(
        gold.get("spatial_state", oracle.get("state", sections["state"])),
        "state",
        sample_id,
    )
    plan = _text_list(
        gold.get("subgoals", oracle.get("plan", sections["plan"])),
        "plan",
        sample_id,
    ) or abstract_subgoals(actions)
    if sections["action"] and sections["action"] != actions:
        raise TrainingConfigError(f"{sample_id}: assistant action differs from oracle")
    if response_format == "cot" and sections["state"] and sections["state"] != state:
        raise TrainingConfigError(f"{sample_id}: assistant state differs from oracle")
    if response_format != "action" and sections["plan"] and sections["plan"] != plan:
        raise TrainingConfigError(f"{sample_id}: assistant plan differs from oracle")
    if response_format != "action":
        try:
            validate_subgoal_abstraction(actions, plan)
        except ValueError as exc:
            raise TrainingConfigError(f"{sample_id}: {exc}") from exc
    _validate_simulator_metadata(sample_id, metadata, verified)
    return CotSample(
        sample_id,
        image,
        instruction,
        tuple(state),
        tuple(plan),
        tuple(actions),
        metadata,
    )


def _normalize_raw_simulator_sample(
    row: dict[str, Any],
    index: int,
    source_dir: Path,
    require_sim_verified: bool,
    max_state_facts: int = 12,
) -> CotSample:
    """Adapt one generator output row without changing its primitive trajectory."""
    sample_id, image, instruction, metadata = _identity_image_metadata(
        row, index, source_dir
    )
    gold = row.get("gold")
    if not isinstance(gold, dict):
        raise TrainingConfigError(f"{sample_id}: gold must be an object")
    actions_value = gold.get("plan_actions")
    if (
        not isinstance(actions_value, list)
        or not actions_value
        or any(not isinstance(action, str) or not action.strip() for action in actions_value)
    ):
        raise TrainingConfigError(
            f"{sample_id}: gold.plan_actions must be a non-empty string list"
        )
    actions = list(actions_value)
    try:
        for action in actions:
            parse_primitive_action(action)
        state = extract_task_relevant_state(row, max_facts=max_state_facts)
        plan = abstract_subgoals(actions)
    except ValueError as exc:
        raise TrainingConfigError(f"{sample_id}: {exc}") from exc
    _validate_simulator_metadata(sample_id, metadata, require_sim_verified)
    return CotSample(
        sample_id=sample_id,
        image_path=image,
        instruction=instruction,
        state=tuple(state),
        plan=tuple(plan),
        actions=tuple(actions),
        metadata=metadata,
    )


def _format_weights(config: dict[str, Any]) -> tuple[str, dict[str, float]]:
    data = require_mapping(config.get("data"), "data")
    response_format = str(data.get("response_format"))
    if response_format not in FORMATS:
        raise TrainingConfigError(f"data.response_format must be one of {sorted(FORMATS)}")
    training = require_mapping(config.get("training"), "training")
    raw = training.get("section_loss_weights")
    if raw is None:
        return response_format, {}
    if not isinstance(raw, dict):
        raise TrainingConfigError("section_loss_weights must be an object")
    required = {
        "action": {"action"},
        "plan_action": {"plan", "action"},
        "cot": {"state", "plan", "action"},
    }[response_format]
    if set(raw) != required:
        raise TrainingConfigError(f"section_loss_weights keys must be {sorted(required)}")
    try:
        weights = {name: float(value) for name, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise TrainingConfigError("section_loss_weights must be numeric") from exc
    if any(value <= 0 for value in weights.values()) or abs(sum(weights.values()) - 1) > 1e-6:
        raise TrainingConfigError("section_loss_weights must be positive and sum to 1.0")
    return response_format, weights


def _system(allowed: list[str], response_format: str) -> str:
    if response_format == "action":
        return system_prompt(allowed, inline_example=True, response_format="action")
    sections = "<state>、<plan>、<action>" if response_format == "cot" else "<plan>、<action>"
    return (
        "你是一个室内家务动作规划助手。\n"
        f"回答必须且只能依次包含 {sections}，不要添加其他段落。\n"
        + ("<state> 只写 simulator-verified 的任务相关空间事实。\n" if response_format == "cot" else "")
        + "<plan> 写高层子目标，不要逐条改写底层动作。\n"
        "<action> 每行写一个 动作名(物体名)。动作名只能从以下集合选择："
        + "、".join(allowed)
        + "。"
    )


def _swift_row(
    sample: CotSample, allowed: list[str], response_format: str, weights: dict[str, float]
) -> dict[str, Any]:
    variant = {"action": "action-only", "plan_action": "plan-only", "cot": "cot"}[response_format]
    sections = parse_response_sections(render_response(sample.state, sample.plan, sample.actions, variant))
    pieces: list[tuple[str, str]] = []
    if response_format == "cot":
        pieces.append(("state", "<state>\n" + "\n".join(sections["state"]) + "\n</state>\n\n"))
    if response_format in {"cot", "plan_action"}:
        plan = [f"{index}. {text}" for index, text in enumerate(sections["plan"], start=1)]
        pieces.append(("plan", "<plan>\n" + "\n".join(plan) + "\n</plan>\n\n"))
    pieces.append(("action", "<action>\n" + "\n".join(sections["action"]) + "\n</action>"))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system(allowed, response_format)},
        {"role": "user", "content": f"<image>\n目标指令：{sample.instruction}"},
    ]
    if weights:
        messages.extend(
            {
                "role": "assistant",
                "content": text,
                "loss_scale": weights[name],
            }
            for name, text in pieces
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": "".join(text for _, text in pieces),
            }
        )
    return {
        "messages": messages,
        "images": [str(sample.image_path)],
    }


def prepare_cot_dataset(config: dict[str, Any], base_dir: Path, overwrite: bool) -> dict[str, Any]:
    data = require_mapping(config.get("data"), "data")
    response_format, weights = _format_weights(config)
    source_format = _source_format(data)
    source = resolve_path(base_dir, str(data.get("source_dataset", "")))
    if not source.is_file():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    output = resolve_path(base_dir, str(data.get("prepared_dir", "prepared")))
    output.mkdir(parents=True, exist_ok=True)
    train_path, val_path, manifest_path = output / "train.jsonl", output / "val.jsonl", output / "manifest.json"
    if not overwrite and any(path.exists() for path in (train_path, val_path, manifest_path)):
        raise FileExistsError("Prepared files already exist (use --overwrite)")
    rows = _read_jsonl(source)
    expected_source_samples = _expected_source_samples(data)
    if expected_source_samples is not None and len(rows) != expected_source_samples:
        raise TrainingConfigError(
            "Source sample count mismatch: "
            f"expected {expected_source_samples}, found {len(rows)} in {source}"
        )
    require_sim_verified = bool(data.get("require_sim_verified", True))
    if source_format == "raw_simulator":
        max_state_facts_value = data.get("max_state_facts", 12)
        if (
            isinstance(max_state_facts_value, bool)
            or not isinstance(max_state_facts_value, int)
            or max_state_facts_value <= 0
        ):
            raise TrainingConfigError("data.max_state_facts must be a positive integer")
        samples = [
            _normalize_raw_simulator_sample(
                row,
                index,
                source.parent,
                require_sim_verified,
                max_state_facts_value,
            )
            for index, row in enumerate(rows, start=1)
        ]
    else:
        samples = [
            _normalize(
                row,
                index,
                source.parent,
                require_sim_verified,
                response_format,
            )
            for index, row in enumerate(rows, start=1)
        ]
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise TrainingConfigError("sample_id values must be unique")
    allowed = [str(value) for value in config.get("allowed_actions", [])]
    if not allowed:
        raise TrainingConfigError("allowed_actions must be non-empty")
    allowed_set = set(allowed)
    for sample in samples:
        unknown = {parse_primitive_action(action).name for action in sample.actions} - allowed_set
        if unknown:
            raise TrainingConfigError(f"{sample.sample_id}: unknown actions {sorted(unknown)}")
        if response_format == "cot" and not sample.state:
            raise TrainingConfigError(f"{sample.sample_id}: CoT requires state facts")
    fields = data.get("split_group_fields", ["scene_id", "counterfactual_group"])
    if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
        raise TrainingConfigError("split_group_fields must be a string list")
    seed = int(data.get("seed", 42))
    ratio = float(data.get("validation_ratio", 0.1))
    val_indices = _choose_validation_indices(samples, ratio, fields, seed)
    train_indices = [index for index in range(len(samples)) if index not in val_indices]
    ordered_val = [index for index in range(len(samples)) if index in val_indices]
    if not train_indices:
        raise TrainingConfigError("Training split is empty")
    _write_jsonl(train_path, (_swift_row(samples[index], allowed, response_format, weights) for index in train_indices))
    _write_jsonl(val_path, (_swift_row(samples[index], allowed, response_format, weights) for index in ordered_val))
    manifest = {
        "source_dataset": str(source),
        "source_sha256": _source_sha256(source),
        "source_format": source_format,
        "response_format": response_format,
        "section_loss_weights": weights,
        "seed": seed,
        "validation_ratio_requested": ratio,
        "split_group_fields": fields,
        "total_samples": len(samples),
        "train_samples": len(train_indices),
        "validation_samples": len(ordered_val),
        "train_sample_ids": [samples[index].sample_id for index in train_indices],
        "validation_sample_ids": [samples[index].sample_id for index in ordered_val],
    }
    _write_json(manifest_path, manifest)
    return manifest


def _validate_prepared_source(
    data: dict[str, Any], base_dir: Path, output: Path
) -> None:
    source = resolve_path(base_dir, str(data.get("source_dataset", "")))
    if not source.is_file():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise TrainingConfigError(f"Invalid prepared manifest: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TrainingConfigError(f"Prepared manifest must be an object: {manifest_path}")
    recorded_sha = manifest.get("source_sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha:
        raise TrainingConfigError(
            f"Prepared manifest has no source_sha256: {manifest_path}; run prepare --overwrite"
        )
    current_sha = _source_sha256(source)
    if current_sha != recorded_sha:
        raise TrainingConfigError(
            "Prepared data is stale because the source dataset changed; "
            "run prepare --overwrite before validate or train"
        )


def validate_cot_prepared_dataset(config: dict[str, Any], base_dir: Path) -> dict[str, int]:
    data = require_mapping(config.get("data"), "data")
    _source_format(data)
    response_format, weights = _format_weights(config)
    output = resolve_path(base_dir, str(data.get("prepared_dir", "prepared")))
    _validate_prepared_source(data, base_dir, output)
    required = {
        "action": ["action"],
        "plan_action": ["plan", "action"],
        "cot": ["state", "plan", "action"],
    }[response_format]
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        path = output / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared split does not exist: {path}")
        rows = _read_jsonl(path) if path.stat().st_size else []
        for index, row in enumerate(rows, start=1):
            messages, images = row.get("messages"), row.get("images")
            expected_message_count = 2 + (len(required) if weights else 1)
            if not isinstance(messages, list) or len(messages) != expected_message_count:
                raise TrainingConfigError(
                    f"{path}:{index}: expected {expected_message_count} messages"
                )
            if not isinstance(images, list) or len(images) != 1 or not Path(str(images[0])).is_file():
                raise TrainingConfigError(f"{path}:{index}: invalid image")
            expected_roles = ["system", "user"] + ["assistant"] * (len(messages) - 2)
            if [message.get("role") for message in messages] != expected_roles:
                raise TrainingConfigError(f"{path}:{index}: invalid roles")
            assistant_messages = messages[2:]
            if any(
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in assistant_messages
            ):
                raise TrainingConfigError(f"{path}:{index}: assistant content must be text")
            content = "".join(message["content"] for message in assistant_messages)
            sections = parse_response_sections(content)
            positions = []
            for name in required:
                if not sections[name] or f"</{name}>" not in content:
                    raise TrainingConfigError(f"{path}:{index}: missing <{name}>")
                positions.append(content.index(f"<{name}>"))
            if positions != sorted(positions):
                raise TrainingConfigError(f"{path}:{index}: invalid section order")
            unexpected = {"state", "plan", "action"} - set(required)
            if any(sections[name] for name in unexpected):
                raise TrainingConfigError(f"{path}:{index}: unexpected response section")
            try:
                for action in sections["action"]:
                    parse_primitive_action(action)
                if sections["plan"]:
                    validate_subgoal_abstraction(sections["action"], sections["plan"])
            except ValueError as exc:
                raise TrainingConfigError(f"{path}:{index}: {exc}") from exc
            if weights:
                expected = [weights[name] for name in required]
                scales = [message.get("loss_scale") for message in assistant_messages]
                try:
                    normalized_scales = [float(value) for value in scales]
                except (TypeError, ValueError) as exc:
                    raise TrainingConfigError(f"{path}:{index}: invalid loss_scale") from exc
                if normalized_scales != expected:
                    raise TrainingConfigError(f"{path}:{index}: invalid loss_scale")
            elif any("loss_scale" in message for message in assistant_messages):
                raise TrainingConfigError(f"{path}:{index}: unexpected loss_scale")
        counts[split] = len(rows)
    if counts["train"] == 0:
        raise TrainingConfigError("Prepared training split is empty")
    return counts
