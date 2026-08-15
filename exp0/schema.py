from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
PLAN_BLOCK_PATTERN = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.IGNORECASE | re.DOTALL)
SUMMARY_BLOCK_PATTERN = re.compile(
    r"<summary>\s*(.*?)\s*(?:</summary>|$)", re.IGNORECASE | re.DOTALL
)

# `PickupObject Plate` instead of `PickupObject(Plate)` accounted for 99.3% of all
# plan lines in the first run. The lenient reader accepts that form, plus list
# markers, code fences and casing drift, and drops anything whose head token is
# not in the closed action vocabulary.
SPACE_ACTION_PATTERN = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*)[\s:：,，-]+(.+?)[。.]?$"
)
LEADING_MARKER_PATTERN = re.compile(r"^(?:[-*+•]|\d+\s*[.、)])\s*")

# Chinese surface forms accepted for each action when scoring the <summary> text.
# Note ToggleObject and OpenObject genuinely overlap on 打开/开启 in Chinese; the
# ordered matcher below resolves most cases by position, and `nl_order_ok` will
# expose the rest.
ACTION_NL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "GotoLocation": ("前往", "走到", "走向", "来到", "去到", "移动到", "到达", "靠近", "走过去"),
    "PickupObject": ("拿起", "拿出", "取出", "拿走", "捡起", "抓起", "取下", "拿到"),
    "PutObject": ("放入", "放到", "放进", "放置", "放在", "放上", "装进", "放好"),
    "SliceObject": ("切开", "切成", "切片", "切下", "切"),
    "CleanObject": ("清洁", "清洗", "洗干净", "洗净", "冲洗", "擦干净", "擦", "洗"),
    "HeatObject": ("加热", "热一下", "微波", "加温", "热"),
    "ToggleObject": ("切换", "开关", "打开", "开启", "点亮", "关掉", "按下"),
    "OpenObject": ("打开", "拉开", "开启", "掀开", "开"),
    "CloseObject": ("关闭", "关上", "合上", "关好", "盖上", "关"),
}


@dataclass(frozen=True)
class ParsedPlan:
    actions: list[str]
    action_names: list[str]
    arguments: list[list[str]]
    has_plan_tags: bool
    all_lines_parseable: bool

    @property
    def structure_valid(self) -> bool:
        return self.has_plan_tags and self.all_lines_parseable and bool(self.actions)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_action(action: str) -> str:
    match = ACTION_PATTERN.match(action.strip())
    if not match:
        return " ".join(action.strip().split())
    name, raw_args = match.groups()
    args = [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    return f"{name}({','.join(args)})"


def parse_action(action: str) -> tuple[str, list[str]] | None:
    match = ACTION_PATTERN.match(action.strip())
    if not match:
        return None
    name, raw_args = match.groups()
    args = [arg.strip() for arg in raw_args.split(",") if arg.strip()]
    return name, args


def parse_plan(text: str) -> ParsedPlan:
    block_match = PLAN_BLOCK_PATTERN.search(text)
    has_tags = block_match is not None
    block = block_match.group(1) if block_match else text
    candidate_lines = [line.strip() for line in block.splitlines() if line.strip()]

    actions: list[str] = []
    names: list[str] = []
    arguments: list[list[str]] = []
    all_parseable = bool(candidate_lines)
    for line in candidate_lines:
        parsed = parse_action(line)
        if parsed is None:
            all_parseable = False
            continue
        name, args = parsed
        actions.append(normalize_action(line))
        names.append(name)
        arguments.append(args)

    return ParsedPlan(
        actions=actions,
        action_names=names,
        arguments=arguments,
        has_plan_tags=has_tags,
        all_lines_parseable=all_parseable,
    )


def parse_plan_lenient(text: str, allowed_actions: list[str] | set[str]) -> list[str]:
    """Best-effort action list. Vocabulary-anchored, so it cannot invent actions."""
    block_match = PLAN_BLOCK_PATTERN.search(text)
    block = block_match.group(1) if block_match else text
    canonical = {name.casefold(): name for name in allowed_actions}

    actions: list[str] = []
    for raw_line in block.splitlines():
        line = LEADING_MARKER_PATTERN.sub("", raw_line.strip()).strip("`* ")
        if not line:
            continue
        strict = parse_action(line)
        if strict is not None:
            raw_name, args = strict
        else:
            loose = SPACE_ACTION_PATTERN.match(line)
            if loose is None:
                continue
            raw_name = loose.group(1)
            args = [
                token.strip(" ()")
                for token in re.split(r"[,，、]", loose.group(2))
                if token.strip(" ()")
            ]
        name = canonical.get(raw_name.casefold())
        if name is None:
            continue
        actions.append(f"{name}({','.join(args)})")
    return actions


def extract_summary(text: str) -> str:
    """Pull the natural-language plan out of a raw generation.

    Prefers an explicit <summary> block. Falls back to whatever follows the plan
    block, with echoed instruction lines removed -- 91% of the first run's
    outputs contained a verbatim copy of a contract sentence, which would
    otherwise be scored as if it were the model's answer.
    """
    summary_match = SUMMARY_BLOCK_PATTERN.search(text)
    if summary_match is not None:
        candidate = summary_match.group(1)
    else:
        plan_match = PLAN_BLOCK_PATTERN.search(text)
        candidate = text[plan_match.end() :] if plan_match else text

    echo_markers = (
        "随后用一句自然语言",
        "动作序列必须放在最前面",
        "请严格输出",
        "可用动作仅限",
        "每行一个动作",
        "不要输出",
        "回答必须",
        "格式示例",
    )
    kept = [
        line.strip()
        for line in candidate.splitlines()
        if line.strip()
        and not line.strip().startswith("<")
        and not any(marker in line for marker in echo_markers)
    ]
    return " ".join(kept).strip()


def _char_ngrams(text: str, size: int = 2) -> Counter[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < size:
        return Counter([compact] if compact else [])
    return Counter(compact[i : i + size] for i in range(len(compact) - size + 1))


def char_ngram_f1(predicted: str, gold: str) -> float:
    pred_grams, gold_grams = _char_ngrams(predicted), _char_ngrams(gold)
    if not pred_grams or not gold_grams:
        return 0.0
    overlap = sum((pred_grams & gold_grams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_grams.values())
    recall = overlap / sum(gold_grams.values())
    return 2 * precision * recall / (precision + recall)


def score_summary(
    predicted_nl: str, gold_actions: list[str], gold_nl: str
) -> dict[str, float]:
    """Deterministic scoring of the natural-language plan.

    No embedding model or judge required, so this runs anywhere. It checks the
    three things that make an NL plan correct: every gold step is mentioned,
    in the right order, naming the right objects. `nl_char_f1` is a soft
    similarity kept alongside for ranking near-misses; feed `pred_nl` from
    scored_predictions.json to an LLM judge later if a softer score is wanted.
    """
    if not predicted_nl:
        return {
            "nl_present": 0.0,
            "nl_action_recall": 0.0,
            "nl_object_recall": 0.0,
            "nl_order_ok": 0.0,
            "nl_plan_match": 0.0,
            "nl_char_f1": 0.0,
        }

    names: list[str] = []
    objects: list[str] = []
    for action in gold_actions:
        parsed = parse_action(action)
        if parsed is None:
            continue
        name, args = parsed
        names.append(name)
        objects.extend(args)

    cursor = 0
    matched = 0
    ordered = True
    for name in names:
        keywords = ACTION_NL_KEYWORDS.get(name, ())
        positions = [
            predicted_nl.find(keyword, cursor)
            for keyword in keywords
            if predicted_nl.find(keyword, cursor) >= 0
        ]
        if positions:
            matched += 1
            cursor = min(positions) + 1
        else:
            ordered = False

    lowered = predicted_nl.casefold()
    unique_objects = list(dict.fromkeys(objects))
    object_hits = sum(1 for obj in unique_objects if obj.casefold() in lowered)
    object_recall = object_hits / len(unique_objects) if unique_objects else 1.0
    action_recall = matched / len(names) if names else 1.0
    order_ok = bool(ordered and names and matched == len(names))

    return {
        "nl_present": 1.0,
        "nl_action_recall": action_recall,
        "nl_object_recall": object_recall,
        "nl_order_ok": float(order_ok),
        "nl_plan_match": float(order_ok and object_recall == 1.0),
        "nl_char_f1": char_ngram_f1(predicted_nl, gold_nl),
    }


def _require_type(
    sample_id: str,
    row: dict[str, Any],
    key: str,
    expected_type: type,
    errors: list[str],
) -> Any:
    value = row.get(key)
    if not isinstance(value, expected_type):
        errors.append(f"{sample_id}: '{key}' must be {expected_type.__name__}")
    return value


def validate_samples(
    samples: list[dict[str, Any]],
    dataset_dir: Path,
    allowed_actions: set[str],
    allowed_objects: set[str],
    min_samples: int,
    max_samples: int,
) -> list[str]:
    errors: list[str] = []
    if not min_samples <= len(samples) <= max_samples:
        errors.append(
            f"Dataset must contain {min_samples}-{max_samples} rows; found {len(samples)}"
        )

    seen_ids: set[str] = set()
    for index, row in enumerate(samples, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            sample_id = f"row_{index}"
            errors.append(f"{sample_id}: missing non-empty 'sample_id'")
        elif sample_id in seen_ids:
            errors.append(f"{sample_id}: duplicate sample_id")
        seen_ids.add(sample_id)

        for key in ("image", "wrong_image", "instruction"):
            value = _require_type(sample_id, row, key, str, errors)
            if key in {"image", "wrong_image"} and isinstance(value, str):
                image_path = (dataset_dir / value).resolve()
                if not image_path.is_file():
                    errors.append(f"{sample_id}: image does not exist: {image_path}")

        scene_graph = _require_type(sample_id, row, "scene_graph", dict, errors)
        if isinstance(scene_graph, dict):
            if not isinstance(scene_graph.get("objects"), list):
                errors.append(f"{sample_id}: scene_graph.objects must be a list")
            if not isinstance(scene_graph.get("relations"), list):
                errors.append(f"{sample_id}: scene_graph.relations must be a list")

        spatial_facts = _require_type(sample_id, row, "spatial_facts", list, errors)
        subgoals = _require_type(sample_id, row, "subgoals", list, errors)
        if isinstance(spatial_facts, list) and not all(isinstance(x, str) for x in spatial_facts):
            errors.append(f"{sample_id}: every spatial fact must be a string")
        if isinstance(subgoals, list) and not all(isinstance(x, str) for x in subgoals):
            errors.append(f"{sample_id}: every subgoal must be a string")

        gold = _require_type(sample_id, row, "gold", dict, errors)
        gold_actions: list[str] = []
        if isinstance(gold, dict):
            raw_actions = gold.get("plan_actions")
            if not isinstance(raw_actions, list) or not raw_actions:
                errors.append(f"{sample_id}: gold.plan_actions must be a non-empty list")
            elif not all(isinstance(action, str) for action in raw_actions):
                errors.append(f"{sample_id}: every gold action must be a string")
            else:
                gold_actions = raw_actions
                for action in gold_actions:
                    parsed = parse_action(action)
                    if parsed is None:
                        errors.append(f"{sample_id}: invalid gold action syntax: {action}")
                        continue
                    name, args = parsed
                    if name not in allowed_actions:
                        errors.append(f"{sample_id}: action outside vocabulary: {name}")
                    if allowed_objects:
                        for arg in args:
                            if arg not in allowed_objects:
                                errors.append(f"{sample_id}: object outside vocabulary: {arg}")
            if not isinstance(gold.get("plan_nl"), str):
                errors.append(f"{sample_id}: gold.plan_nl must be a string")

        meta = _require_type(sample_id, row, "meta", dict, errors)
        if isinstance(meta, dict):
            if meta.get("sim_verified") is not True:
                errors.append(f"{sample_id}: meta.sim_verified must be true")
            if gold_actions and meta.get("plan_length") != len(gold_actions):
                errors.append(
                    f"{sample_id}: meta.plan_length must equal len(gold.plan_actions)"
                )
    return errors

