from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp0.schema import ACTION_NL_KEYWORDS, parse_action, parse_plan_lenient
from training.audit_eva import gold_actions, metadata
from training.evaluate_in_domain_predictions import ALLOWED_ACTIONS, prediction_text
from training.evaluate_section_losses import load_jsonl


ACTION_ARITY = {
    "GotoLocation": 1,
    "PickupObject": 1,
    "PutObject": 2,
    "SliceObject": 1,
    "CleanObject": 1,
    "HeatObject": 1,
    "ToggleObject": 1,
    "OpenObject": 1,
    "CloseObject": 1,
}
ACTION_ORDER = tuple(sorted(ALLOWED_ACTIONS))
ACTION_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in sorted(ACTION_ORDER, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
CONDITIONAL_PATTERN = re.compile(
    r"(?:如果|若(?:是|果)?|假如|当.+?时|否则|视情况|根据.+?状态|\bif\b|\bwhen\b|\bunless\b|depending\s+on)",
    re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(
    r"(?:无需|不需要|不要|不可|不能|不应|不执行|避免|跳过|\bdo\s+not\b|\bdon't\b|\bskip\b)",
    re.IGNORECASE,
)
LEADING_MARKER_PATTERN = re.compile(r"^(?:[-*+•]|\d+\s*[.、)])\s*")
PLAN_BLOCK_PATTERN = re.compile(
    r"<plan>\s*(.*?)(?:</plan>|<action>|$)", re.IGNORECASE | re.DOTALL
)
ACTION_BLOCK_PATTERN = re.compile(
    r"<action>\s*(.*?)(?:</action>|$)", re.IGNORECASE | re.DOTALL
)


@dataclass(frozen=True)
class RelaxedParse:
    actions: tuple[str, ...]
    relaxed_parseable: bool
    deterministic_plan: bool
    source: str
    reason: str


def _canonical_objects(raw_values: list[str], object_map: dict[str, str]) -> list[str] | None:
    values: list[str] = []
    for raw in raw_values:
        value = raw.strip().strip("()[]{} `*.,，。:：;；")
        value = re.sub(r"_\d+$", "", value)
        canonical = object_map.get(value.casefold())
        if canonical is None:
            return None
        values.append(canonical)
    return values


def _canonical_action(
    name: str,
    raw_arguments: list[str],
    action_map: dict[str, str],
    object_map: dict[str, str],
) -> str | None:
    canonical_name = action_map.get(name.casefold())
    if canonical_name is None:
        return None
    arguments = _canonical_objects(raw_arguments, object_map)
    if arguments is None or len(arguments) != ACTION_ARITY[canonical_name]:
        return None
    return f"{canonical_name}({','.join(arguments)})"


def _object_pattern(allowed_objects: set[str]) -> tuple[re.Pattern[str], dict[str, str]]:
    canonical = {name.casefold(): name for name in allowed_objects}
    names = sorted(allowed_objects, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(name) for name in names) + r")(?:_\d+)?(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern, canonical


def _objects_in_text(
    text: str, object_pattern: re.Pattern[str], object_map: dict[str, str]
) -> list[str]:
    return [object_map[match.group(1).casefold()] for match in object_pattern.finditer(text)]


def _contract_actions(
    block: str,
    action_map: dict[str, str],
    object_map: dict[str, str],
) -> tuple[list[str], bool, bool]:
    actions: list[str] = []
    mentioned = False
    incomplete = False
    for raw_line in block.splitlines():
        line = LEADING_MARKER_PATTERN.sub("", raw_line.strip()).strip("`* ")
        if not line:
            continue
        if CONDITIONAL_PATTERN.search(line):
            if ACTION_NAME_PATTERN.search(line):
                mentioned = True
                incomplete = True
            continue
        recovered = parse_plan_lenient(line, set(ACTION_ORDER))
        if recovered:
            mentioned = True
            for value in recovered:
                parsed = parse_action(value)
                if parsed is None:
                    incomplete = True
                    continue
                canonical = _canonical_action(parsed[0], parsed[1], action_map, object_map)
                if canonical is None:
                    incomplete = True
                else:
                    actions.append(canonical)
        elif ACTION_NAME_PATTERN.search(line):
            mentioned = True
            incomplete = True
    return actions, mentioned, incomplete


def _keyword_occurrences(line: str) -> list[tuple[int, int, str]]:
    occurrences: list[tuple[int, int, str]] = []
    for action_name, keywords in ACTION_NL_KEYWORDS.items():
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), line, re.IGNORECASE):
                if action_name == "GotoLocation" and keyword == "移动到":
                    prefix = line[: match.start()]
                    if not (
                        re.search(r"(?:玩家|Agent|智能体|机器人)\s*$", prefix, re.IGNORECASE)
                        or not re.search(r"将\s*\S+", prefix)
                    ):
                        continue
                occurrences.append((match.start(), match.end(), action_name))
    occurrences.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    deduplicated: list[tuple[int, int, str]] = []
    occupied_end = -1
    for item in occurrences:
        if item[0] < occupied_end:
            continue
        deduplicated.append(item)
        occupied_end = item[1]
    return deduplicated


def _natural_line_actions(
    line: str,
    action_map: dict[str, str],
    object_pattern: re.Pattern[str],
    object_map: dict[str, str],
) -> tuple[list[str], bool, bool]:
    explicit = list(ACTION_NAME_PATTERN.finditer(line))
    occurrences: list[tuple[int, int, str]]
    if explicit:
        occurrences = [
            (match.start(), match.end(), action_map[match.group(1).casefold()])
            for match in explicit
        ]
    else:
        occurrences = _keyword_occurrences(line)
    if not occurrences:
        return [], False, False
    if CONDITIONAL_PATTERN.search(line) or NEGATION_PATTERN.search(line):
        return [], True, True

    actions: list[str] = []
    incomplete = False
    for index, (start, end, action_name) in enumerate(occurrences):
        segment_end = occurrences[index + 1][0] if index + 1 < len(occurrences) else len(line)
        segment = line[end:segment_end]
        arguments: list[str]
        if explicit:
            parenthesized = re.match(r"\s*\(([^)]*)\)", segment)
            if parenthesized is not None:
                arguments = [
                    value.strip()
                    for value in re.split(r"[,，、]", parenthesized.group(1))
                    if value.strip()
                ]
            else:
                arguments = _objects_in_text(segment, object_pattern, object_map)
        else:
            arguments = (
                _objects_in_text(line, object_pattern, object_map)
                if len(occurrences) == 1
                else _objects_in_text(segment, object_pattern, object_map)
            )
        canonical = _canonical_action(action_name, arguments, action_map, object_map)
        if canonical is None:
            incomplete = True
        else:
            actions.append(canonical)
    return actions, True, incomplete


def parse_relaxed_action_plan(text: str, allowed_objects: set[str]) -> RelaxedParse:
    action_map = {name.casefold(): name for name in ACTION_ORDER}
    object_pattern, object_map = _object_pattern(allowed_objects)

    action_match = ACTION_BLOCK_PATTERN.search(text)
    if action_match is not None:
        actions, mentioned, incomplete = _contract_actions(
            action_match.group(1), action_map, object_map
        )
        if actions and not incomplete:
            return RelaxedParse(tuple(actions), True, True, "action_contract", "ok")

    plan_match = PLAN_BLOCK_PATTERN.search(text)
    block = plan_match.group(1) if plan_match is not None else text
    actions: list[str] = []
    mentioned = False
    incomplete = False
    for raw_line in block.splitlines():
        line = LEADING_MARKER_PATTERN.sub("", raw_line.strip()).strip("`* ")
        if not line or line.startswith("<"):
            continue
        recovered, line_mentioned, line_incomplete = _natural_line_actions(
            line, action_map, object_pattern, object_map
        )
        actions.extend(recovered)
        mentioned |= line_mentioned
        incomplete |= line_incomplete

    if actions and not incomplete:
        return RelaxedParse(tuple(actions), True, True, "natural_plan", "ok")
    if mentioned or actions:
        reason = "conditional_or_incomplete_action" if incomplete else "no_complete_sequence"
        return RelaxedParse((), True, False, "natural_plan", reason)
    return RelaxedParse((), False, False, "none", "no_action_semantics")


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise ValueError(f"Missing or duplicate sample_id in {path}: {sample_id}")
        result[sample_id] = row
    return result


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    exact = sum(bool(row["relaxed_action_exact"]) for row in records)
    matches = sum(int(row["position_matches"]) for row in records)
    gold_steps = sum(int(row["gold_steps"]) for row in records)
    predicted_steps = sum(int(row["predicted_steps"]) for row in records)
    recall = matches / gold_steps if gold_steps else 0.0
    precision = matches / predicted_steps if predicted_steps else 0.0
    return {
        "samples": count,
        "relaxed_action_sequence_em": exact / count if count else 0.0,
        "relaxed_step_position_recall": recall,
        "relaxed_step_position_precision": precision,
        "relaxed_step_position_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "relaxed_parse_rate": sum(bool(row["relaxed_parseable"]) for row in records) / count,
        "deterministic_plan_rate": sum(bool(row["deterministic_plan"]) for row in records) / count,
        "position_matches": matches,
        "gold_steps": gold_steps,
        "predicted_steps": predicted_steps,
    }


def evaluate(
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    before_predictions: dict[str, dict[str, Any]],
    after_predictions: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids", [])]
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    missing = [sample_id for sample_id in validation_ids if sample_id not in raw_by_id]
    if missing:
        raise ValueError(f"Raw validation rows missing: {missing[:10]}")
    allowed_objects = {
        argument
        for row in raw_rows
        for action in gold_actions(row)
        for parsed in [parse_action(action)]
        if parsed is not None
        for argument in parsed[1]
    }

    records: list[dict[str, Any]] = []
    by_label: dict[str, list[dict[str, Any]]] = {"before": [], "after": []}
    for sample_id in validation_ids:
        raw = raw_by_id[sample_id]
        gold = tuple(gold_actions(raw))
        for label, predictions in (("before", before_predictions), ("after", after_predictions)):
            if sample_id not in predictions:
                raise ValueError(f"{label} predictions missing {sample_id}")
            parsed = parse_relaxed_action_plan(
                prediction_text(predictions[sample_id]), allowed_objects
            )
            matches = sum(left == right for left, right in zip(parsed.actions, gold))
            record = {
                "sample_id": sample_id,
                "scene_id": str(metadata(raw).get("scene_id", "")),
                "label": label,
                "gold_actions": list(gold),
                "relaxed_actions": list(parsed.actions),
                "relaxed_parseable": parsed.relaxed_parseable,
                "deterministic_plan": parsed.deterministic_plan,
                "parse_source": parsed.source,
                "parse_reason": parsed.reason,
                "relaxed_action_exact": parsed.actions == gold,
                "position_matches": matches,
                "gold_steps": len(gold),
                "predicted_steps": len(parsed.actions),
            }
            records.append(record)
            by_label[label].append(record)

    result = {
        "scope": {
            "validation_samples": len(validation_ids),
            "selection": manifest.get("evaluation_subset", {}),
            "parser_contract": {
                "gold_or_instruction_used_for_parsing": False,
                "accepted": [
                    "canonical action syntax",
                    "minor punctuation/parenthesis/list-marker variants",
                    "explicit action names in prose with explicit ontology objects",
                    "explicit Chinese action verbs with explicit ontology objects",
                ],
                "rejected_as_nondeterministic": [
                    "conditional branches",
                    "negated actions",
                    "missing or unknown object arguments",
                    "actions requiring omitted arguments",
                ],
            },
        },
        "before": _summarize(by_label["before"]),
        "after": _summarize(by_label["after"]),
    }
    return result, records


def write_outputs(result: dict[str, Any], records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "scored_predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = []
    for label in ("before", "after"):
        rows.append({"model": label, **result[label]})
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    before, after = result["before"], result["after"]
    percent = lambda value: f"{float(value):.2%}"
    lines = [
        "# Relaxed semantic action metrics",
        "",
        f"评测样本：{result['scope']['validation_samples']} 条。Strict 指标不变；本报告只增加 relaxed 指标。",
        "",
        "| 指标 | 训练前 | 训练后 | 差值 |",
        "|:---|---:|---:|---:|",
    ]
    for name, key in (
        ("Relaxed Action Sequence EM", "relaxed_action_sequence_em"),
        ("Relaxed Step Position Recall", "relaxed_step_position_recall"),
        ("Relaxed Step Position Precision", "relaxed_step_position_precision"),
    ):
        lines.append(
            f"| {name} | {percent(before[key])} | {percent(after[key])} | "
            f"{(after[key] - before[key]) * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 审计诊断",
            "",
            "| 指标 | 训练前 | 训练后 |",
            "|:---|---:|---:|",
            f"| Relaxed Parse Rate | {percent(before['relaxed_parse_rate'])} | {percent(after['relaxed_parse_rate'])} |",
            f"| Deterministic Plan Rate | {percent(before['deterministic_plan_rate'])} | {percent(after['deterministic_plan_rate'])} |",
            f"| Relaxed Step F1 | {percent(before['relaxed_step_position_f1'])} | {percent(after['relaxed_step_position_f1'])} |",
            "",
            "解析器不读取 instruction、gold state、图片或单样本 gold action 来补动作或参数。条件分支、否定动作和参数缺失不会形成 deterministic plan。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate conservative relaxed semantic action metrics")
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--before-predictions", type=Path, required=True)
    parser.add_argument("--after-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result, records = evaluate(
        load_jsonl(args.raw_data.resolve()),
        manifest,
        _prediction_map(args.before_predictions.resolve()),
        _prediction_map(args.after_predictions.resolve()),
    )
    write_outputs(result, records, output_dir)
    print(output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
