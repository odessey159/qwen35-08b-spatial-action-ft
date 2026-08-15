from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .schema import normalize_action, parse_plan, read_jsonl, write_json


METRIC_NAMES = (
    "sequence_exact_match",
    "step_level_match",
    "structure_valid",
    "action_vocab_violation_rate",
    "object_vocab_violation_rate",
)


def _score_row(
    prediction: dict[str, Any],
    sample: dict[str, Any],
    allowed_actions: set[str],
    allowed_objects: set[str],
) -> dict[str, Any]:
    parsed = parse_plan(str(prediction.get("raw_output", "")))
    pred_actions = parsed.actions
    gold_actions = [normalize_action(action) for action in sample["gold"]["plan_actions"]]

    denominator = max(len(pred_actions), len(gold_actions), 1)
    positional_matches = sum(
        pred == gold for pred, gold in zip(pred_actions, gold_actions)
    )
    invalid_actions = sum(name not in allowed_actions for name in parsed.action_names)
    action_denominator = max(len(parsed.action_names), 1)

    object_arguments = [arg for args in parsed.arguments for arg in args]
    invalid_objects = (
        sum(arg not in allowed_objects for arg in object_arguments)
        if allowed_objects
        else 0
    )
    object_denominator = max(len(object_arguments), 1)

    return {
        "sample_id": sample["sample_id"],
        "condition": prediction["condition"],
        "sequence_exact_match": float(pred_actions == gold_actions),
        "step_level_match": positional_matches / denominator,
        "structure_valid": float(parsed.structure_valid),
        "action_vocab_violation_rate": invalid_actions / action_denominator,
        "object_vocab_violation_rate": (
            invalid_objects / object_denominator if allowed_objects else None
        ),
        "pred_actions": pred_actions,
        "gold_actions": gold_actions,
    }


def _aggregate(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[row["condition"]].append(row)

    results: dict[str, dict[str, float | None]] = {}
    for condition, rows in sorted(grouped.items()):
        condition_result: dict[str, float | None] = {"count": float(len(rows))}
        for metric in METRIC_NAMES:
            values = [row[metric] for row in rows if row[metric] is not None]
            condition_result[metric] = mean(values) if values else None
        results[condition] = condition_result
    return results


def _mean_metric(
    summary: dict[str, dict[str, float | None]],
    conditions: list[str],
    metric: str,
) -> float | None:
    values = [summary[c][metric] for c in conditions if c in summary and summary[c][metric] is not None]
    return mean(values) if values else None


def diagnose(
    summary: dict[str, dict[str, float | None]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    metric = "sequence_exact_match"
    a = _mean_metric(summary, ["A"], metric)
    a_prime = _mean_metric(summary, ["A_prime"], metric)
    b = _mean_metric(summary, ["B_natural", "B_json", "B_triples"], metric)
    c = _mean_metric(summary, ["C"], metric)
    d = _mean_metric(summary, ["D"], metric)
    d_structure = _mean_metric(summary, ["D"], "structure_valid")

    near_gap = float(thresholds["near_gap"])
    large_gap = float(thresholds["large_gap"])
    floor_score = float(thresholds["floor_score"])
    d_min_score = float(thresholds["d_min_score"])

    conclusions: list[str] = []
    if d is not None and d_structure is not None and (
        d < d_min_score or d_structure < d_min_score
    ):
        conclusions.append("D 较低：先处理输出格式、动作词表或结构合法性问题。")
    if a is not None and a_prime is not None and abs(a - a_prime) <= near_gap:
        conclusions.append("A 与 A′ 接近：模型没有有效使用图像。")
    if a is not None and b is not None and b - a >= large_gap:
        conclusions.append("B 显著高于 A：感知是主要瓶颈。")
    if a is not None and b is not None and abs(b - a) <= near_gap and max(a, b) <= floor_score:
        conclusions.append("A 与 B 接近且都低：reasoning/planning 是主要瓶颈。")
    if b is not None and c is not None and abs(c - b) <= near_gap:
        conclusions.append("C 接近 B：感知损失可被正确空间事实补偿。")
    if b is not None and c is not None and b - c >= large_gap:
        conclusions.append("C 显著低于 B：检查图像与文本事实的融合方式。")

    floor_effect = (
        a is not None
        and b is not None
        and c is not None
        and max(a, b, c) <= floor_score
    )
    if floor_effect:
        conclusions.append("A/B/C 全部接近地板：按总纲升级为各条件约 300 step 短 LoRA 后重测。")
    if not conclusions:
        conclusions.append("没有触发预设判读规则；需要结合各项分数人工分析。")

    return {
        "scores": {"A": a, "A_prime": a_prime, "B_macro": b, "C": c, "D": d},
        "deltas": {
            "A_minus_A_prime": a - a_prime if a is not None and a_prime is not None else None,
            "B_minus_A": b - a if b is not None and a is not None else None,
            "C_minus_A": c - a if c is not None and a is not None else None,
        },
        "floor_effect": floor_effect,
        "conclusions": conclusions,
    }


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _write_csv(path: Path, summary: dict[str, dict[str, float | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "count", *METRIC_NAMES])
        for condition, metrics in sorted(summary.items()):
            writer.writerow([condition, metrics["count"], *(metrics[name] for name in METRIC_NAMES)])


def _write_report(
    path: Path,
    summary: dict[str, dict[str, float | None]],
    diagnosis: dict[str, Any],
) -> None:
    lines = [
        "# Exp 0 诊断结果",
        "",
        "| 条件 | N | Sequence EM | Step Match | 结构合法率 | 动作越界率 | 物体越界率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in sorted(summary.items()):
        lines.append(
            f"| {condition} | {int(metrics['count'] or 0)} | "
            f"{_format_percent(metrics['sequence_exact_match'])} | "
            f"{_format_percent(metrics['step_level_match'])} | "
            f"{_format_percent(metrics['structure_valid'])} | "
            f"{_format_percent(metrics['action_vocab_violation_rate'])} | "
            f"{_format_percent(metrics['object_vocab_violation_rate'])} |"
        )

    lines.extend(["", "## 条件差值", ""])
    for name, value in diagnosis["deltas"].items():
        lines.append(f"- {name}: {_format_percent(value)}")

    lines.extend(["", "## 判读", ""])
    lines.extend(f"- {text}" for text in diagnosis["conclusions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    config: dict[str, Any],
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> None:
    samples = {row["sample_id"]: row for row in read_jsonl(dataset_path)}
    predictions = read_jsonl(predictions_path)
    allowed_actions = set(config["allowed_actions"])
    allowed_objects = set(config.get("allowed_objects", []))

    expected_conditions = {
        "D",
        "A_prime",
        "A",
        "B_natural",
        "B_json",
        "B_triples",
        "C",
    }
    seen_keys: set[tuple[str, str]] = set()
    scored_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        sample_id = prediction.get("sample_id")
        condition = prediction.get("condition")
        if sample_id not in samples:
            raise ValueError(f"Prediction references unknown sample_id: {sample_id}")
        if condition not in expected_conditions:
            raise ValueError(f"Unknown condition in predictions: {condition}")
        key = (str(sample_id), str(condition))
        if key in seen_keys:
            raise ValueError(f"Duplicate prediction: {key}")
        seen_keys.add(key)
        scored_rows.append(
            _score_row(prediction, samples[str(sample_id)], allowed_actions, allowed_objects)
        )

    missing = [
        (sample_id, condition)
        for sample_id in samples
        for condition in expected_conditions
        if (sample_id, condition) not in seen_keys
    ]
    if missing:
        preview = ", ".join(f"{sid}/{condition}" for sid, condition in missing[:10])
        raise ValueError(f"Missing {len(missing)} predictions; first entries: {preview}")

    summary = _aggregate(scored_rows)
    diagnosis_result = diagnose(summary, config["diagnosis_thresholds"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", {"conditions": summary, "diagnosis": diagnosis_result})
    write_json(output_dir / "scored_predictions.json", scored_rows)
    _write_csv(output_dir / "metrics_by_condition.csv", summary)
    _write_report(output_dir / "report.md", summary, diagnosis_result)

