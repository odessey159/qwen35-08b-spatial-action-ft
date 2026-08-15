from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .prompts import FORMAT_EXAMPLE_OBJECTS, FORMAT_EXAMPLE_PLAN
from .schema import (
    extract_summary,
    normalize_action,
    parse_plan,
    parse_plan_lenient,
    read_jsonl,
    score_summary,
    write_json,
)


# Ordered as reported. The action sequence is the primary measure: it is a closed
# vocabulary, so exact match is unambiguous and needs no judge. It is scored with
# the lenient reader (see schema.parse_plan_lenient) -- the strict parse is kept
# alongside purely as a format-compliance signal, not as a capability measure.
# The <summary> metrics stay on as a secondary read on the natural language half
# of the dual output required by §1.1.
NL_METRICS = (
    "nl_plan_match",
    "nl_action_recall",
    "nl_object_recall",
    "nl_order_ok",
    "nl_char_f1",
    "nl_present",
)
ACTION_METRICS = (
    "action_seq_em",
    "action_step_match",
    "action_parsed",
)
FORMAT_METRICS = (
    "strict_seq_em",
    "strict_structure_valid",
    "action_vocab_violation_rate",
    "object_vocab_violation_rate",
    "example_echo_rate",
    "contract_echo_rate",
)
METRIC_NAMES = ACTION_METRICS + NL_METRICS + FORMAT_METRICS

EXPECTED_CONDITIONS = ("D", "A_prime", "A", "B_natural", "B_json", "B_triples", "C")
B_CONDITIONS = ("B_natural", "B_json", "B_triples")

_CONTRACT_ECHO_MARKERS = (
    "随后用一句自然语言",
    "动作序列必须放在最前面",
    "请严格输出",
    "可用动作仅限",
    "ActionName",
    "每行一个动作",
)


def _score_row(
    prediction: dict[str, Any],
    sample: dict[str, Any],
    allowed_actions: set[str],
    allowed_objects: set[str],
) -> dict[str, Any]:
    raw_output = str(prediction.get("raw_output", ""))
    gold_actions = [normalize_action(action) for action in sample["gold"]["plan_actions"]]
    gold_nl = str(sample["gold"].get("plan_nl", ""))

    predicted_nl = extract_summary(raw_output)
    nl_scores = score_summary(predicted_nl, gold_actions, gold_nl)

    lenient_actions = parse_plan_lenient(raw_output, allowed_actions)
    strict = parse_plan(raw_output)

    denominator = max(len(lenient_actions), len(gold_actions), 1)
    positional_matches = sum(
        pred == gold for pred, gold in zip(lenient_actions, gold_actions)
    )

    invalid_actions = sum(name not in allowed_actions for name in strict.action_names)
    object_arguments = [arg for args in strict.arguments for arg in args]
    invalid_objects = (
        sum(arg not in allowed_objects for arg in object_arguments)
        if allowed_objects
        else 0
    )

    echoed_example = any(action in raw_output for action in FORMAT_EXAMPLE_PLAN) or any(
        obj in raw_output for obj in FORMAT_EXAMPLE_OBJECTS
    )

    return {
        "sample_id": sample["sample_id"],
        "condition": prediction["condition"],
        "task_group": sample.get("meta", {}).get("task_group"),
        "counterfactual": bool(sample.get("meta", {}).get("counterfactual_group")),
        **nl_scores,
        "action_seq_em": float(lenient_actions == gold_actions),
        "action_step_match": positional_matches / denominator,
        "action_parsed": float(bool(lenient_actions)),
        "strict_seq_em": float(strict.actions == gold_actions),
        "strict_structure_valid": float(strict.structure_valid),
        "action_vocab_violation_rate": invalid_actions / max(len(strict.action_names), 1),
        "object_vocab_violation_rate": (
            invalid_objects / max(len(object_arguments), 1) if allowed_objects else None
        ),
        "example_echo_rate": float(echoed_example),
        "contract_echo_rate": float(
            any(marker in raw_output for marker in _CONTRACT_ECHO_MARKERS)
        ),
        "pred_nl": predicted_nl,
        "gold_nl": gold_nl,
        "pred_actions": lenient_actions,
        "gold_actions": gold_actions,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {"count": float(len(rows))}
    for metric in METRIC_NAMES:
        values = [row[metric] for row in rows if row.get(metric) is not None]
        result[metric] = mean(values) if values else None
    return result


def _aggregate(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        grouped[row["condition"]].append(row)
    return {condition: _summarize(rows) for condition, rows in sorted(grouped.items())}


def _slice_report(scored_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Per-condition scores on the slices that carry the experiment's signal.

    81.7% of this dataset is solvable from the instruction text alone; the
    counterfactual pairs are the only slice where an image is required. Reporting
    only the pooled number hides that, so A - A' is also emitted per slice.
    """
    slices: dict[str, Any] = {}
    definitions: list[tuple[str, Any]] = [
        ("all", lambda row: True),
        ("counterfactual", lambda row: row["counterfactual"]),
        ("non_counterfactual", lambda row: not row["counterfactual"]),
    ]
    task_groups = sorted({row["task_group"] for row in scored_rows if row["task_group"]})
    definitions += [
        (f"task_group:{group}", (lambda g: lambda row: row["task_group"] == g)(group))
        for group in task_groups
    ]

    for name, predicate in definitions:
        subset = [row for row in scored_rows if predicate(row)]
        if not subset:
            continue
        per_condition = {
            condition: _summarize([row for row in subset if row["condition"] == condition])
            for condition in EXPECTED_CONDITIONS
        }
        a = per_condition["A"][metric]
        a_prime = per_condition["A_prime"][metric]
        slices[name] = {
            "count": len(subset) // len(EXPECTED_CONDITIONS),
            "metric": metric,
            "scores": {c: per_condition[c][metric] for c in EXPECTED_CONDITIONS},
            "A_minus_A_prime": (a - a_prime) if a is not None and a_prime is not None else None,
        }
    return slices


def _mean_metric(
    summary: dict[str, dict[str, float | None]],
    conditions: list[str],
    metric: str,
) -> float | None:
    values = [
        summary[c][metric]
        for c in conditions
        if c in summary and summary[c][metric] is not None
    ]
    return mean(values) if values else None


def diagnose(
    summary: dict[str, dict[str, float | None]],
    thresholds: dict[str, float],
    metric: str,
) -> dict[str, Any]:
    a = _mean_metric(summary, ["A"], metric)
    a_prime = _mean_metric(summary, ["A_prime"], metric)
    b = _mean_metric(summary, list(B_CONDITIONS), metric)
    c = _mean_metric(summary, ["C"], metric)
    d = _mean_metric(summary, ["D"], metric)

    b_values = [
        summary[name][metric]
        for name in B_CONDITIONS
        if name in summary and summary[name][metric] is not None
    ]
    b_spread = (max(b_values) - min(b_values)) if b_values else None

    near_gap = float(thresholds["near_gap"])
    large_gap = float(thresholds["large_gap"])
    floor_score = float(thresholds["floor_score"])
    d_min_score = float(thresholds["d_min_score"])

    # §6.1 judgement table: D is a lower-bound check. If the model cannot emit a
    # correct plan when the correct sub-goals are handed to it, every other
    # comparison is measuring output plumbing rather than capability, so no
    # further conclusion is emitted. The previous version appended the remaining
    # conclusions anyway, which is how report.md came to assert "模型没有有效使用
    # 图像" directly under a line saying the other conditions must not be read.
    if d is None:
        return {
            "primary_metric": metric,
            "scores": {"A": a, "A_prime": a_prime, "B_macro": b, "C": c, "D": d},
            "deltas": {},
            "d_gate_passed": False,
            "floor_effect": None,
            "conclusions": ["缺少 D 条件的分数，无法判读。"],
        }

    if d < d_min_score:
        return {
            "primary_metric": metric,
            "scores": {"A": a, "A_prime": a_prime, "B_macro": b, "C": c, "D": d},
            "deltas": {},
            "d_gate_passed": False,
            "floor_effect": None,
            "conclusions": [
                f"D（完整 oracle）在 {metric} 上只有 {d:.1%}，低于门槛 {d_min_score:.0%}。"
                "按总纲 §6.1，这说明问题出在输出格式、动作词表或结构合法性，"
                "A/A′/B/C 之间的对比在此之前没有意义，因此不再输出其余判读。",
            ],
        }

    conclusions: list[str] = [
        f"D（完整 oracle）在 {metric} 上达到 {d:.1%}，通过下限检查，以下对比可读。"
    ]
    if a is not None and a_prime is not None and abs(a - a_prime) <= near_gap:
        conclusions.append("A 与 A′ 接近：模型没有有效使用图像。")
    if a is not None and a_prime is not None and a - a_prime > near_gap:
        conclusions.append(f"A 高于 A′ {a - a_prime:.1%}：这是视觉的真实贡献量。")
    if a is not None and b is not None and b - a >= large_gap:
        conclusions.append("B 显著高于 A：感知是主要瓶颈。")
    if (
        a is not None
        and b is not None
        and abs(b - a) <= near_gap
        and max(a, b) <= floor_score
    ):
        conclusions.append("A 与 B 接近且都低：reasoning/planning 是主要瓶颈。")
    if b is not None and c is not None and abs(c - b) <= near_gap:
        conclusions.append("C 接近 B：感知损失可被正确空间事实补偿。")
    if b is not None and c is not None and b - c >= large_gap:
        conclusions.append("C 显著低于 B：检查图像与文本事实的融合方式。")
    if b_spread is not None and b_spread > near_gap:
        conclusions.append(
            f"三种场景图序列化之间相差 {b_spread:.1%}，超过 near_gap："
            "按总纲 §6.1(2)，B 的结论应标注为受序列化格式混淆。"
        )

    floor_effect = (
        a is not None and b is not None and c is not None and max(a, b, c) <= floor_score
    )
    if floor_effect:
        conclusions.append("A/B/C 全部接近地板：按总纲升级为各条件约 300 step 短 LoRA 后重测。")

    return {
        "primary_metric": metric,
        "scores": {"A": a, "A_prime": a_prime, "B_macro": b, "C": c, "D": d},
        "deltas": {
            "A_minus_A_prime": a - a_prime if a is not None and a_prime is not None else None,
            "B_minus_A": b - a if b is not None and a is not None else None,
            "C_minus_A": c - a if c is not None and a is not None else None,
            "B_serialization_spread": b_spread,
        },
        "d_gate_passed": True,
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
            writer.writerow(
                [condition, metrics["count"], *(metrics[name] for name in METRIC_NAMES)]
            )


def _write_report(
    path: Path,
    summary: dict[str, dict[str, float | None]],
    diagnosis: dict[str, Any],
    slices: dict[str, Any],
) -> None:
    metric = diagnosis["primary_metric"]
    lines = [
        "# Exp 0 诊断结果",
        "",
        f"主指标：**`{metric}`**（动作序列，宽松解析后的精确匹配）。",
        "严格解析结果只作为格式合规率上报，不代表能力；自然语言指标作为次要参考。",
        "",
        "## 动作序列（主，宽松解析）",
        "",
        "| 条件 | N | 序列精确匹配 | 步级匹配 | 解析出动作 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, m in sorted(summary.items()):
        lines.append(
            f"| {condition} | {int(m['count'] or 0)} | "
            + " | ".join(_format_percent(m[name]) for name in ACTION_METRICS)
            + " |"
        )

    lines += [
        "",
        "## 自然语言输出（次）",
        "",
        "| 条件 | NL 计划匹配 | 动作召回 | 物体召回 | 顺序正确 | 字符 F1 | 有输出 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, m in sorted(summary.items()):
        lines.append(
            f"| {condition} | "
            + " | ".join(_format_percent(m[name]) for name in NL_METRICS)
            + " |"
        )

    lines += [
        "",
        "## 格式合规与污染检查",
        "",
        "| 条件 | 严格序列 EM | 严格结构合法 | 动作越界 | 物体越界 | 抄示例 | 抄提示词 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, m in sorted(summary.items()):
        lines.append(
            f"| {condition} | "
            + " | ".join(_format_percent(m[name]) for name in FORMAT_METRICS)
            + " |"
        )

    lines += ["", f"## 按切片的 {metric}", ""]
    header_conditions = list(EXPECTED_CONDITIONS)
    lines.append("| 切片 | N | " + " | ".join(header_conditions) + " | A − A′ |")
    lines.append("|---|---:" + "|---:" * (len(header_conditions) + 1) + "|")
    for name, payload in slices.items():
        cells = [_format_percent(payload["scores"][c]) for c in header_conditions]
        lines.append(
            f"| {name} | {payload['count']} | "
            + " | ".join(cells)
            + f" | {_format_percent(payload['A_minus_A_prime'])} |"
        )

    lines += ["", "## 条件差值", ""]
    if diagnosis["deltas"]:
        for name, value in diagnosis["deltas"].items():
            lines.append(f"- {name}: {_format_percent(value)}")
    else:
        lines.append("- D 未通过下限检查，不计算条件差值。")

    lines += ["", "## 判读", ""]
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
    metric = str(config.get("primary_metric", "action_seq_em"))
    if metric not in METRIC_NAMES:
        raise ValueError(f"primary_metric must be one of {METRIC_NAMES}: {metric}")

    expected_conditions = set(EXPECTED_CONDITIONS)
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
    slices = _slice_report(scored_rows, metric)
    diagnosis_result = diagnose(summary, config["diagnosis_thresholds"], metric)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "metrics.json",
        {"conditions": summary, "slices": slices, "diagnosis": diagnosis_result},
    )
    write_json(output_dir / "scored_predictions.json", scored_rows)
    _write_csv(output_dir / "metrics_by_condition.csv", summary)
    _write_report(output_dir / "report.md", summary, diagnosis_result, slices)
