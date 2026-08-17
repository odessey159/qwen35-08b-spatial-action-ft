from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

from exp0.schema import normalize_action, parse_action, parse_plan
from training.evaluate_section_losses import load_jsonl


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("meta", row.get("metadata", row.get("_meta", {})))
    return value if isinstance(value, dict) else {}


def gold_actions(row: dict[str, Any]) -> tuple[str, ...]:
    value: Any = None
    if isinstance(row.get("gold"), dict):
        value = row["gold"].get("plan_actions")
    if value is None and isinstance(row.get("oracle"), dict):
        value = row["oracle"].get("actions")
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{row.get('sample_id')}: no gold action list")
    return tuple(normalize_action(item) for item in value)


def prediction_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_output", row.get("prediction", row.get("output", ""))))


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def clustered_bootstrap_interval(
    outcomes: list[bool] | list[float],
    cluster_ids: list[str],
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Percentile CI from resampling whole scene clusters with replacement."""

    if len(outcomes) != len(cluster_ids) or not outcomes:
        raise ValueError("Clustered outcomes must be non-empty and aligned")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster, outcome in zip(cluster_ids, outcomes):
        grouped[cluster].append(float(outcome))
    clusters = sorted(grouped)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in clusters:
            values = grouped[rng.choice(clusters)]
            total += sum(values)
            count += len(values)
        estimates.append(total / count)
    alpha = (1 - confidence) / 2
    return [_percentile(estimates, alpha), _percentile(estimates, 1 - alpha)]


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("-inf")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def mcnemar_exact_p(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, candidate_only)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2)
        for k in range(smaller + 1)
    ]
    return min(1.0, 2 * math.exp(_logsumexp(log_probabilities)))


def paired_difference(
    baseline: list[bool],
    candidate: list[bool],
    confidence: float = 0.95,
    cluster_ids: list[str] | None = None,
    bootstrap_resamples: int = 1000,
) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired outcomes must be non-empty and have equal lengths")
    baseline_only = sum(left and not right for left, right in zip(baseline, candidate))
    candidate_only = sum(right and not left for left, right in zip(baseline, candidate))
    both_correct = sum(left and right for left, right in zip(baseline, candidate))
    neither_correct = len(baseline) - baseline_only - candidate_only - both_correct
    differences = [float(right) - float(left) for left, right in zip(baseline, candidate)]
    mean = sum(differences) / len(differences)
    if cluster_ids is not None:
        interval = clustered_bootstrap_interval(
            differences,
            cluster_ids,
            confidence=confidence,
            resamples=bootstrap_resamples,
        )
        interval_method = "scene_cluster_percentile_bootstrap"
    elif len(differences) > 1:
        variance = sum((value - mean) ** 2 for value in differences) / (
            len(differences) - 1
        )
        standard_error = math.sqrt(variance / len(differences))
        z = NormalDist().inv_cdf(0.5 + confidence / 2)
        interval = [
            max(-1.0, mean - z * standard_error),
            min(1.0, mean + z * standard_error),
        ]
        interval_method = "normal_approximation"
    else:
        interval = [mean, mean]
        interval_method = "singleton"
    return {
        "count": len(baseline),
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "both_correct": both_correct,
        "neither_correct": neither_correct,
        "accuracy_delta": mean,
        "accuracy_delta_ci95": interval,
        "ci_method": interval_method,
        "bootstrap_resamples": bootstrap_resamples if cluster_ids is not None else None,
        "mcnemar_exact_p": mcnemar_exact_p(baseline_only, candidate_only),
    }


def _target_object(actions: tuple[str, ...]) -> str:
    for action in reversed(actions):
        parsed = parse_action(action)
        if parsed is not None and parsed[1]:
            return parsed[1][0]
    return "unknown"


def _error_type(
    predicted: tuple[str, ...], gold: tuple[str, ...], state: str
) -> str:
    if predicted == gold:
        return "exact"
    if not predicted:
        return "empty_or_unparseable"
    has_open = any(action.startswith("OpenObject(") for action in predicted)
    gold_has_open = any(action.startswith("OpenObject(") for action in gold)
    if state == "closed" and gold_has_open and not has_open:
        return "missing_open_for_closed"
    if state == "open" and not gold_has_open and has_open:
        return "redundant_open_for_open"
    if predicted and gold and predicted[0] != gold[0]:
        return "navigation_or_target_error"
    if len(predicted) != len(gold):
        return "wrong_action_count"
    return "other_action_error"


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id"))
        if not sample_id or sample_id == "None":
            raise ValueError(f"Prediction without sample_id: {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate prediction for {sample_id}: {path}")
        result[sample_id] = row
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_selected_raw_rows(path: Path, sample_ids: set[str]) -> list[dict[str, Any]]:
    """Stream a large raw dataset and retain only requested validation rows."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
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
            sample_id = str(row.get("sample_id"))
            if sample_id not in sample_ids:
                continue
            if sample_id in seen:
                raise ValueError(f"Duplicate selected raw sample: {sample_id}")
            seen.add(sample_id)
            result.append(row)
    missing = sorted(sample_ids - seen)
    if missing:
        raise ValueError(
            f"Raw dataset is missing {len(missing)} selected samples; first IDs: {missing[:10]}"
        )
    return result


def _complete_validation_pairs(
    raw_rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[tuple[str, list[dict[str, Any]]]]:
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids", [])]
    if not validation_ids:
        raise ValueError("Manifest has no validation_sample_ids")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id in validation_ids:
        if sample_id not in raw_by_id:
            raise ValueError(f"Raw validation sample is missing: {sample_id}")
        row = raw_by_id[sample_id]
        group = metadata(row).get("counterfactual_group")
        if group not in (None, ""):
            groups[str(group)].append(row)
    incomplete = {group: len(rows) for group, rows in groups.items() if len(rows) != 2}
    if incomplete:
        preview = dict(list(sorted(incomplete.items()))[:10])
        raise ValueError(f"Incomplete counterfactual validation groups: {preview}")
    return [(group, groups[group]) for group in sorted(groups)]


def compare(
    raw_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = _complete_validation_pairs(raw_rows, manifest)
    required_ids = {
        str(row["sample_id"]) for _, pair_rows in pairs for row in pair_rows
    }
    for label, predictions in (
        (baseline_label, baseline_rows),
        (candidate_label, candidate_rows),
    ):
        missing = sorted(required_ids - set(predictions))
        if missing:
            raise ValueError(
                f"{label} is missing {len(missing)} counterfactual predictions; "
                f"first IDs: {missing[:10]}"
            )

    labels = (baseline_label, candidate_label)
    predictions_by_label = {
        baseline_label: baseline_rows,
        candidate_label: candidate_rows,
    }
    sample_outcomes: dict[str, list[bool]] = {label: [] for label in labels}
    pair_outcomes: dict[str, list[bool]] = {label: [] for label in labels}
    structures: dict[str, list[bool]] = {label: [] for label in labels}
    error_counts: dict[str, Counter[str]] = {label: Counter() for label in labels}
    state_counts: dict[str, dict[str, list[bool]]] = {
        label: defaultdict(list) for label in labels
    }
    object_counts: dict[str, dict[str, list[bool]]] = {
        label: defaultdict(list) for label in labels
    }
    generated_tokens: dict[str, list[int]] = {label: [] for label in labels}
    seconds: dict[str, list[float]] = {label: [] for label in labels}
    pair_rows_output: list[dict[str, Any]] = []
    sample_clusters: list[str] = []
    pair_clusters: list[str] = []

    for group, raw_pair in pairs:
        raw_pair = sorted(
            raw_pair,
            key=lambda row: (str(metadata(row).get("receptacle_state", "")), str(row["sample_id"])),
        )
        gold_by_id = {str(row["sample_id"]): gold_actions(row) for row in raw_pair}
        states_by_id = {
            str(row["sample_id"]): str(metadata(row).get("receptacle_state", "unknown"))
            for row in raw_pair
        }
        target = _target_object(next(iter(gold_by_id.values())))
        output_row: dict[str, Any] = {
            "counterfactual_group": group,
            "target_object": target,
            "sample_ids": " | ".join(str(row["sample_id"]) for row in raw_pair),
            "states": " | ".join(states_by_id[str(row["sample_id"])] for row in raw_pair),
            "gold_actions": " | ".join(
                "; ".join(gold_by_id[str(row["sample_id"])]) for row in raw_pair
            ),
        }
        scenes = {str(metadata(row).get("scene_id", "")) for row in raw_pair}
        if len(scenes) != 1 or "" in scenes:
            raise ValueError(f"{group}: counterfactual pair must have one scene_id")
        pair_scene = next(iter(scenes))
        pair_clusters.append(pair_scene)
        sample_clusters.extend([pair_scene] * len(raw_pair))
        for label in labels:
            member_exact: list[bool] = []
            predicted_pair: list[tuple[str, ...]] = []
            for row in raw_pair:
                sample_id = str(row["sample_id"])
                prediction_row = predictions_by_label[label][sample_id]
                parsed = parse_plan(prediction_text(prediction_row))
                predicted = tuple(parsed.actions)
                gold = gold_by_id[sample_id]
                exact = predicted == gold
                member_exact.append(exact)
                predicted_pair.append(predicted)
                sample_outcomes[label].append(exact)
                structures[label].append(parsed.structure_valid)
                state = states_by_id[sample_id]
                state_counts[label][state].append(exact)
                object_counts[label][target].append(exact)
                error_counts[label][_error_type(predicted, gold, state)] += 1
                if isinstance(prediction_row.get("generated_tokens"), int):
                    generated_tokens[label].append(int(prediction_row["generated_tokens"]))
                if isinstance(prediction_row.get("seconds"), (int, float)):
                    seconds[label].append(float(prediction_row["seconds"]))
            pair_exact = all(member_exact)
            pair_outcomes[label].append(pair_exact)
            output_row[f"{label}_actions"] = " | ".join(
                "; ".join(actions) if actions else "<unparseable>"
                for actions in predicted_pair
            )
            output_row[f"{label}_member_exact"] = " | ".join(
                "1" if value else "0" for value in member_exact
            )
            output_row[f"{label}_pair_exact"] = int(pair_exact)
            output_row[f"{label}_same_actions"] = int(
                predicted_pair[0] == predicted_pair[1]
            )
        baseline_pair_exact = bool(output_row[f"{baseline_label}_pair_exact"])
        candidate_pair_exact = bool(output_row[f"{candidate_label}_pair_exact"])
        if candidate_pair_exact and not baseline_pair_exact:
            comparison = "candidate_fixed"
        elif baseline_pair_exact and not candidate_pair_exact:
            comparison = "candidate_regressed"
        elif baseline_pair_exact:
            comparison = "both_exact"
        else:
            comparison = "both_not_exact"
        output_row["comparison"] = comparison
        pair_rows_output.append(output_row)

    model_summaries: dict[str, Any] = {}
    for label in labels:
        sample_successes = sum(sample_outcomes[label])
        pair_successes = sum(pair_outcomes[label])
        same_actions = sum(int(row[f"{label}_same_actions"]) for row in pair_rows_output)
        correct_members_per_pair = [
            sum(int(value) for value in row[f"{label}_member_exact"].split(" | "))
            for row in pair_rows_output
        ]
        model_summaries[label] = {
            "samples": len(sample_outcomes[label]),
            "pairs": len(pair_outcomes[label]),
            "sample_exact_accuracy": sample_successes / len(sample_outcomes[label]),
            "sample_exact_ci95": clustered_bootstrap_interval(
                sample_outcomes[label], sample_clusters
            ),
            "pair_exact_accuracy": pair_successes / len(pair_outcomes[label]),
            "pair_exact_ci95": clustered_bootstrap_interval(
                pair_outcomes[label], pair_clusters
            ),
            "ci_method": "scene_cluster_percentile_bootstrap",
            "bootstrap_resamples": 1000,
            "pair_member_outcomes": {
                "both_correct": correct_members_per_pair.count(2),
                "one_member_correct": correct_members_per_pair.count(1),
                "zero_members_correct": correct_members_per_pair.count(0),
            },
            "strict_action_structure_valid_rate": sum(structures[label]) / len(structures[label]),
            "same_action_sequence_rate": same_actions / len(pair_outcomes[label]),
            "error_counts": dict(error_counts[label].most_common()),
            "by_receptacle_state": {
                name: {
                    "count": len(values),
                    "exact_accuracy": sum(values) / len(values),
                }
                for name, values in sorted(state_counts[label].items())
            },
            "by_target_object": {
                name: {
                    "count": len(values),
                    "exact_accuracy": sum(values) / len(values),
                }
                for name, values in sorted(object_counts[label].items())
            },
            "generation": {
                "mean_generated_tokens": (
                    sum(generated_tokens[label]) / len(generated_tokens[label])
                    if generated_tokens[label]
                    else None
                ),
                "mean_seconds_per_sample": (
                    sum(seconds[label]) / len(seconds[label]) if seconds[label] else None
                ),
            },
        }

    summary = {
        "scope": {
            "validation_samples": len(manifest.get("validation_sample_ids", [])),
            "complete_counterfactual_pairs": len(pairs),
            "counterfactual_samples": len(required_ids),
            "counterfactual_scenes": len(
                {
                    str(metadata(row).get("scene_id"))
                    for _, rows in pairs
                    for row in rows
                    if metadata(row).get("scene_id") not in (None, "")
                }
            ),
            "gold_discriminative_pairs": sum(
                gold_actions(rows[0]) != gold_actions(rows[1]) for _, rows in pairs
            ),
            "same_instruction_pairs": sum(
                str(rows[0].get("instruction")) == str(rows[1].get("instruction"))
                for _, rows in pairs
            ),
            "same_scene_pairs": sum(
                str(metadata(rows[0]).get("scene_id"))
                == str(metadata(rows[1]).get("scene_id"))
                for _, rows in pairs
            ),
            "open_closed_state_pairs": sum(
                {
                    str(metadata(row).get("receptacle_state")) for row in rows
                }
                == {"open", "closed"}
                for _, rows in pairs
            ),
            "distinct_image_hash_pairs": sum(
                bool(metadata(rows[0]).get("image_sha256"))
                and bool(metadata(rows[1]).get("image_sha256"))
                and metadata(rows[0]).get("image_sha256")
                != metadata(rows[1]).get("image_sha256")
                for _, rows in pairs
            ),
        },
        "models": model_summaries,
        "paired_comparison": {
            "baseline": baseline_label,
            "candidate": candidate_label,
            "sample_exact": paired_difference(
                sample_outcomes[baseline_label],
                sample_outcomes[candidate_label],
                cluster_ids=sample_clusters,
            ),
            "pair_exact": paired_difference(
                pair_outcomes[baseline_label],
                pair_outcomes[candidate_label],
                cluster_ids=pair_clusters,
            ),
            "pair_outcome_counts": dict(
                Counter(row["comparison"] for row in pair_rows_output)
            ),
        },
    }
    return summary, pair_rows_output


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_p(value: float) -> str:
    return "p<1e-300" if value == 0.0 else f"p={value:.3g}"


def write_outputs(
    summary: dict[str, Any], pair_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "pair_outcomes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    comparison = summary["paired_comparison"]
    baseline_label = comparison["baseline"]
    candidate_label = comparison["candidate"]
    baseline = summary["models"][baseline_label]
    candidate = summary["models"][candidate_label]
    sample_delta = comparison["sample_exact"]
    pair_delta = comparison["pair_exact"]
    lines = [
        "# Counterfactual 成对评测",
        "",
        f"验证集包含 {summary['scope']['complete_counterfactual_pairs']} 个完整反事实对（{summary['scope']['counterfactual_samples']} 条样本）；每一对的 gold action 均不同。",
        f"这些样本来自 {summary['scope']['counterfactual_scenes']} 个场景；其中同指令 {summary['scope']['same_instruction_pairs']} 对、同场景 {summary['scope']['same_scene_pairs']} 对、open/closed 状态对 {summary['scope']['open_closed_state_pairs']} 对、图像哈希不同 {summary['scope']['distinct_image_hash_pairs']} 对。",
        "95% CI 使用按 `scene_id` 重采样的 1000 次 percentile cluster bootstrap。",
        "",
        "## 总体结果",
        "",
        "| 模型 | sample exact | 95% CI | pair exact | 95% CI | 同动作率 | 结构合规率 |",
        "|:---|---:|:---:|---:|:---:|---:|---:|",
    ]
    for label, model in ((baseline_label, baseline), (candidate_label, candidate)):
        lines.append(
            f"| {label} | {_format_percent(model['sample_exact_accuracy'])} | "
            f"{_format_percent(model['sample_exact_ci95'][0])}–{_format_percent(model['sample_exact_ci95'][1])} | "
            f"{_format_percent(model['pair_exact_accuracy'])} | "
            f"{_format_percent(model['pair_exact_ci95'][0])}–{_format_percent(model['pair_exact_ci95'][1])} | "
            f"{_format_percent(model['same_action_sequence_rate'])} | "
            f"{_format_percent(model['strict_action_structure_valid_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## 成对差异",
            "",
            f"- sample exact 变化：{sample_delta['accuracy_delta']:+.2%}（95% CI {sample_delta['accuracy_delta_ci95'][0]:+.2%} 到 {sample_delta['accuracy_delta_ci95'][1]:+.2%}；McNemar {_format_p(sample_delta['mcnemar_exact_p'])}）。",
            f"- pair exact 变化：{pair_delta['accuracy_delta']:+.2%}（95% CI {pair_delta['accuracy_delta_ci95'][0]:+.2%} 到 {pair_delta['accuracy_delta_ci95'][1]:+.2%}；McNemar {_format_p(pair_delta['mcnemar_exact_p'])}）。",
            f"- pair 修复/退化：{pair_delta['candidate_only_correct']} / {pair_delta['baseline_only_correct']}；两者都对/都错：{pair_delta['both_correct']} / {pair_delta['neither_correct']}。",
            "",
            "## 单模型 pair 失败分解",
            "",
            "| 模型 | 两条都对 | 仅一条对 | 两条都错 |",
            "|:---|---:|---:|---:|",
        ]
    )
    for label, model in ((baseline_label, baseline), (candidate_label, candidate)):
        outcomes = model["pair_member_outcomes"]
        lines.append(
            f"| {label} | {outcomes['both_correct']} | {outcomes['one_member_correct']} | {outcomes['zero_members_correct']} |"
        )
    lines.extend(
        [
            "",
            "## 状态分层",
            "",
            "| 状态 | " + baseline_label + " | " + candidate_label + " | 差值 |",
            "|:---|---:|---:|---:|",
        ]
    )
    states = sorted(
        set(baseline["by_receptacle_state"]) | set(candidate["by_receptacle_state"])
    )
    for state in states:
        left = baseline["by_receptacle_state"][state]["exact_accuracy"]
        right = candidate["by_receptacle_state"][state]["exact_accuracy"]
        lines.append(
            f"| {state} | {_format_percent(left)} | {_format_percent(right)} | {right-left:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## 容器类型分层",
            "",
            "| 容器 | n | " + baseline_label + " | " + candidate_label + " | 差值 |",
            "|:---|---:|---:|---:|---:|",
        ]
    )
    objects = sorted(set(baseline["by_target_object"]) | set(candidate["by_target_object"]))
    for target in objects:
        left_row = baseline["by_target_object"][target]
        right_row = candidate["by_target_object"][target]
        left = left_row["exact_accuracy"]
        right = right_row["exact_accuracy"]
        lines.append(
            f"| {target} | {left_row['count']} | {_format_percent(left)} | {_format_percent(right)} | {right-left:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## 错误类型（按样本）",
            "",
            "| 类型 | " + baseline_label + " | " + candidate_label + " |",
            "|:---|---:|---:|",
        ]
    )
    error_types = sorted(set(baseline["error_counts"]) | set(candidate["error_counts"]))
    for error_type in error_types:
        lines.append(
            f"| {error_type} | {baseline['error_counts'].get(error_type, 0)} | {candidate['error_counts'].get(error_type, 0)} |"
        )
    lines.extend(
        [
            "",
            "## 生成开销",
            "",
            "| 模型 | 平均生成 token | 平均秒/样本 |",
            "|:---|---:|---:|",
        ]
    )
    for label, model in ((baseline_label, baseline), (candidate_label, candidate)):
        generation = model["generation"]
        mean_tokens = generation["mean_generated_tokens"]
        mean_seconds = generation["mean_seconds_per_sample"]
        lines.append(
            f"| {label} | {mean_tokens:.2f} | {mean_seconds:.3f} |"
            if mean_tokens is not None and mean_seconds is not None
            else f"| {label} | n/a | n/a |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two checkpoints on complete validation counterfactual pairs"
    )
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--baseline-label", default="step0")
    parser.add_argument("--candidate-label", default="step11193")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_path = args.raw_data.resolve()
    manifest_path = args.manifest.resolve()
    baseline_path = args.baseline_predictions.resolve()
    candidate_path = args.candidate_predictions.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_ids = {
        str(value) for value in manifest.get("validation_sample_ids", [])
    }
    if not validation_ids:
        raise ValueError("Manifest has no validation_sample_ids")
    raw_rows = _load_selected_raw_rows(raw_path, validation_ids)
    summary, pair_rows = compare(
        raw_rows=raw_rows,
        manifest=manifest,
        baseline_rows=_load_predictions(baseline_path),
        candidate_rows=_load_predictions(candidate_path),
        baseline_label=str(args.baseline_label),
        candidate_label=str(args.candidate_label),
    )
    summary["inputs"] = {
        "raw_data": {
            "path": str(raw_path),
            "sha256": str(manifest.get("source_sha256", "")) or _sha256(raw_path),
        },
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "baseline_predictions": {
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
        },
        "candidate_predictions": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
    }
    output_dir = args.output_dir.resolve()
    write_outputs(summary, pair_rows, output_dir)
    print(output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
