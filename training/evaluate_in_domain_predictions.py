from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from exp0.schema import normalize_action, parse_action, parse_plan
from training.audit_eva import STRICT_COT, has_placeholder_copy, metadata
from training.compare_counterfactual_predictions import clustered_bootstrap_interval
from training.evaluate_section_losses import load_jsonl


STATE_BLOCK = re.compile(r"<state>\s*(.*?)\s*</state>", re.IGNORECASE | re.DOTALL)
ALLOWED_ACTIONS = {
    "GotoLocation",
    "PickupObject",
    "PutObject",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "ToggleObject",
    "OpenObject",
    "CloseObject",
}


def prediction_text(row: dict[str, Any]) -> str:
    return str(row.get("raw_output", row.get("prediction", row.get("output", ""))))


def gold_actions(row: dict[str, Any]) -> tuple[str, ...]:
    value: Any = None
    if isinstance(row.get("gold"), dict):
        value = row["gold"].get("plan_actions")
    if value is None and isinstance(row.get("oracle"), dict):
        value = row["oracle"].get("actions")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{row.get('sample_id')}: no gold action sequence")
    return tuple(normalize_action(item) for item in value)


def _normalized_instruction(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("instruction", "")).split()).casefold()


def _normalized_facts(text: str) -> set[str]:
    match = STATE_BLOCK.search(text)
    if match is None:
        return set()
    result: set[str] = set()
    for line in match.group(1).splitlines():
        value = " ".join(line.split()).strip().rstrip(".。").casefold()
        if value:
            result.add(value)
    return result


def _prepared_gold_facts(row: dict[str, Any]) -> set[str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Prepared validation row has no state assistant message")
    return _normalized_facts(str(messages[2].get("content", "")))


def _load_predictions(path: Path, required_ids: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Prediction without sample_id: {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate prediction for {sample_id}: {path}")
        if sample_id in required_ids:
            result[sample_id] = row
    missing = sorted(required_ids - set(result))
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} validation rows: {missing[:10]}")
    return result


def _target_object(actions: tuple[str, ...]) -> str:
    for action in reversed(actions):
        parsed = parse_action(action)
        if parsed is not None and parsed[1]:
            return parsed[1][-1]
    return "unknown"


def _slice_names(row: dict[str, Any], actions: tuple[str, ...]) -> list[str]:
    meta = metadata(row)
    group = meta.get("counterfactual_group")
    result = [
        "all",
        f"cf:{'yes' if group not in (None, '') else 'no'}",
        f"task_group:{meta.get('task_group', 'unknown')}",
        f"plan_length:{meta.get('plan_length', len(actions))}",
    ]
    state = meta.get("receptacle_state")
    if state not in (None, ""):
        result.append(f"receptacle_state:{state}")
    if group not in (None, ""):
        result.append(f"container_type:{_target_object(actions)}")
    return result


def _text_oracle_predictions(
    training_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, tuple[str, ...]]:
    train_counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for row in training_rows:
        if metadata(row).get("sim_verified") is not True or metadata(row).get("target_visible") is False:
            continue
        train_counts[_normalized_instruction(row)][gold_actions(row)] += 1
    return {
        str(row.get("sample_id")): counts.most_common(1)[0][0]
        for row in validation_rows
        for counts in [train_counts[_normalized_instruction(row)]]
        if counts
    }


def _score_one(
    raw: dict[str, Any],
    gold_facts: set[str],
    prediction: dict[str, Any],
    allowed_objects: set[str],
    oracle_actions: tuple[str, ...] | None,
) -> dict[str, Any]:
    text = prediction_text(prediction)
    parsed = parse_plan(text)
    predicted = tuple(parsed.actions)
    gold = gold_actions(raw)
    predicted_facts = _normalized_facts(text)
    fact_tp = len(predicted_facts & gold_facts)
    fact_fp = len(predicted_facts - gold_facts)
    fact_fn = len(gold_facts - predicted_facts)
    parsed_lines = [parse_action(action) for action in predicted]
    names = [value[0] for value in parsed_lines if value is not None]
    arguments = [argument for value in parsed_lines if value is not None for argument in value[1]]
    return {
        "sample_id": str(raw["sample_id"]),
        "scene_id": str(metadata(raw).get("scene_id", "")),
        "counterfactual_group": str(metadata(raw).get("counterfactual_group") or ""),
        "slices": _slice_names(raw, gold),
        "action_exact": predicted == gold,
        "oracle_action_exact": oracle_actions == gold if oracle_actions is not None else False,
        "oracle_covered": oracle_actions is not None,
        "position_steps_correct": sum(left == right for left, right in zip(predicted, gold)),
        "gold_steps": len(gold),
        "predicted_steps": len(predicted),
        "state_exact": predicted_facts == gold_facts,
        "fact_tp": fact_tp,
        "fact_fp": fact_fp,
        "fact_fn": fact_fn,
        "strict_structure_valid": bool(STRICT_COT.fullmatch(text)) and parsed.structure_valid,
        "invalid_action_vocab": any(name not in ALLOWED_ACTIONS for name in names),
        "invalid_object_vocab": any(argument not in allowed_objects for argument in arguments),
        "placeholder_copy": has_placeholder_copy(text),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    action = [bool(record["action_exact"]) for record in records]
    scenes = [str(record["scene_id"]) for record in records]
    fact_tp = sum(int(record["fact_tp"]) for record in records)
    fact_fp = sum(int(record["fact_fp"]) for record in records)
    fact_fn = sum(int(record["fact_fn"]) for record in records)
    precision = _ratio(fact_tp, fact_tp + fact_fp)
    recall = _ratio(fact_tp, fact_tp + fact_fn)
    state_correct = [record for record in records if record["state_exact"]]
    state_wrong = [record for record in records if not record["state_exact"]]
    oracle_covered = [record for record in records if record["oracle_covered"]]
    return {
        "count": len(records),
        "scene_count": len(set(scenes)),
        "action_sequence_exact": _ratio(sum(action), len(action)),
        "action_sequence_exact_ci95": clustered_bootstrap_interval(action, scenes),
        "step_position_match_recall": _ratio(
            sum(int(record["position_steps_correct"]) for record in records),
            sum(int(record["gold_steps"]) for record in records),
        ),
        "step_position_match_precision": _ratio(
            sum(int(record["position_steps_correct"]) for record in records),
            sum(int(record["predicted_steps"]) for record in records),
        ),
        "state_fact_precision": precision,
        "state_fact_recall": recall,
        "state_fact_f1": _ratio(2 * precision * recall, precision + recall),
        "state_exact": _ratio(len(state_correct), len(records)),
        "p_action_exact_given_state_exact": _ratio(
            sum(record["action_exact"] for record in state_correct), len(state_correct)
        ),
        "p_action_exact_given_state_wrong": _ratio(
            sum(record["action_exact"] for record in state_wrong), len(state_wrong)
        ),
        "strict_structure_valid": _ratio(
            sum(record["strict_structure_valid"] for record in records), len(records)
        ),
        "invalid_action_vocab_rate": _ratio(
            sum(record["invalid_action_vocab"] for record in records), len(records)
        ),
        "invalid_object_vocab_rate": _ratio(
            sum(record["invalid_object_vocab"] for record in records), len(records)
        ),
        "placeholder_copy_rate": _ratio(
            sum(record["placeholder_copy"] for record in records), len(records)
        ),
        "text_oracle_coverage": _ratio(len(oracle_covered), len(records)),
        "text_oracle_action_exact": _ratio(
            sum(record["oracle_action_exact"] for record in records), len(records)
        ),
    }


def _summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for name in record["slices"]:
            by_slice[name].append(record)
    return {name: _summarize(values) for name, values in sorted(by_slice.items())}


def _pair_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["counterfactual_group"]:
            groups[str(record["counterfactual_group"])].append(record)
    incomplete = {name: len(values) for name, values in groups.items() if len(values) != 2}
    if incomplete:
        raise ValueError(f"Incomplete CF prediction groups: {dict(list(sorted(incomplete.items()))[:10])}")
    pair_exact = [all(value["action_exact"] for value in values) for values in groups.values()]
    oracle_pair_exact = [
        all(value["oracle_covered"] and value["oracle_action_exact"] for value in values)
        for values in groups.values()
    ]
    scenes = [str(values[0]["scene_id"]) for values in groups.values()]
    return {
        "pairs": len(pair_exact),
        "pair_exact": _ratio(sum(pair_exact), len(pair_exact)),
        "pair_exact_ci95": clustered_bootstrap_interval(pair_exact, scenes),
        "text_oracle_pair_exact": _ratio(sum(oracle_pair_exact), len(oracle_pair_exact)),
    }


def evaluate(
    raw_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    prediction_rows: dict[str, dict[str, Any]],
    a_prime_rows: dict[str, dict[str, Any]] | None = None,
    oracle_training_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids", [])]
    if len(validation_ids) != len(prepared_rows):
        raise ValueError("Prepared validation rows do not align with manifest IDs")
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    missing_raw = [sample_id for sample_id in validation_ids if sample_id not in raw_by_id]
    if missing_raw:
        raise ValueError(f"Raw validation rows are missing: {missing_raw[:10]}")
    object_source_rows = raw_rows + (oracle_training_rows or [])
    allowed_objects = {
        argument
        for row in object_source_rows
        for action in gold_actions(row)
        for parsed in [parse_action(action)]
        if parsed is not None
        for argument in parsed[1]
    }
    validation_rows = [raw_by_id[sample_id] for sample_id in validation_ids]
    if oracle_training_rows is None:
        validation_id_set = set(validation_ids)
        oracle_training_rows = [
            row for row in raw_rows if str(row.get("sample_id")) not in validation_id_set
        ]
    oracle_by_id = _text_oracle_predictions(oracle_training_rows, validation_rows)
    gold_facts_by_id = {
        sample_id: _prepared_gold_facts(prepared)
        for sample_id, prepared in zip(validation_ids, prepared_rows)
    }

    records = [
        _score_one(
            raw_by_id[sample_id],
            gold_facts_by_id[sample_id],
            prediction_rows[sample_id],
            allowed_objects,
            oracle_by_id.get(sample_id),
        )
        for sample_id in validation_ids
    ]
    result: dict[str, Any] = {
        "scope": {
            "validation_samples": len(validation_ids),
            "validation_scenes": len({record["scene_id"] for record in records}),
            "ci_method": "scene_cluster_percentile_bootstrap",
            "bootstrap_resamples": 1000,
        },
        "correct_image": {
            "slices": _summaries(records),
            "counterfactual_pairs": _pair_summary(records),
        },
    }
    if a_prime_rows is not None:
        a_prime_records = [
            _score_one(
                raw_by_id[sample_id],
                gold_facts_by_id[sample_id],
                a_prime_rows[sample_id],
                allowed_objects,
                oracle_by_id.get(sample_id),
            )
            for sample_id in validation_ids
        ]
        differences = [
            float(left["action_exact"]) - float(right["action_exact"])
            for left, right in zip(records, a_prime_records)
        ]
        result["a_prime"] = {
            "slices": _summaries(a_prime_records),
            "counterfactual_pairs": _pair_summary(a_prime_records),
        }
        result["visual_contribution"] = {
            "action_exact_a_minus_a_prime": sum(differences) / len(differences),
            "action_exact_a_minus_a_prime_ci95": clustered_bootstrap_interval(
                differences, [record["scene_id"] for record in records]
            ),
        }
    return result


def _percent(value: float) -> str:
    return f"{value:.2%}"


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    correct = result["correct_image"]
    overall = correct["slices"]["all"]
    pair = correct["counterfactual_pairs"]
    lines = [
        "# CoT-100K in-domain 自由生成评测",
        "",
        "95% CI 使用按 `scene_id` 重采样的 1000 次 percentile cluster bootstrap。",
        "",
        "## 主指标",
        "",
        "| 指标 | A（正确图） | A′（换图） |",
        "|:---|---:|---:|",
        f"| action sequence exact | {_percent(overall['action_sequence_exact'])} | "
        + (
            _percent(result["a_prime"]["slices"]["all"]["action_sequence_exact"])
            if "a_prime" in result
            else "运行中"
        )
        + " |",
        f"| CF pair exact | {_percent(pair['pair_exact'])} | "
        + (
            _percent(result["a_prime"]["counterfactual_pairs"]["pair_exact"])
            if "a_prime" in result
            else "运行中"
        )
        + " |",
        f"| step position match recall | {_percent(overall['step_position_match_recall'])} | "
        + (
            _percent(result["a_prime"]["slices"]["all"]["step_position_match_recall"])
            if "a_prime" in result
            else "运行中"
        )
        + " |",
    ]
    if "visual_contribution" in result:
        lift = result["visual_contribution"]
        lines.extend(
            [
                "",
                f"A − A′ action exact = **{lift['action_exact_a_minus_a_prime']:+.2%}** "
                f"（95% CI {lift['action_exact_a_minus_a_prime_ci95'][0]:+.2%} 到 "
                f"{lift['action_exact_a_minus_a_prime_ci95'][1]:+.2%}）。",
            ]
        )
    lines.extend(
        [
            "",
            "## 感知 / 规划归因与格式健康度（A）",
            "",
            "| 指标 | 数值 |",
            "|:---|---:|",
            f"| state fact P / R / F1 | {_percent(overall['state_fact_precision'])} / {_percent(overall['state_fact_recall'])} / {_percent(overall['state_fact_f1'])} |",
            f"| P(action 对 \\| state exact) | {_percent(overall['p_action_exact_given_state_exact'])} |",
            f"| P(action 对 \\| state 错) | {_percent(overall['p_action_exact_given_state_wrong'])} |",
            f"| strict structure valid | {_percent(overall['strict_structure_valid'])} |",
            f"| action vocab 越界率 | {_percent(overall['invalid_action_vocab_rate'])} |",
            f"| object vocab 越界率 | {_percent(overall['invalid_object_vocab_rate'])} |",
            f"| placeholder copy | {_percent(overall['placeholder_copy_rate'])} |",
            f"| 同切片 train-fitted text oracle | {_percent(overall['text_oracle_action_exact'])}（coverage {_percent(overall['text_oracle_coverage'])}） |",
            "",
            "## 必报切片（A）",
            "",
            "| 切片 | n | action exact | 95% CI | state F1 | text oracle |",
            "|:---|---:|---:|:---:|---:|---:|",
        ]
    )
    for name, values in correct["slices"].items():
        if name == "all":
            continue
        ci = values["action_sequence_exact_ci95"]
        lines.append(
            f"| {name} | {values['count']} | {_percent(values['action_sequence_exact'])} | "
            f"{_percent(ci[0])}–{_percent(ci[1])} | {_percent(values['state_fact_f1'])} | "
            f"{_percent(values['text_oracle_action_exact'])} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score full in-domain CoT generations")
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--a-prime-predictions", type=Path)
    parser.add_argument("--oracle-raw-data", type=Path)
    parser.add_argument("--oracle-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required_ids = {str(value) for value in manifest.get("validation_sample_ids", [])}
    if (args.oracle_raw_data is None) != (args.oracle_manifest is None):
        raise ValueError("--oracle-raw-data and --oracle-manifest must be provided together")
    oracle_training_rows: list[dict[str, Any]] | None = None
    if args.oracle_raw_data is not None and args.oracle_manifest is not None:
        oracle_manifest = json.loads(args.oracle_manifest.read_text(encoding="utf-8"))
        oracle_train_ids = {str(value) for value in oracle_manifest.get("train_sample_ids", [])}
        if not oracle_train_ids:
            raise ValueError("Oracle manifest has no train_sample_ids")
        oracle_training_rows = [
            row
            for row in load_jsonl(args.oracle_raw_data.resolve())
            if str(row.get("sample_id")) in oracle_train_ids
        ]
        if len(oracle_training_rows) != len(oracle_train_ids):
            raise ValueError("Oracle raw data does not cover all oracle training IDs")
    result = evaluate(
        raw_rows=load_jsonl(args.raw_data.resolve()),
        prepared_rows=load_jsonl(args.val_file.resolve()),
        manifest=manifest,
        prediction_rows=_load_predictions(args.predictions.resolve(), required_ids),
        a_prime_rows=(
            _load_predictions(args.a_prime_predictions.resolve(), required_ids)
            if args.a_prime_predictions is not None
            else None
        ),
        oracle_training_rows=oracle_training_rows,
    )
    output_dir = args.output_dir.resolve()
    write_outputs(result, output_dir)
    print(output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
