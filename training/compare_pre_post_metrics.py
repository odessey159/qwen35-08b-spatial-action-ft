from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from training.audit_eva import metadata
from training.evaluate_in_domain_predictions import evaluate
from training.evaluate_section_losses import load_jsonl


METRICS: tuple[tuple[str, str, str], ...] = (
    ("Action Sequence Exact", "action_sequence_exact", "higher"),
    ("CF Pair Exact", "cf_pair_exact", "higher"),
    ("Step Position Match Recall", "step_position_match_recall", "higher"),
    ("Step Position Match Precision", "step_position_match_precision", "higher"),
    ("State Fact Precision", "state_fact_precision", "higher"),
    ("State Fact Recall", "state_fact_recall", "higher"),
    ("State Fact F1", "state_fact_f1", "higher"),
    ("State Exact", "state_exact", "higher"),
    ("P(Action Exact | State Exact)", "p_action_exact_given_state_exact", "higher"),
    ("P(Action Exact | State Wrong)", "p_action_exact_given_state_wrong", "diagnostic"),
    ("Strict Structure Valid", "strict_structure_valid", "higher"),
    ("Invalid Action Vocab", "invalid_action_vocab_rate", "lower"),
    ("Invalid Object Vocab", "invalid_object_vocab_rate", "lower"),
    ("Placeholder Copy", "placeholder_copy_rate", "lower"),
    ("Train-fitted Text Oracle", "text_oracle_action_exact", "diagnostic"),
    ("Text Oracle Coverage", "text_oracle_coverage", "diagnostic"),
)


def _prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"Prediction without sample_id: {path}")
        if sample_id in result:
            raise ValueError(f"Duplicate prediction for {sample_id}: {path}")
        result[sample_id] = row
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_values(result: dict[str, Any]) -> dict[str, float]:
    overall = result["correct_image"]["slices"]["all"]
    values = {
        key: float(overall[key])
        for _, key, _ in METRICS
        if key != "cf_pair_exact"
    }
    values["cf_pair_exact"] = float(
        result["correct_image"]["counterfactual_pairs"]["pair_exact"]
    )
    return values


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _signed_pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _write_report(comparison: dict[str, Any], output_dir: Path) -> None:
    scope = comparison["scope"]
    lines = [
        "# 训练前后自由生成指标比较",
        "",
        (
            f"比较范围：训练前后共同覆盖的 `{scope['samples']}` 条 validation 样本，"
            f"包含 `{scope['counterfactual_pairs']}` 个完整 CF pair、"
            f"`{scope['scenes']}` 个场景。"
        ),
        "",
        "所有指标均由同一评分器在同一批样本上重新计算；差值为训练后减训练前。",
        "",
        "| 指标 | 训练前 | 训练后 | 差值 | 方向 |",
        "|:---|---:|---:|---:|:---:|",
    ]
    for item in comparison["metrics"]:
        lines.append(
            f"| {item['name']} | {_percent(item['before'])} | "
            f"{_percent(item['after'])} | {_signed_pp(item['delta'])} | "
            f"{item['direction']} |"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- `CF Pair Exact` 的分母是完整反事实对；一对中的两个样本都 action exact 才计为正确。",
            "- `Train-fitted Text Oracle` 与 `Text Oracle Coverage` 只由固定 train/validation split 决定，训练前后应完全相同。",
            "- `P(Action Exact | State Exact)` 在没有任何 state-exact 样本时按评分器约定记为 0，而不是统计学意义上的可估计条件概率。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(
    raw_path: Path,
    prepared_path: Path,
    manifest_path: Path,
    before_path: Path,
    after_path: Path,
) -> dict[str, Any]:
    raw_rows = load_jsonl(raw_path)
    prepared_rows = load_jsonl(prepared_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids", [])]
    if len(validation_ids) != len(prepared_rows):
        raise ValueError("Prepared validation rows do not align with manifest IDs")

    before_all = _prediction_map(before_path)
    after_all = _prediction_map(after_path)
    common_ids = [
        sample_id
        for sample_id in validation_ids
        if sample_id in before_all and sample_id in after_all
    ]
    if not common_ids:
        raise ValueError("The before/after predictions have no common validation samples")

    prepared_by_id = dict(zip(validation_ids, prepared_rows))
    subset_manifest = dict(manifest)
    subset_manifest["validation_sample_ids"] = common_ids
    subset_prepared = [prepared_by_id[sample_id] for sample_id in common_ids]
    train_ids = {str(value) for value in manifest.get("train_sample_ids", [])}
    oracle_training_rows = [
        row for row in raw_rows if str(row.get("sample_id")) in train_ids
    ]
    if len(oracle_training_rows) != len(train_ids):
        raise ValueError("Raw data do not cover every training ID needed by the text oracle")

    before_result = evaluate(
        raw_rows=raw_rows,
        prepared_rows=subset_prepared,
        manifest=subset_manifest,
        prediction_rows={sample_id: before_all[sample_id] for sample_id in common_ids},
        oracle_training_rows=oracle_training_rows,
    )
    after_result = evaluate(
        raw_rows=raw_rows,
        prepared_rows=subset_prepared,
        manifest=subset_manifest,
        prediction_rows={sample_id: after_all[sample_id] for sample_id in common_ids},
        oracle_training_rows=oracle_training_rows,
    )
    before_values = _metric_values(before_result)
    after_values = _metric_values(after_result)
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    scenes = {
        str(metadata(raw_by_id[sample_id]).get("scene_id", ""))
        for sample_id in common_ids
    }
    return {
        "scope": {
            "samples": len(common_ids),
            "validation_samples": len(validation_ids),
            "coverage": len(common_ids) / len(validation_ids),
            "scenes": len(scenes),
            "counterfactual_pairs": before_result["correct_image"][
                "counterfactual_pairs"
            ]["pairs"],
            "selection": "ordered intersection of before and after prediction IDs",
        },
        "metrics": [
            {
                "name": name,
                "key": key,
                "direction": direction,
                "before": before_values[key],
                "after": after_values[key],
                "delta": after_values[key] - before_values[key],
            }
            for name, key, direction in METRICS
        ],
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in (
                ("raw_data", raw_path),
                ("prepared_validation", prepared_path),
                ("manifest", manifest_path),
                ("before_predictions", before_path),
                ("after_predictions", after_path),
            )
        },
        "before_evaluation": before_result,
        "after_evaluation": after_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare before/after generations on their common validation samples"
    )
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--before-predictions", type=Path, required=True)
    parser.add_argument("--after-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare(
        raw_path=args.raw_data.resolve(),
        prepared_path=args.val_file.resolve(),
        manifest_path=args.manifest.resolve(),
        before_path=args.before_predictions.resolve(),
        after_path=args.after_predictions.resolve(),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("name", "key", "direction", "before", "after", "delta"),
        )
        writer.writeheader()
        writer.writerows(result["metrics"])
    _write_report(result, output_dir)
    print(output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
