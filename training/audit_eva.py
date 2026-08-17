from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from exp0.schema import normalize_action, parse_plan
from training.evaluate_section_losses import load_jsonl


PLACEHOLDER_PATTERNS = (
    re.compile(r"\bActionName\b", re.IGNORECASE),
    re.compile(r"\(\s*Object\s*\)", re.IGNORECASE),
    re.compile(r"动作名\s*\(\s*物体名\s*\)"),
)
PLAN_BLOCK = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.IGNORECASE | re.DOTALL)
STRICT_COT = re.compile(
    r"^\s*<state>\s*.+?\s*</state>\s*"
    r"<plan>\s*.+?\s*</plan>\s*"
    r"<action>\s*.+?\s*</action>\s*$",
    re.IGNORECASE | re.DOTALL,
)


class DisjointSet:
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


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("meta", row.get("metadata", row.get("_meta", {})))
    return value if isinstance(value, dict) else {}


def gold_actions(row: dict[str, Any]) -> tuple[str, ...]:
    value: Any = None
    if isinstance(row.get("gold"), dict):
        value = row["gold"].get("plan_actions")
    if value is None and isinstance(row.get("oracle"), dict):
        value = row["oracle"].get("actions")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{row.get('sample_id')}: no gold action list")
    return tuple(normalize_action(item) for item in value)


def _split_ids(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    train = {str(value) for value in manifest.get("train_sample_ids", [])}
    val = {str(value) for value in manifest.get("validation_sample_ids", [])}
    if not train or not val or train & val:
        raise ValueError("Manifest has invalid train/validation sample IDs")
    return train, val


def split_audit(
    rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    train_ids, val_ids = _split_ids(manifest)
    row_ids = [str(row.get("sample_id")) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Raw sample IDs are not unique")
    missing = (train_ids | val_ids) - set(row_ids)
    unassigned = set(row_ids) - (train_ids | val_ids)
    fields = [str(value) for value in manifest.get("split_group_fields", [])]
    if not fields:
        raise ValueError("Manifest has no split_group_fields")

    groups = DisjointSet(len(rows))
    first_seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        meta = metadata(row)
        for field in fields:
            value = meta.get(field)
            if value in (None, ""):
                continue
            key = (field, json.dumps(value, ensure_ascii=False, sort_keys=True))
            if key in first_seen:
                groups.union(index, first_seen[key])
            else:
                first_seen[key] = index
    components: dict[int, list[str]] = defaultdict(list)
    for index, sample_id in enumerate(row_ids):
        components[groups.find(index)].append(sample_id)
    component_sizes = sorted((len(value) for value in components.values()), reverse=True)
    crossing = [
        value
        for value in components.values()
        if any(sample_id in train_ids for sample_id in value)
        and any(sample_id in val_ids for sample_id in value)
    ]

    train_rows = [row for row in rows if str(row.get("sample_id")) in train_ids]
    val_rows = [row for row in rows if str(row.get("sample_id")) in val_ids]
    train_scenes = {str(metadata(row).get("scene_id")) for row in train_rows}
    val_scenes = {str(metadata(row).get("scene_id")) for row in val_rows}
    train_cf = {
        str(metadata(row).get("counterfactual_group"))
        for row in train_rows
        if metadata(row).get("counterfactual_group") not in (None, "")
    }
    val_cf = {
        str(metadata(row).get("counterfactual_group"))
        for row in val_rows
        if metadata(row).get("counterfactual_group") not in (None, "")
    }
    cf_counts = Counter(
        str(metadata(row).get("counterfactual_group"))
        for row in rows
        if metadata(row).get("counterfactual_group") not in (None, "")
    )
    unnamespaced = sorted(
        group for group in cf_counts if re.match(r"^shard_\d+_", group) is None
    )
    return {
        "sample_count": len(rows),
        "manifest_train_count": len(train_ids),
        "manifest_validation_count": len(val_ids),
        "missing_manifest_ids": sorted(missing)[:20],
        "unassigned_raw_ids": sorted(unassigned)[:20],
        "split_group_fields": fields,
        "component_count": len(components),
        "largest_component_size": component_sizes[0],
        "largest_component_ratio": component_sizes[0] / len(rows),
        "component_size_p95": component_sizes[max(0, round(len(component_sizes) * 0.05) - 1)],
        "cross_split_component_count": len(crossing),
        "cross_split_component_examples": crossing[:5],
        "merge_split_valid": not missing and not unassigned and not crossing,
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(val_scenes),
        "scene_overlap_count": len(train_scenes & val_scenes),
        "scene_overlap": sorted(train_scenes & val_scenes)[:50],
        "scene_disjoint": not (train_scenes & val_scenes),
        "counterfactual_group_overlap_count": len(train_cf & val_cf),
        "counterfactual_group_overlap": sorted(train_cf & val_cf)[:50],
        "counterfactual_groups": len(cf_counts),
        "incomplete_counterfactual_groups": {
            key: value for key, value in cf_counts.items() if value != 2
        },
        "unnamespaced_counterfactual_groups": unnamespaced[:50],
        "counterfactual_namespace_safe": not unnamespaced,
    }


def text_oracle_audit(
    rows: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    train_ids, val_ids = _split_ids(manifest)
    filtered = [
        row
        for row in rows
        if metadata(row).get("sim_verified") is True
        and metadata(row).get("target_visible") is not False
    ]
    all_targets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    train_counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for row in filtered:
        instruction = " ".join(str(row.get("instruction", "")).split()).casefold()
        target = gold_actions(row)
        all_targets[instruction].add(target)
        if str(row.get("sample_id")) in train_ids:
            train_counts[instruction][target] += 1

    val_rows = [row for row in filtered if str(row.get("sample_id")) in val_ids]
    deterministic = 0
    covered = 0
    train_majority_correct = 0
    val_counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for row in val_rows:
        instruction = " ".join(str(row.get("instruction", "")).split()).casefold()
        target = gold_actions(row)
        deterministic += int(len(all_targets[instruction]) == 1)
        val_counts[instruction][target] += 1
        if train_counts[instruction]:
            covered += 1
            prediction = train_counts[instruction].most_common(1)[0][0]
            train_majority_correct += int(prediction == target)
    bayes_correct = sum(max(counts.values()) for counts in val_counts.values())
    count = len(val_rows)
    train_oracle_accuracy = train_majority_correct / count if count else 0.0
    bayes_oracle_accuracy = bayes_correct / count if count else 0.0
    return {
        "filter": "sim_verified == true and target_visible != false",
        "filtered_validation_count": count,
        "text_deterministic_count": deterministic,
        "filtered_text_deterministic_rate": deterministic / count if count else 0.0,
        "train_text_oracle_coverage": covered / count if count else 0.0,
        "train_text_oracle_accuracy": train_oracle_accuracy,
        "train_text_oracle_accuracy_on_covered": (
            train_majority_correct / covered if covered else 0.0
        ),
        "validation_text_bayes_oracle_accuracy": bayes_oracle_accuracy,
        "visual_lift_headroom_vs_train_oracle": 1 - train_oracle_accuracy,
        "irreducible_visual_ambiguity_rate": 1 - bayes_oracle_accuracy,
        "definition": (
            "The train-fitted text oracle predicts the most frequent training action "
            "sequence for each exact normalized instruction. The validation Bayes "
            "oracle uses the best possible per-instruction majority on validation and "
            "is an analysis upper bound, not a deployable score."
        ),
    }


def has_placeholder_copy(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in PLACEHOLDER_PATTERNS)


def structure_audit(
    prediction_path: Path | None,
    cot_contract: bool,
    scope: str = "unspecified",
    expected_count: int | None = None,
) -> dict[str, Any]:
    if prediction_path is None or not prediction_path.is_file():
        result: dict[str, Any] = {"status": "missing", "required": True}
        if expected_count is not None:
            result["expected_count"] = expected_count
        return result
    rows = load_jsonl(prediction_path)
    by_condition: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        output = str(row.get("raw_output", row.get("prediction", row.get("output", ""))))
        condition = str(row.get("condition", "all"))
        parsed = parse_plan(output)
        structure_valid = (
            bool(STRICT_COT.fullmatch(output)) and parsed.structure_valid
            if cot_contract
            else parsed.structure_valid
        )
        by_condition[condition].append(
            {
                "strict_structure_valid": float(structure_valid),
                "placeholder_copy": float(has_placeholder_copy(output)),
            }
        )
    status = "available" if expected_count is None or len(rows) >= expected_count else "partial"
    result = {
        "status": status,
        "required": status != "available",
        "prediction_path": str(prediction_path),
        "scope": scope,
        "structure_contract": "state_plan_action" if cot_contract else "legacy_plan",
        "prediction_count": len(rows),
        "conditions": {
            condition: {
                "count": len(values),
                "strict_structure_valid": sum(v["strict_structure_valid"] for v in values)
                / len(values),
                "placeholder_copy_rate": sum(v["placeholder_copy"] for v in values)
                / len(values),
            }
            for condition, values in sorted(by_condition.items())
        },
    }
    if expected_count is not None:
        result["expected_count"] = expected_count
        result["coverage"] = len(rows) / expected_count if expected_count else 0.0
    return result


def _plan_text(output: str) -> str:
    match = PLAN_BLOCK.search(output)
    return " ".join(match.group(1).split()).casefold() if match else ""


def counterfactual_pair_audit(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    prediction_path: Path | None,
) -> dict[str, Any]:
    _, val_ids = _split_ids(manifest)
    raw_by_id = {str(row.get("sample_id")): row for row in rows}
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in val_ids:
        row = raw_by_id[sample_id]
        group = metadata(row).get("counterfactual_group")
        if group not in (None, ""):
            groups[str(group)].append(sample_id)
    complete = {group: ids for group, ids in groups.items() if len(ids) == 2}
    gold_discriminative_total = sum(
        int(gold_actions(raw_by_id[ids[0]]) != gold_actions(raw_by_id[ids[1]]))
        for ids in complete.values()
    )
    if prediction_path is None or not prediction_path.is_file():
        return {
            "status": "missing",
            "required": True,
            "validation_cf_groups": len(groups),
            "complete_validation_cf_pairs": len(complete),
            "gold_discriminative_pairs": gold_discriminative_total,
        }
    predictions = {
        str(row.get("sample_id")): str(
            row.get("raw_output", row.get("prediction", row.get("output", "")))
        )
        for row in load_jsonl(prediction_path)
    }
    evaluated = 0
    pair_exact = 0
    member_exact = 0
    one_member_correct = 0
    zero_members_correct = 0
    same_plan = 0
    same_actions = 0
    gold_discriminative = 0
    for ids in complete.values():
        if any(sample_id not in predictions for sample_id in ids):
            continue
        evaluated += 1
        gold = [gold_actions(raw_by_id[sample_id]) for sample_id in ids]
        outputs = [predictions[sample_id] for sample_id in ids]
        predicted_actions = [tuple(parse_plan(output).actions) for output in outputs]
        matches = [predicted == target for predicted, target in zip(predicted_actions, gold)]
        correct_members = sum(matches)
        pair_exact += int(correct_members == 2)
        member_exact += correct_members
        one_member_correct += int(correct_members == 1)
        zero_members_correct += int(correct_members == 0)
        same_plan += int(_plan_text(outputs[0]) == _plan_text(outputs[1]))
        same_actions += int(predicted_actions[0] == predicted_actions[1])
        gold_discriminative += int(gold[0] != gold[1])
    return {
        "status": "available" if evaluated == len(complete) else "partial",
        "required": evaluated != len(complete),
        "prediction_path": str(prediction_path),
        "validation_cf_groups": len(groups),
        "complete_validation_cf_pairs": len(complete),
        "evaluated_pairs": evaluated,
        "pair_coverage": evaluated / len(complete) if complete else 0.0,
        "pair_exact_accuracy": pair_exact / evaluated if evaluated else 0.0,
        "sample_exact_accuracy": member_exact / (2 * evaluated) if evaluated else 0.0,
        "one_member_correct_pairs": one_member_correct,
        "one_member_correct_rate": one_member_correct / evaluated if evaluated else 0.0,
        "zero_members_correct_pairs": zero_members_correct,
        "zero_members_correct_rate": zero_members_correct / evaluated if evaluated else 0.0,
        "same_plan_rate": same_plan / evaluated if evaluated else 0.0,
        "same_action_sequence_rate": same_actions / evaluated if evaluated else 0.0,
        "gold_discriminative_pairs": gold_discriminative,
        "definition": "A pair is correct only when both counterfactual members have exact primitive action sequences.",
    }


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    split = result["split_audit"]
    oracle = result["text_oracle"]
    lines = [
        "# EVA 归因与泄漏审计",
        "",
        "| 检查项 | 结果 | 判定 |",
        "|:---|---:|:---|",
        f"| 并查集 component 跨 split | {split['cross_split_component_count']} | {'通过' if split['merge_split_valid'] else '失败'} |",
        f"| 最大 component 占比 | {split['largest_component_ratio']:.2%} | 诊断值 |",
        f"| train/val scene_id 重叠 | {split['scene_overlap_count']} | {'通过' if split['scene_disjoint'] else '失败'} |",
        f"| train/val CF group 重叠 | {split['counterfactual_group_overlap_count']} | {'通过' if split['counterfactual_group_overlap_count'] == 0 else '失败'} |",
        f"| 过滤后纯文本确定率 | {oracle['filtered_text_deterministic_rate']:.2%} | 描述同一文本是否只有唯一答案 |",
        f"| train 拟合文本 oracle 准确率 | {oracle['train_text_oracle_accuracy']:.2%} | coverage {oracle['train_text_oracle_coverage']:.2%}；视觉最大增益余量 {oracle['visual_lift_headroom_vs_train_oracle']:.2%} |",
        f"| validation Bayes 文本 oracle | {oracle['validation_text_bayes_oracle_accuracy']:.2%} | 分析上界；不可消除的视觉歧义 {oracle['irreducible_visual_ambiguity_rate']:.2%} |",
        "",
        "## 模型输出审计",
        "",
        "| 输出轨 | 状态 | 可否用于当前 run 归因 |",
        "|:---|:---|:---|",
        f"| base structure / placeholder | {result['base_structure']['status']} | scope: {result['base_structure'].get('scope', 'missing')} |",
        f"| format-only arm | {result['format_only_structure']['status']} | {'可以' if result['format_only_structure']['status'] == 'available' else '不可以：分数不完整（arm 已实现）'} |",
        f"| full-CoT generation | {result['full_cot_structure']['status']} | {'可以' if result['full_cot_structure']['status'] == 'available' else '仅 pilot 或缺失：不能外推完整 val'} |",
        f"| CF pair-level | {result['counterfactual_pairs']['status']} | {'可以' if result['counterfactual_pairs']['status'] == 'available' else '仅 pilot 或缺失：需覆盖全部 CF 对'} |",
        "",
        "```json",
        json.dumps(
            {
                "base_structure": result["base_structure"],
                "format_only_structure": result["format_only_structure"],
                "full_cot_structure": result["full_cot_structure"],
                "counterfactual_pairs": result["counterfactual_pairs"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "缺失的 format-only 或模型预测被标为 `required: true`；缺失时不能完成格式收益归因或 CF 配对能力结论。",
    ]
    (output_dir / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit EVA attribution and split leakage")
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--base-scope", default="unspecified")
    parser.add_argument("--base-cot-contract", action="store_true")
    parser.add_argument("--current-scope", default="current_validation")
    parser.add_argument("--format-only-predictions", type=Path)
    parser.add_argument("--full-cot-predictions", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_jsonl(args.raw_data.resolve())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _, validation_ids = _split_ids(manifest)
    result = {
        "split_audit": split_audit(rows, manifest),
        "text_oracle": text_oracle_audit(rows, manifest),
        "base_structure": structure_audit(
            args.base_predictions,
            cot_contract=args.base_cot_contract,
            scope=args.base_scope,
        ),
        "format_only_structure": structure_audit(
            args.format_only_predictions,
            cot_contract=True,
            scope=args.current_scope,
            expected_count=len(validation_ids),
        ),
        "full_cot_structure": structure_audit(
            args.full_cot_predictions,
            cot_contract=True,
            scope=args.current_scope,
            expected_count=len(validation_ids),
        ),
        "counterfactual_pairs": counterfactual_pair_audit(
            rows, manifest, args.full_cot_predictions
        ),
    }
    write_report(result, args.output_dir.resolve())
    print(args.output_dir.resolve() / "AUDIT.md")


if __name__ == "__main__":
    main()
