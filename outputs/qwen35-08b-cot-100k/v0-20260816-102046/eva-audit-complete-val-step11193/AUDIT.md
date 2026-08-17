# EVA 归因与泄漏审计

| 检查项 | 结果 | 判定 |
|:---|---:|:---|
| 并查集 component 跨 split | 0 | 通过 |
| 最大 component 占比 | 0.05% | 诊断值 |
| train/val scene_id 重叠 | 0 | 通过 |
| train/val CF group 重叠 | 0 | 通过 |
| 过滤后纯文本确定率 | 25.72% | 描述同一文本是否只有唯一答案 |
| train 拟合文本 oracle 准确率 | 62.54% | coverage 99.96%；视觉最大增益余量 37.46% |
| validation Bayes 文本 oracle | 62.92% | 分析上界；不可消除的视觉歧义 37.08% |

## 模型输出审计

| 输出轨 | 状态 | 可否用于当前 run 归因 |
|:---|:---|:---|
| base structure / placeholder | missing | scope: missing |
| format-only arm | missing | 不可以：分数不完整（arm 已实现） |
| full-CoT generation | available | 可以 |
| CF pair-level | available | 可以 |

```json
{
  "base_structure": {
    "status": "missing",
    "required": true
  },
  "format_only_structure": {
    "status": "missing",
    "required": true,
    "expected_count": 9951
  },
  "full_cot_structure": {
    "status": "available",
    "required": false,
    "prediction_path": "outputs/qwen35-08b-cot-100k/v0-20260816-102046/gpu-generation/val-all-step11193.jsonl",
    "scope": "current_100k_full_validation",
    "structure_contract": "state_plan_action",
    "prediction_count": 9951,
    "conditions": {
      "all": {
        "count": 9951,
        "strict_structure_valid": 1.0,
        "placeholder_copy_rate": 0.0
      }
    },
    "expected_count": 9951,
    "coverage": 1.0
  },
  "counterfactual_pairs": {
    "status": "available",
    "required": false,
    "prediction_path": "outputs/qwen35-08b-cot-100k/v0-20260816-102046/gpu-generation/val-all-step11193.jsonl",
    "validation_cf_groups": 1860,
    "complete_validation_cf_pairs": 1860,
    "evaluated_pairs": 1860,
    "pair_coverage": 1.0,
    "pair_exact_accuracy": 0.7913978494623656,
    "sample_exact_accuracy": 0.8701612903225806,
    "one_member_correct_pairs": 293,
    "one_member_correct_rate": 0.1575268817204301,
    "zero_members_correct_pairs": 95,
    "zero_members_correct_rate": 0.051075268817204304,
    "same_plan_rate": 0.439247311827957,
    "same_action_sequence_rate": 0.15053763440860216,
    "gold_discriminative_pairs": 1860,
    "definition": "A pair is correct only when both counterfactual members have exact primitive action sequences."
  }
}
```

缺失的 format-only 或模型预测被标为 `required: true`；缺失时不能完成格式收益归因或 CF 配对能力结论。
