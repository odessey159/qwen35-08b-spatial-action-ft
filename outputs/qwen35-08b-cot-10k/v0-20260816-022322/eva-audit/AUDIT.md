# EVA 归因与泄漏审计

| 检查项 | 结果 | 判定 |
|:---|---:|:---|
| 并查集 component 跨 split | 0 | 通过 |
| 最大 component 占比 | 0.12% | 诊断值 |
| train/val scene_id 重叠 | 0 | 通过 |
| train/val CF group 重叠 | 0 | 通过 |
| 过滤后纯文本确定率 | 41.50% | 描述同一文本是否只有唯一答案 |
| train 拟合文本 oracle 准确率 | 72.60% | coverage 99.30%；视觉最大增益余量 27.40% |
| validation Bayes 文本 oracle | 74.00% | 分析上界；不可消除的视觉歧义 26.00% |

## 模型输出审计

| 输出轨 | 状态 | 可否用于当前 10K 归因 |
|:---|:---|:---|
| base structure / placeholder | available | 历史 Exp0，仅作诊断，不能与当前 val 直接作差 |
| format-only arm | missing | 不可以：分数缺失（arm 已实现） |
| full-CoT generation | missing | 不可以：当前 checkpoint 预测缺失 |
| CF pair-level | missing | 不可以：需 full-CoT 配对预测 |

```json
{
  "base_structure": {
    "status": "available",
    "prediction_path": "exp0\\outputs\\predictions.jsonl",
    "scope": "historical_exp0_not_current_10k_validation",
    "structure_contract": "legacy_plan",
    "prediction_count": 1680,
    "conditions": {
      "A": {
        "count": 240,
        "strict_structure_valid": 0.020833333333333332,
        "placeholder_copy_rate": 0.004166666666666667
      },
      "A_prime": {
        "count": 240,
        "strict_structure_valid": 0.004166666666666667,
        "placeholder_copy_rate": 0.004166666666666667
      },
      "B_json": {
        "count": 240,
        "strict_structure_valid": 0.004166666666666667,
        "placeholder_copy_rate": 0.0
      },
      "B_natural": {
        "count": 240,
        "strict_structure_valid": 0.0,
        "placeholder_copy_rate": 0.0
      },
      "B_triples": {
        "count": 240,
        "strict_structure_valid": 0.008333333333333333,
        "placeholder_copy_rate": 0.0
      },
      "C": {
        "count": 240,
        "strict_structure_valid": 0.24166666666666667,
        "placeholder_copy_rate": 0.24166666666666667
      },
      "D": {
        "count": 240,
        "strict_structure_valid": 0.5875,
        "placeholder_copy_rate": 0.6125
      }
    }
  },
  "format_only_structure": {
    "status": "missing",
    "required": true
  },
  "full_cot_structure": {
    "status": "missing",
    "required": true
  },
  "counterfactual_pairs": {
    "status": "missing",
    "required": true,
    "validation_cf_groups": 106,
    "complete_validation_cf_pairs": 106,
    "gold_discriminative_pairs": 106
  }
}
```

缺失的 format-only 或模型预测被标为 `required: true`；缺失时不能完成格式收益归因或 CF 配对能力结论。
