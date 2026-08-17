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
| base structure / placeholder | missing | 历史 Exp0，仅作诊断，不能与当前 val 直接作差 |
| format-only arm | missing | 不可以：分数不完整（arm 已实现） |
| full-CoT generation | missing | 仅 pilot 或缺失：不能外推完整 val |
| CF pair-level | missing | 仅 pilot 或缺失：需覆盖全部 CF 对 |

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
    "status": "missing",
    "required": true,
    "expected_count": 9951
  },
  "counterfactual_pairs": {
    "status": "missing",
    "required": true,
    "validation_cf_groups": 1860,
    "complete_validation_cf_pairs": 1860,
    "gold_discriminative_pairs": 1860
  }
}
```

缺失的 format-only 或模型预测被标为 `required: true`；缺失时不能完成格式收益归因或 CF 配对能力结论。
