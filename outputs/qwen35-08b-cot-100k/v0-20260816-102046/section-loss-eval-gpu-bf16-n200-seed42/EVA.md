# 三段 CoT Checkpoint EVA

- 验证样本：200 / 9951
- 抽样 seed：42
- 设备 / dtype：cuda / bfloat16
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `all-causal diagnostic` 计算 `Σ(section weight × token CE) / 全部 causal token 数`，只用于对齐诊断，不是 trainer eval loss 的复刻；Swift 的 collator、loss_scale 与逐 batch 归约会改变实际口径。

## Checkpoint 汇总

| step | logged eval loss | all-causal diagnostic | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 0 | 0.428906 | 0.048814 | 1.101163 | 0.13 |
| 11193 | 0.005917 | 0.011617 | 0.273972 | 0.12 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | state | 0.3 | 10503 | 1.317189 | 1.502708 | 0.420204 | 3.7329 | 0.6963 | 0.6644 | 0.0000 |
| 0 | plan | 0.3 | 3758 | 1.194791 | 2.023518 | 0.293320 | 3.3029 | 0.7797 | 0.6793 | 0.0000 |
| 0 | action | 0.4 | 5009 | 0.577951 | 0.865746 | 0.144964 | 1.7824 | 0.8734 | 0.8295 | 0.0000 |
| 11193 | state | 0.3 | 10503 | 0.502646 | 0.036193 | 2.757947 | 1.6531 | 0.9484 | 0.9855 | 0.0000 |
| 11193 | plan | 0.3 | 3758 | 0.000021 | 0.000027 | 0.000013 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 11193 | action | 0.4 | 5009 | 0.000015 | 0.000016 | 0.000013 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
