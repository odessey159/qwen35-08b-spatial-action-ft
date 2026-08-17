# 三段 CoT Checkpoint EVA

- 验证样本：9951 / 9951
- 抽样 seed：42
- 设备 / dtype：cuda / bfloat16
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `all-causal diagnostic` 计算 `Σ(section weight × token CE) / 全部 causal token 数`，只用于对齐诊断，不是 trainer eval loss 的复刻；Swift 的 collator、loss_scale 与逐 batch 归约会改变实际口径。

## Checkpoint 汇总

| step | logged eval loss | all-causal diagnostic | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 0 | 0.428906 | 0.049624 | 1.091738 | 0.12 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | state | 0.3 | 553738 | 1.287582 | 1.452780 | 0.431366 | 3.6240 | 0.6992 | 0.6711 | 0.0000 |
| 0 | plan | 0.3 | 187134 | 1.186445 | 2.000356 | 0.299685 | 3.2754 | 0.7810 | 0.6819 | 0.0000 |
| 0 | action | 0.4 | 247503 | 0.581972 | 0.876851 | 0.143424 | 1.7896 | 0.8736 | 0.8287 | 0.0000 |
