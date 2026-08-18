# 三段 CoT Checkpoint EVA

- 验证样本：4 / 9951
- 抽样 seed：42
- 设备 / dtype：cpu / float32
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `all-causal diagnostic` 计算 `Σ(section weight × token CE) / 全部 causal token 数`，只用于对齐诊断，不是 trainer eval loss 的复刻；Swift 的 collator、loss_scale 与逐 batch 归约会改变实际口径。

## Checkpoint 汇总

| step | logged eval loss | all-causal diagnostic | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 0 | 0.428906 | 0.043755 | 1.105823 | 56.79 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | state | 0.3 | 177 | 1.328484 | 1.566222 | 0.397345 | 3.7753 | 0.7006 | 0.6525 | 0.0000 |
| 0 | plan | 0.3 | 67 | 1.150639 | 2.177664 | 0.266257 | 3.1602 | 0.7910 | 0.6774 | 0.0000 |
| 0 | action | 0.4 | 91 | 0.639736 | 1.029929 | 0.142241 | 1.8960 | 0.8681 | 0.8039 | 0.0000 |
