# 三段 CoT Checkpoint EVA

- 验证样本：1 / 1000
- 抽样 seed：42
- 设备 / dtype：CPU / float32
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `weighted approx` 按 ms-swift 的实际公式 `Σ(weight × token CE) / 全部监督 token 数` 计算。

## Checkpoint 汇总

| step | logged eval loss | weighted approx | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 2250 | 0.008684 | 0.060544 | 0.209149 | 7.56 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2250 | state | 0.3 | 70 | 0.328644 | 0.073596 | 2.057306 | 1.3891 | 0.9429 | 0.9836 | 0.0000 |
| 2250 | plan | 0.3 | 16 | 0.000043 | 0.000028 | 0.000055 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2250 | action | 0.4 | 24 | 0.000025 | 0.000030 | 0.000017 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
