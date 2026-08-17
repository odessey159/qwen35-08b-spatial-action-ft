# 三段 CoT Checkpoint EVA

- 验证样本：8 / 1000
- 抽样 seed：42
- 设备 / dtype：CPU / float32
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `weighted approx` 按 ms-swift 的实际公式 `Σ(weight × token CE) / 全部 causal token 数` 计算；system/user/image/空 think 的权重为 0，但仍进入分母。

## Checkpoint 汇总

| step | logged eval loss | weighted approx | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 2250 | 0.008684 | 0.008705 | 0.193677 | 20.03 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2250 | state | 0.3 | 474 | 0.337050 | 0.036584 | 2.014654 | 1.4008 | 0.9515 | 0.9876 | 0.0000 |
| 2250 | plan | 0.3 | 148 | 0.000067 | 0.000095 | 0.000038 | 1.0001 | 1.0000 | 1.0000 | 1.0000 |
| 2250 | action | 0.4 | 203 | 0.000057 | 0.000081 | 0.000020 | 1.0001 | 1.0000 | 1.0000 | 1.0000 |
