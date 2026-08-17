# 三段 CoT Checkpoint EVA

- 验证样本：1 / 1000
- 抽样 seed：42
- 设备 / dtype：CPU / float32
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `weighted approx` 按 ms-swift 的实际公式 `Σ(weight × token CE) / 全部监督 token 数` 计算。

## Checkpoint 汇总

| step | logged eval loss | weighted approx | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 2250 | 0.008684 | 0.089480 | 0.330221 | 31.95 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2250 | state | 0.3 | 71 | 0.425136 | 0.073596 | 2.569530 | 1.5298 | 0.9437 | 0.9836 | 0.0000 |
| 2250 | plan | 0.3 | 17 | 0.399968 | 0.000028 | 0.679927 | 1.4918 | 0.9412 | 1.0000 | 0.0000 |
| 2250 | action | 0.4 | 24 | 0.000027 | 0.000044 | 0.000003 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
