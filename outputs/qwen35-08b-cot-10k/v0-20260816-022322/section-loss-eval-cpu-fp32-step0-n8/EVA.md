# 三段 CoT Checkpoint EVA

- 验证样本：8 / 1000
- 抽样 seed：42
- 设备 / dtype：CPU / float32
- `full loss` 包含 XML 标签、换行及回合结束 token；`body loss` 只统计标签内部正文；`format loss` 是两者之差对应的结构 token。
- `weighted approx` 按 ms-swift 的实际公式 `Σ(weight × token CE) / 全部 causal token 数` 计算；system/user/image/空 think 的权重为 0，但仍进入分母。

## Checkpoint 汇总

| step | logged eval loss | weighted approx | raw micro loss | seconds/sample |
|---:|---:|---:|---:|---:|
| 0 | 0.438532 | 0.049365 | 1.053449 | 17.68 |

## 分段详细指标

| step | section | weight | tokens | full loss | body loss | format loss | full ppl | full token acc | body token acc | exact match |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | state | 0.3 | 474 | 1.245616 | 1.383162 | 0.477653 | 3.4751 | 0.7194 | 0.6940 | 0.0000 |
| 0 | plan | 0.3 | 148 | 1.131371 | 1.928163 | 0.290312 | 3.0999 | 0.8243 | 0.7500 | 0.0000 |
| 0 | action | 0.4 | 203 | 0.547934 | 0.819648 | 0.130173 | 1.7297 | 0.8670 | 0.8211 | 0.0000 |
