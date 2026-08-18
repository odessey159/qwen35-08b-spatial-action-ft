# 训练前后自由生成指标比较

比较范围：训练前后共同覆盖的 `200` 条 validation 样本，包含 `100` 个完整 CF pair、`53` 个场景。

所有指标均由同一评分器在同一批样本上重新计算；差值为训练后减训练前。

| 指标 | 训练前 | 训练后 | 差值 | 方向 |
|:---|---:|---:|---:|:---:|
| Action Sequence Exact | 0.00% | 93.00% | +93.00 pp | higher |
| CF Pair Exact | 0.00% | 88.00% | +88.00 pp | higher |
| Step Position Match Recall | 0.00% | 97.86% | +97.86 pp | higher |
| Step Position Match Precision | 0.00% | 96.22% | +96.22 pp | higher |
| State Fact Precision | 2.48% | 87.95% | +85.46 pp | higher |
| State Fact Recall | 0.37% | 89.60% | +89.22 pp | higher |
| State Fact F1 | 0.65% | 88.77% | +88.11 pp | higher |
| State Exact | 0.00% | 58.00% | +58.00 pp | higher |
| P(Action Exact | State Exact) | 0.00% | 100.00% | +100.00 pp | higher |
| P(Action Exact | State Wrong) | 0.00% | 83.33% | +83.33 pp | diagnostic |
| Strict Structure Valid | 0.00% | 100.00% | +100.00 pp | higher |
| Invalid Action Vocab | 0.00% | 0.00% | +0.00 pp | lower |
| Invalid Object Vocab | 0.00% | 0.00% | +0.00 pp | lower |
| Placeholder Copy | 0.00% | 0.00% | +0.00 pp | lower |
| Train-fitted Text Oracle | 44.50% | 44.50% | +0.00 pp | diagnostic |
| Text Oracle Coverage | 100.00% | 100.00% | +0.00 pp | diagnostic |

## 口径说明

- `CF Pair Exact` 的分母是完整反事实对；一对中的两个样本都 action exact 才计为正确。
- `Train-fitted Text Oracle` 与 `Text Oracle Coverage` 只由固定 train/validation split 决定，训练前后应完全相同。
- `P(Action Exact | State Exact)` 在没有任何 state-exact 样本时按评分器约定记为 0，而不是统计学意义上的可估计条件概率。
