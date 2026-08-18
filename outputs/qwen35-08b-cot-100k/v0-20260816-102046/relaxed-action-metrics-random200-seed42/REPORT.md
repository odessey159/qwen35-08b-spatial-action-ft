# Relaxed semantic action metrics

评测样本：200 条。Strict 指标不变；本报告只增加 relaxed 指标。

| 指标 | 训练前 | 训练后 | 差值 |
|:---|---:|---:|---:|
| Relaxed Action Sequence EM | 0.00% | 92.50% | +92.50 pp |
| Relaxed Step Position Recall | 2.42% | 97.14% | +94.71 pp |
| Relaxed Step Position Precision | 5.82% | 96.29% | +90.47 pp |

## 审计诊断

| 指标 | 训练前 | 训练后 |
|:---|---:|---:|
| Relaxed Parse Rate | 73.00% | 100.00% |
| Deterministic Plan Rate | 9.50% | 100.00% |
| Relaxed Step F1 | 3.42% | 96.71% |

解析器不读取 instruction、gold state、图片或单样本 gold action 来补动作或参数。条件分支、否定动作和参数缺失不会形成 deterministic plan。
