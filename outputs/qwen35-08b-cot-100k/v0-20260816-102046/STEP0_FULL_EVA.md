# Step0 全量重算

重算日期：2026-08-17

## 评测范围

- teacher-forced 三段评估：完整 validation `9951 / 9951`，不再抽样。
- 自由生成评估：validation 中全部 `1860` 个完整反事实对，共 `3720` 条样本。
- 模型：原始 `Qwen3.5-0.8B`（step 0）。
- 计算环境：RTX 4090，CUDA，bfloat16。

## 全量三段指标

| section | tokens | full loss | body loss | format loss | full PPL | full token acc | body token acc | exact match |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| state | 553738 | 1.287582 | 1.452780 | 0.431366 | 3.6240 | 69.9237% | 67.1127% | 0 / 9951 |
| plan | 187134 | 1.186445 | 2.000356 | 0.299685 | 3.2754 | 78.1013% | 68.1916% | 0 / 9951 |
| action | 247503 | 0.581972 | 0.876851 | 0.143424 | 1.7896 | 87.3638% | 82.8701% | 0 / 9951 |

汇总：

- `raw micro loss = 1.091738`
- `weighted approx = 0.049624`
- trainer 日志中的完整 validation `eval_loss = 0.428906`
- 平均计算时间 `0.12 s/sample`

## 相对旧 n8 的变化

| 指标 | n8 | 全量 9951 | 变化 |
|:---|---:|---:|---:|
| weighted approx | 0.044410 | 0.049624 | +11.74% |
| raw micro loss | 1.078015 | 1.091738 | +1.27% |
| state full loss | 1.258938 | 1.287582 | +2.28% |
| plan full loss | 1.174779 | 1.186445 | +0.99% |
| action full loss | 0.596985 | 0.581972 | -2.51% |
| state full token acc | 71.2500% | 69.9237% | -1.3263 pp |
| plan full token acc | 79.5455% | 78.1013% | -1.4442 pp |
| action full token acc | 87.5706% | 87.3638% | -0.2068 pp |

n8 的总体方向没有颠倒，但它不足以支持精确数值，尤其是 plan/action body loss 和
`weighted approx`。后续应使用本次全量结果。

## Step0 全量反事实生成

| 指标 | 结果 |
|:---|---:|
| 预测样本 | 3720 / 3720 |
| 完整反事实对 | 1860 / 1860 |
| strict `<state><plan><action>` 合规 | 0.00% |
| sample exact | 0 / 3720（0.00%，95% CI 0.00%–0.10%） |
| pair exact | 0 / 1860（0.00%，95% CI 0.00%–0.21%） |
| empty / unparseable | 3720 / 3720 |
| 相同 plan 的反事实对 | 69.78% |
| 相同解析 action sequence | 100.00% |
| 平均生成 token | 256.00 |

`same action sequence = 100%` 不能解释为模型稳定地产生同一有效动作：所有输出都无法解析，
解析结果为空序列。模型也全部跑满 256 个生成 token。step0 在这组反事实任务上没有产生可用
的 zero-shot action 输出。

## 口径修正

全量 `weighted approx = 0.049624` 仍只有 trainer `eval_loss = 0.428906` 的 11.57%。
因此旧报告中“n8 没抽到难例/长尾导致差距”的解释不成立。两者存在稳定的归一化、mask 或
预处理口径差异；在核对 ms-swift 的真实 denominator 之前：

- trainer `eval_loss` 只用于训练曲线内部比较；
- 本脚本的 full/body/format 指标只在同一脚本、同一数据口径下比较；
- 不应把 `weighted approx` 与 trainer `eval_loss` 直接作绝对值对照。

## 结果文件

- `section-loss-eval-gpu-bf16-step0-full/`：全量三段 JSON、CSV 和 EVA 报告。
- `eva-audit-complete-cf-step0/`：step0 的完整 CF 审计。
- `counterfactual-comparison/`：step0 与 step11193 的逐对结果、置信区间和错误分层。
- `training/run_cot100k_step0_full_server.sh`：可复现的服务器全量评估入口。
