# 训练前后 200 样本评测汇总

评测日期：2026-08-18

| 能力 | 当前主要指标 | 训练前 | 训练后 | 当前证据强度 |
|:---|:---|---:|---:|:---|
| **空间状态认知** | State body token acc（随机 n=200） | 66.44% | **98.55%** | 中（配对 n=200） |
|  | State body loss（随机 n=200） | 1.502708 | **0.036193** | 中（配对 n=200） |
|  | State Fact F1（随机 n=200，自由生成） | 0.00% | **88.62%** | 中（配对 n=200） |
|  | State Exact（随机 n=200，自由生成） | 0.00% | **51.00%** | 中（配对 n=200） |
| **规划能力** | Plan body token acc（随机 n=200） | 67.93% | **100.00%** | 中（配对 n=200） |
|  | Plan body loss（随机 n=200） | 2.023518 | **0.000027** | 中（配对 n=200） |
|  | Action body token acc（随机 n=200） | 82.95% | **100.00%** | 中（配对 n=200） |
|  | **CF Pair Exact（随机 100 pairs / 200 samples）** | 0.00% | **81.00%** | 中（随机 100 对） |
|  | CF Sample Exact（随机 100 pairs / 200 samples） | 0.00% | **87.00%** | 中（随机 100 对） |
|  | Same Action Sequence Rate ↓（随机 100 pairs） | 100.00% | **10.00%** | 中（随机 100 对） |
|  | Strict Action Sequence EM（随机 n=200） | 0.00% | **92.50%** | 中（配对 n=200） |
|  | Strict Step Position Match Recall（随机 n=200） | 0.00% | **97.14%** | 中（配对 n=200） |
|  | Strict Step Position Match Precision（随机 n=200） | 0.00% | **96.29%** | 中（配对 n=200） |
|  | **Relaxed Action Sequence EM（随机 n=200）** | 0.00% | **92.50%** | 中（配对 n=200） |
|  | Relaxed Step Position Match Recall（随机 n=200） | 2.42% | **97.14%** | 中（配对 n=200） |
|  | Relaxed Step Position Match Precision（随机 n=200） | 5.82% | **96.29%** | 中（配对 n=200） |

## 口径

- Teacher-forced 三段指标与 State/Overall Action 自由生成指标使用 validation 中完全相同的随机 200 条，抽样算法为 `random.Random(42).sample`；训练前后样本索引逐项一致。
- Teacher-forced 在服务器 RTX 4090 上以 CUDA/bfloat16 运行。step 0 和 step 11193 分别耗时 25.99 秒和 23.97 秒。
- State/Overall Action 自由生成中，训练前 200 条预测在服务器上重新生成；训练后从既有 9951 条服务器预测中抽取同一批样本。训练后 Action EM 的 scene-cluster bootstrap 95% CI 为 88.83%–95.98%。
- Strict action 指标只接受原评测器的规范 action contract；Relaxed 指标在不改动 Strict 结果的前提下，额外接受轻微标点/括号/list marker 变化，以及带显式动作名和显式 ontology object 的中英文自然语言表达。
- Relaxed 解析器不读取当前样本的 instruction、gold state、图片或 gold action 来补动作与参数；条件分支、否定动作、缺参或未知 object 均不形成 deterministic plan。训练后输出已满足 Strict contract，因此三项 Relaxed 后测与 Strict 后测相同。
- CF 指标从 validation 的全部 1860 个完整 CF pair 中按 seed 42 随机抽取 100 对。每一对均为同指令、同场景、open/closed 状态相反、图片哈希不同且 gold action 不同。
- CF 训练后 Sample Exact 的 95% CI 为 81.50%–92.50%，Pair Exact 的 95% CI 为 73.73%–88.42%。
- 200 样本结果用于替换原来的 n=8 pilot 和补齐缺失的训练前自由生成指标；已有 9951 样本 / 1860 pair 全量结果仍是更强的总体估计，不应被本表的 n=200 数值覆盖。

## 产物

- `section-loss-eval-gpu-bf16-n200-seed42/`：训练前后同一随机 200 条的 teacher-forced 分段指标。
- `eval-subset-random200-seed42/` 与 `pre-post-comparison-random200-seed42/`：同一随机 200 条的自由生成预测与 State/Overall Action 评分。
- `relaxed-action-metrics-random200-seed42/`：同一随机 200 条的 Relaxed 指标、解析审计和逐样本判分记录。
- `eval-subset-cf100-seed42/` 与 `counterfactual-comparison-cf100-seed42-n200/`：随机 100 个完整 CF pair 的预测子集、置信区间和错误分解。
