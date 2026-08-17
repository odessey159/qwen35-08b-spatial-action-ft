# Qwen3.5-0.8B CoT-10K 详细 EVA

## 结论先行

1. 训练已完成 `2250/2250` step；完整 1000 条 validation 的总 loss 从 `0.438532` 降至 `0.008684`，token accuracy 从 `73.72%` 升至 `98.77%`。
2. 原始 Qwen3.5-0.8B 权重已复制到本地，并作为 step 0 在相同的 8 条 validation 样本上完成 CPU float32 前向。0→250 时，`state/plan/action body loss` 分别下降 `90.82% / 99.97% / 97.05%`；`state format loss` 却从 `0.477653` **升至** `2.459163`。因此这组直接对照不支持“首段大幅下降主要是学会 CoT/XML 格式”，更符合目标正文 token 被迅速拟合，尤其是 plan/action。
3. 固定 8 条的 step-0 all-causal diagnostic 为 `0.049365`，而 trainer 日志为 `0.438532`。后续 100K 全量扫描证明这种差异主要来自聚合口径，不能归因于“难例长尾”。分段数字只作为同样本逐 token CE 趋势；teacher forcing 下的正文拟合也可能来自语言/模板记忆，不能等同于视觉规划能力。format-only arm 仍是正式因果归因不可裁的对照组。
4. 总 loss 绝对降幅的 `96.63%` 发生在 step 0→250；step 250 后当前最终 loss 几乎全部集中在 `<state>`，plan/action 已饱和。state format 在 250→750 下降后又反弹，最终仍高于 step 0；因此最后瓶颈更像 state 边界/状态表达，而不是 action 生成。
5. 数据 split 审计通过：并查集跨 split component=0，train/val `scene_id` 重叠=0，CF group 重叠=0；最大 component 仅 `12/10000=0.12%`，没有出现 56% 巨型连通分量。
6. 文本捷径很强：train-fitted text oracle 为 `72.60%`（coverage `99.30%`），validation Bayes text oracle 为 `74.00%`。相对可部署的 train oracle，视觉通道最多只剩 `27.40` 个百分点的净增益空间，报告中必须明说。
7. 本机 CPU 已对同一个完整 CF 对分别生成 step 0 与 step 2250：step 0 严格结构 `0/2`、CF pair exact `0/1`、same-plan `1/1`；step 2250 为 `2/2`、`1/1`、`0/1`。该对从“忽略开/关差异、输出同一计划”变为正确区分，但覆盖仅 `1/106`，不能外推完整 CF 分数。
8. format-only 数据/训练配置已实现，但服务器正在训练，尚未运行，因此当前仍不能完成“格式收益 vs 能力收益”的正式归因。

## 完整 validation 总指标

| step | eval loss | eval token acc | 相对 step 0 loss 降幅 |
|---:|---:|---:|---:|
| 0 | 0.438532 | 73.7151% | 0.00% |
| 250 | 0.023159 | 97.2723% | 94.72% |
| 500 | 0.017140 | 97.9288% | 96.09% |
| 750 | 0.013179 | 98.1730% | 96.99% |
| 1000 | 0.010358 | 98.5823% | 97.64% |
| 1250 | 0.009799 | 98.6192% | 97.77% |
| 1500 | 0.008905 | 98.7393% | 97.97% |
| 1750 | 0.008729 | 98.7741% | 98.01% |
| 2000 | 0.008690 | 98.7752% | 98.02% |
| 2250 | 0.008684 | 98.7711% | 98.02% |

- 0→250 的 loss 下降量为 `0.415373`，占总下降量 `0.429848` 的 `96.63%`。
- 0→250 的 token accuracy 增长为 `23.557` 个百分点，占总增长 `25.056` 个百分点的 `94.02%`。
- step 1750 后 loss 基本进入平台区：1750→2250 只下降 `0.00004454`（相对 1750 为 `0.51%`）。

## 三段 loss：固定 8/1000 CPU 样本

抽样 seed=42，validation 索引为 `25, 114, 142, 228, 250, 281, 654, 759`。这是 checkpoint 趋势 pilot，不是完整 validation 置信估计。

### Full loss（正文 + XML/换行/回合结束 token）

| step | state | plan | action |
|---:|---:|---:|---:|
| 0 | 1.245616 | 1.131371 | 0.547934 |
| 250 | 0.481274 | 0.000320 | 0.014673 |
| 750 | 0.303571 | 0.000223 | 0.013458 |
| 1500 | 0.335229 | 0.000070 | 0.000064 |
| 2250 | 0.337050 | 0.000067 | 0.000057 |

### Body loss（只统计标签内语义正文）

| step | state body | plan body | action body |
|---:|---:|---:|---:|
| 0 | 1.383162 | 1.928163 | 0.819648 |
| 250 | 0.127025 | 0.000599 | 0.024209 |
| 750 | 0.058110 | 0.000189 | 0.022200 |
| 1500 | 0.035503 | 0.000099 | 0.000092 |
| 2250 | 0.036584 | 0.000095 | 0.000081 |

### Format loss（XML 标签、换行和结束 token）

| step | state format | plan format | action format |
|---:|---:|---:|---:|
| 0 | 0.477653 | 0.290312 | 0.130173 |
| 250 | 2.459163 | 0.000025 | 0.000009 |
| 750 | 1.674065 | 0.000259 | 0.000017 |
| 1500 | 2.008694 | 0.000039 | 0.000021 |
| 2250 | 2.014654 | 0.000038 | 0.000020 |

### 分段 token accuracy / exact match

| step | state body token acc | plan body token acc | action body token acc | state body EM | plan body EM | action body EM |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 69.40% | 75.00% | 82.11% | 0/8 | 0/8 | 0/8 |
| 250 | 95.27% | 100.00% | 99.19% | 2/8 | 8/8 | 7/8 |
| 750 | 98.26% | 100.00% | 99.19% | 3/8 | 8/8 | 7/8 |
| 1500 | 98.51% | 100.00% | 100.00% | 2/8 | 8/8 | 8/8 |
| 2250 | 98.76% | 100.00% | 100.00% | 4/8 | 8/8 | 8/8 |

### 聚合口径诊断

这里的数值计算 `Σ(section_weight × token_CE) / 全部 causal token 数`，只应视为 all-causal diagnostic。后续 100K 的 9951 条全量 step-0 扫描证明该全局分母不能复刻 Swift trainer 的逐 batch eval loss；最终 checkpoint 的偶然接近不能证明分母实现等价。三段逐 token CE 的同样本趋势仍有效，但 diagnostic 与 trainer loss 不应直接对齐。

| step | 8-sample all-causal diagnostic | logged 1000-sample eval loss | 说明 |
|---:|---:|---:|:---|
| 0 | 0.049365 | 0.438532 | 聚合口径不同，不可直接比较 |
| 250 | 0.012646 | 0.023159 | 小样本低估早期难例 |
| 750 | 0.008039 | 0.013179 | 小样本低估早期难例 |
| 1500 | 0.008658 | 0.008905 | 已接近完整集 |
| 2250 | 0.008705 | 0.008684 | 相差 0.24% |

## 归因、数据泄漏与视觉依赖审计

| 检查项 | 当前结果 | 覆盖 | 判定 |
|:---|:---|:---|:---|
| 历史 base（Exp0 A）strict structure | 2.08% | 240 条；legacy `<plan>` contract | base 格式能力很低，但不是当前 10K val，不能直接作差 |
| 历史 base（Exp0 A）placeholder copy | 0.42% | 240 条 | 低；D oracle 条件为 61.25%，D 的高结构率受占位符模仿污染 |
| format-only arm score | 缺失 | arm/配置已实现，尚未训练 | **正式归因阻塞项，不可裁** |
| full-CoT strict structure（step 0→2250） | 0/2 → 2/2 | 同一 2/1000 CPU pilot | step 0 action 缺少参数，严格判无效；最终通过 |
| placeholder copy（step 0→2250） | 0/2 → 0/2 | 同一 2/1000 CPU pilot | 两端均未复制占位符 |
| CF pair exact（step 0→2250） | 0/1 → 1/1 | 同一 1/106 对 | 最终为正向证据；样本太少，不能报告正式 100% |
| CF same-plan（step 0→2250） | 1/1 → 0/1 | 同一 1/106 对 | step 0 两图输出同一 plan；最终能区分该对 |
| CF same-action-sequence（step 0→2250） | 1/1 → 0/1 | 同一 1/106 对 | 与 same-plan 结论一致 |
| merge 后 DSU 跨 split component | 0 | 10000 条 / 914 components | 通过 |
| 最大 DSU component | 12（0.12%） | 10000 条 | 未发现 56% 巨型分量 |
| CF group namespace | 0 个未加 shard 前缀 | 1256 groups | 通过，没有 shard id 撞车迹象 |
| train/val scene_id 重叠 | 0 | 822 train scenes / 92 val scenes | 通过；分数未因 scene 泄漏作废 |
| train/val CF group 重叠 | 0 | 1256 groups | 通过 |
| validation 完整 CF 对 | 106 | 106/106 gold 两侧动作不同 | 配对评测集有效 |
| filtered train-fitted text oracle | 72.60% | coverage 99.30% | 视觉最大净增益余量 27.40 个百分点 |
| validation Bayes text oracle | 74.00% | 1000 条过滤后 val | 分析上界；视觉歧义部分约 26.00% |
| exact-text deterministic rate | 41.50% | 1000 条过滤后 val | 与 oracle accuracy 口径不同，不可拿来算视觉 headroom |

## 当前不能下的结论

- 不能说 98.77% token accuracy 代表 98.77% 视觉规划能力；它是 teacher-forcing token 指标，且文本 oracle 已达 72.60%。
- 可以比较本次本地 step 0 与 step 2250 的同一 CF 对和同一 8 条 loss 样本；但不能用历史 Exp0 base 的结构率与当前 10K 直接作差，因为 prompt、contract 和 eval split 不同。
- 不能用 1/1 CF pair exact 报正式 100%；正式值必须覆盖 validation 的 106 个完整、gold-discriminative CF 对。
- 在 format-only arm 未训练前，不能裁掉“模型主要学会 CoT 表达格式”这一解释。

## 已实现与待运行

- `models/Qwen3.5-0.8B-original`：从服务器复制的原始 Qwen 权重；13 个运行时文件，主权重 `1,746,942,600` bytes，与服务器一致并已成功加载 473 个 tensors。
- `training/evaluate_section_losses.py`：按 checkpoint 输出 state/plan/action 的 full/body/format loss、PPL、token accuracy、EM，并输出显式标注的聚合诊断；支持 safetensors index/shard、CUDA，以及 `--base-model-dir` step 0。
- `training/audit_eva.py`：输出结构/占位符、CF pair-level、DSU split、scene/CF 重叠和 text oracle 审计。
- `training/generate_cpu_predictions.py`：本机 CPU 按 manifest 取完整 CF 对并生成预测。
- `training/config.cot.format-only.10k.server.json`：format-only train 标签三元组确定性无自配错排，validation 保持真实标签。
- 服务器空闲后必须补：format-only 完整训练；当前 full-CoT 与 format-only 在同一 1000 val 上生成；完整 106 CF 对 pair scorer；必要时对更多 checkpoint 跑完整三段 loss。
