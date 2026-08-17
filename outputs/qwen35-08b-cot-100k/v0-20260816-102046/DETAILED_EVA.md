# Qwen3.5-0.8B CoT-100K 详细 EVA

> **2026-08-17 step0 更新：** step0 已改用完整 `9951/9951` validation 重算，正式数值见
> `STEP0_FULL_EVA.md` 和 `section-loss-eval-gpu-bf16-step0-full/`。本文后面的固定 n8
> 跨-checkpoint 表仅保留为同样本趋势 pilot，其 step0 数值不再作为正式基线；全量重算也证明
> `weighted approx` 与 trainer `eval_loss` 的差距不是“n8 未抽到难例”，而是已定位的聚合口径差异：Swift 按 batch 结合 `labels` 与 `loss_scale` 归约，不能由一个全局 token 分母复刻。

## 结论先行

1. 100K 训练正常完成 `11193/11193` step、1 epoch，训练集 `89538`、validation `9951`；总耗时 `9098.61s`（2h31m39s），峰值显存 `23.03 GiB`。完整 validation 最佳 checkpoint 是 step `11000`，loss `0.005916716`；最终 step `11193` 为 `0.005917093`，差异仅 `0.0064%`。
2. 完整 validation loss 从 step 0 的 `0.428906` 降至 `0.005917`（-98.62%），token accuracy 从 `73.97%` 升至 `99.19%`。总 loss 绝对降幅的 `97.08%` 已发生在 0→1000；9000→最终仅再下降 `0.14%`，后段已经平台化。
3. 固定 8 条同样本的 step 0→3000 直接对照显示，`state/plan/action body loss` 分别下降 `97.10% / 99.998% / 99.994%`，但 `state format loss` 从 `0.405031` 升至 `2.818685`（+595.92%）。因此大幅 loss 下降不能解释成“主要学会 XML/CoT 格式”；它主要来自目标正文 token，尤其 plan/action 的快速拟合。
4. 这不等价于视觉能力：teacher forcing 下的正文拟合仍可能来自语言模板和文本捷径。100K 的 train-fitted text oracle 已达 `62.54%`，coverage `99.96%`；视觉通道相对该 oracle 最多剩 `37.46` 个百分点净增益空间。format-only arm 尚未训练，正式因果归因仍未闭环。
5. 本次已在服务器对 validation 的全部 `1860/1860` 个完整、gold-discriminative CF 对生成并评分。最终模型 pair exact 为 `79.25%`（1474/1860），样本级 exact 为 `87.10%`；292 对仅一侧正确，94 对两侧都错。pair-level 明显低于样本级，说明只看样本分会高估能力。
6. step 0 在同一 1860 对上 strict structure `0/3720`、pair exact `0/1860`、same-action `1860/1860`；最终为 strict structure `3720/3720`、pair exact `1474/1860`、same-action `276/1860`。模型确实学会了结构并显著开始看图，但仍有 `14.84%` 的 CF 对对两张不同图输出同一动作序列，视觉依赖尚不充分。
7. split 审计通过：merge 后并查集跨 split component=0；最大 component 仅 `47/99489=0.047%`；train/val `scene_id` 重叠=0、CF group 重叠=0，且所有 CF group 都带 shard namespace。没有发现 56% 巨型分量或 scene 泄漏。
8. 与 10K 各自 validation 的最终日志作非配对参考，100K loss 低 `31.86%`，token accuracy 高 `0.417` 个百分点；但两者 validation split 不同，不能把该差值当成严格的数据规模因果收益。

## 完整 validation 总指标

| step | eval loss | eval token acc | 相对 step 0 loss 降幅 |
|---:|---:|---:|---:|
| 0 | 0.428906 | 73.9738% | 0.00% |
| 1000 | 0.018266 | 97.7421% | 95.74% |
| 2000 | 0.012446 | 98.3000% | 97.10% |
| 3000 | 0.009727 | 98.7065% | 97.73% |
| 4000 | 0.008186 | 98.8472% | 98.09% |
| 5000 | 0.007204 | 99.0169% | 98.32% |
| 6000 | 0.006520 | 99.0976% | 98.48% |
| 7000 | 0.006155 | 99.1573% | 98.56% |
| 8000 | 0.006004 | 99.1784% | 98.60% |
| 9000 | 0.005926 | 99.1898% | 98.62% |
| 10000 | 0.005919 | 99.1884% | 98.62% |
| 11000 | **0.005917** | **99.1907%** | 98.62% |
| 11193 | 0.005917 | 99.1883% | 98.62% |

- 0→1000 的 loss 下降量占总下降量 `97.08%`，token accuracy 增长占最终总增长 `94.26%`。
- step 11000 是 trainer 记录的最佳 checkpoint；最终模型与最佳模型几乎相同，没有实质性末段过拟合信号。
- 由于 `save_total_limit=10`，checkpoint-1000/2000 已被清理；分段 loss 从现存最早的 checkpoint-3000 开始。

## 三段 loss：固定 8/9951 GPU 样本

抽样 seed=42，validation 索引为 `409, 1679, 1824, 2286, 3657, 4012, 4506, 8935`。设备为 RTX 4090、bfloat16。这是相同样本的 checkpoint 趋势 pilot，不是完整 validation 分段置信估计。

### Step 0 全量 baseline（9951/9951）

| section | tokens | full loss | body loss | format loss | full token acc | body token acc |
|:---|---:|---:|---:|---:|---:|---:|
| state | 553738 | 1.287582 | 1.452780 | 0.431366 | 69.92% | 67.11% |
| plan | 187134 | 1.186445 | 2.000356 | 0.299685 | 78.10% | 68.19% |
| action | 247503 | 0.581972 | 0.876851 | 0.143424 | 87.36% | 82.87% |

这份全量 baseline 与固定 8 条的 section CE 接近，支持用固定 8 条观察 checkpoint 内部趋势；但它不解决 trainer loss 的 batch 聚合口径差异。

### Full loss（正文 + XML/换行/EOS token）

| step | state | plan | action |
|---:|---:|---:|---:|
| 0 | 1.258938 | 1.174779 | 0.596985 |
| 3000 | 0.541791 | 0.000025 | 0.000037 |
| 6000 | 0.481374 | 0.000019 | 0.000018 |
| 9000 | 0.491328 | 0.000025 | 0.000017 |
| 11000 | 0.497451 | 0.000023 | 0.000018 |
| 11193 | 0.491432 | 0.000023 | 0.000018 |

### Body loss（只统计标签内正文）

| step | state body | plan body | action body |
|---:|---:|---:|---:|
| 0 | 1.446381 | 2.202163 | 0.971334 |
| 3000 | 0.041984 | 0.000042 | 0.000056 |
| 6000 | 0.019453 | 0.000029 | 0.000023 |
| 9000 | 0.014630 | 0.000042 | 0.000021 |
| 11000 | 0.014843 | 0.000038 | 0.000021 |
| 11193 | 0.015101 | 0.000038 | 0.000021 |

### Format loss（XML 标签、换行和 EOS token）

| step | state format | plan format | action format |
|---:|---:|---:|---:|
| 0 | 0.405031 | 0.318625 | 0.143088 |
| 3000 | 2.818685 | 0.000011 | 0.000014 |
| 6000 | 2.585678 | 0.000010 | 0.000013 |
| 9000 | 2.662956 | 0.000011 | 0.000013 |
| 11000 | 2.696002 | 0.000011 | 0.000015 |
| 11193 | 2.661383 | 0.000012 | 0.000014 |

### Body token accuracy / exact match

| step | state body acc | plan body acc | action body acc | state EM | plan EM | action EM |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.99% | 68.33% | 82.47% | 0/8 | 0/8 | 0/8 |
| 3000 | 98.17% | 100.00% | 100.00% | 0/8 | 8/8 | 8/8 |
| 6000 | 100.00% | 100.00% | 100.00% | 0/8 | 8/8 | 8/8 |
| 9000 | 99.70% | 100.00% | 100.00% | 0/8 | 8/8 | 8/8 |
| 11000 | 99.70% | 100.00% | 100.00% | 0/8 | 8/8 | 8/8 |
| 11193 | 99.70% | 100.00% | 100.00% | 0/8 | 8/8 | 8/8 |

plan/action 在 step 3000 已对这 8 条全部精确匹配；后续 loss 几乎全部由 state 尤其 state format token 主导。state body 持续改善到约 step 9000，但 state format 始终高于 step 0，并在后段平台波动。

### 聚合口径诊断

| step | 8-sample all-causal diagnostic | logged 9951-sample eval loss | 说明 |
|---:|---:|---:|:---|
| 0 | 0.044410 | 0.428906 | 口径不同，不可据此归因于长尾 |
| 3000 | 0.012038 | 0.009727 | 小样本偏高 |
| 6000 | 0.010696 | 0.006520 | 小样本偏高 |
| 9000 | 0.010917 | 0.005926 | 小样本偏高 |
| 11000 | 0.011053 | 0.005917 | 小样本偏高 |
| 11193 | 0.010919 | 0.005917 | 小样本偏高 |

额外完成的 **9951/9951 step-0 全量扫描**得到：all-causal diagnostic `0.049624`、全局 supervised-token diagnostic `0.328851`、raw micro loss `1.091738`，而 trainer 日志仍为 `0.428906`。这证明差异主要是聚合口径，而不是固定 8 条的“难例长尾”：trainer 在 Swift collator/template 生成 labels 与 loss_scale 后逐 batch 归约，不能由一个全局 token 分母复刻。因此：完整总 loss 只引用 trainer 日志；full/body/format 表只比较同一采样集上的逐 token CE，不把 diagnostic 与 trainer loss 作数值对齐。

## 结构、占位符与 CF 配对结果

结构率覆盖全部 CF 样本 `3720/9951=37.38%`；CF pair 指标覆盖全部完整 CF 对 `1860/1860=100%`，且 1860 对的 gold 动作均有差异。

| 指标 | step 0 | step 11193 | 判定 |
|:---|---:|---:|:---|
| strict structure valid | 0/3720（0.00%） | 3720/3720（100.00%） | 结构适配非常显著，但这是 CF 子集结构率 |
| placeholder copy | 0/3720（0.00%） | 0/3720（0.00%） | 结构提升不是复制占位符造成 |
| CF pair exact | 0/1860（0.00%） | 1474/1860（79.25%） | 正式 pair-level 分数 |
| CF sample exact | 0/3720（0.00%） | 3240/3720（87.10%） | 高于 pair-level，单样本会高估能力 |
| 仅一侧正确的 CF 对 | 0 | 292（15.70%） | pair scorer 会把这些判为失败 |
| 两侧都错的 CF 对 | 1860（100.00%） | 94（5.05%） | 仍有明确能力缺口 |
| same-plan rate | 1298/1860（69.78%） | 815/1860（43.82%） | 高层 plan 可能合法相同，不能单独当视觉盲证据 |
| same-action-sequence rate | 1860/1860（100.00%） | 276/1860（14.84%） | 276 对仍对不同图输出同一动作序列 |

step 0 的输出经常退化为重复、截断的自然语言计划，无法形成合法 action；最终模型的结构完全合规。尽管最终 CF pair exact 已达 79.25%，`same-action=14.84%` 表明模型仍会在一部分 open/closed 反事实图像上忽略关键视觉差异。

## 数据泄漏与文本捷径审计

| 检查项 | 当前结果 | 覆盖 | 判定 |
|:---|---:|:---|:---|
| 原始 / train / validation 样本 | 99489 / 89538 / 9951 | 全量 | 数量一致，无未分配或缺失 ID |
| merge 后 DSU 跨 split component | 0 | 7595 components | 通过 |
| 最大 DSU component | 47（0.047%） | 99489 条 | 未发现巨型连通分量 |
| CF group namespace | 0 个未加 shard 前缀 | 18265 groups | 通过，无 shard group id 撞车 |
| train/val scene_id 重叠 | 0 | 6843 train scenes / 752 val scenes | 通过；分数未因 scene 泄漏作废 |
| train/val CF group 重叠 | 0 | 18265 groups | 通过 |
| validation 完整 CF 对 | 1860 | 1860/1860 gold-discriminative | pair 评测集有效 |
| filtered train-fitted text oracle | 62.54% | coverage 99.96% | 视觉最大净增益余量 37.46pp |
| validation Bayes text oracle | 62.92% | 9951 条过滤后 val | 分析上界；视觉歧义约 37.08% |
| exact-text deterministic rate | 25.72% | 9951 条过滤后 val | 与 oracle accuracy 口径不同 |

与 10K 的 train-fitted text oracle `72.60%` 相比，100K 为 `62.54%`；100K validation 的精确文本映射更歧义，因而视觉区分的理论空间更大。这是两个 split 的分布诊断，不应解释为模型分数。

## format-only 对照组

- `training/config.cot.format-only.100k.server.json` 已实现：train 标签使用确定性 `permuted_triplet` 错排，validation 保持真实标签。
- 该 arm 尚未训练，因此目前可以说“正文 loss 的下降并非只由格式 token 贡献”，但仍不能给出“视觉能力收益占多少、格式适配收益占多少”的正式因果比例。
- 要闭环，需要用相同 100K train budget 训练 format-only arm，并在同一 9951 validation / 1860 CF 对上运行完全相同的生成与 scorer。

## 10K / 100K 非配对参考

| run | validation size | final step | final eval loss | final token acc | CF pair coverage |
|:---|---:|---:|---:|---:|:---|
| CoT-10K | 1000 | 2250 | 0.008684 | 98.7711% | 1/106 pilot，不能作正式比较 |
| CoT-100K | 9951 | 11193 | 0.005917 | 99.1883% | 1860/1860，正式 pair score 79.25% |

100K 的 own-validation loss 比 10K 低 31.86%，accuracy 高 0.417pp；但 validation 样本并不相同，且 10K 没有完整 CF 生成，因此不能据此报告严格的数据规模提升比例。

## 产物与复现

- `section-loss-eval-gpu-bf16-n8/EVA.md`：step 0、3000、6000、9000、11000、11193 的同样本 full/body/format loss。
- `section-loss-eval-gpu-bf16-step0-full/EVA.md`：9951/9951 的 step-0 全量分段 baseline 与聚合口径诊断。
- `eva-audit-complete-cf-step0/audit.json`：step 0 的 1860 对完整 CF 审计。
- `eva-audit-complete-cf-step11193/audit.json`：最终模型的 1860 对完整 CF 审计。
- 服务器预测：`gpu-generation/cf-all-step0.jsonl` 与 `gpu-generation/cf-all-step11193.jsonl`，各 3720 条；未复制回本地以避免无必要的大文件同步。
- `training/evaluate_section_losses.py` 已支持 CPU/CUDA 和 base step 0；`training/generate_cpu_predictions.py` 已支持 CUDA、SDPA 与 batch generation；`training/audit_eva.py` 已支持 base/current 同一 CoT contract 和 pair/sample 失败分解。
