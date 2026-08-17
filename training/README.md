# Qwen3.5-0.8B 训练程序

该目录把已有空间规划样本转换为 ms-swift 标准多模态格式，并在服务器上启动
Qwen3.5-0.8B 的 bf16 SFT。默认方案为全参数微调，同时保留 LoRA 作为替代配置。
当前服务器适配目标为：

- RTX 4090 24GB，CUDA 可用且支持 bf16
- Python 3.12：`/usr/local/miniconda3/envs/py312/bin/python`
- 模型：`/model/ModelScope/Qwen/Qwen3.5-0.8B`
- ms-swift 4.4.2

## 已实现的约束

- 原有 Exp 0 schema 与 `training/data.py` 保持兼容；embodied CoT 的
  `conversations + oracle` schema 由 `training/cot_data.py` 单独转换。
- 输出为 ms-swift 的 `messages + images` JSONL；图像使用运行转换程序的机器上的绝对路径。
- 按 `scene_id` 和 `counterfactual_group` 的连通分组切分，避免同场景或同一反事实对跨越
  train/validation。
- Exp-A、Exp-B 和 Exp-C 始终在 `<action>` 中保留完全相同的 primitive action sequence。
- Exp-C 使用 `state=0.3 / plan=0.3 / action=0.4` 的非二值 `loss_scale`；Exp-B 删除
  state 后保留 C 的 `plan:action=3:4` 相对权重，即 `plan=3/7 / action=4/7`；Exp-A
  使用 `action=1.0`。三组都关闭 Liger kernel，以保留同一套分段 loss 机制。
- prepared JSONL 为每个连续 assistant 区块写入一个标量 `loss_scale`；ms-swift 的 Swift
  backend 会在 tokenize 前合并这些连续区块，从而保持单次生成的输出形态与分段权重对应。
- LoRA 替代方案使用 `all-linear`，覆盖 Qwen3.5 的混合注意力线性层。
- 不使用 QLoRA，也不手写 multimodal collator/mRoPE position ids。

## 训练与实验配置

默认的 `config.server.json` 是全参数微调：

```json
"tuner_type": "full",
"freeze_policy": {
  "llm": false,
  "vit": false,
  "aligner": false
}
```

这会更新 LLM、ViT 和 aligner 的原始参数，不注入 LoRA。单卡起点为学习率 `1e-5`、
micro batch 2、梯度累积 2，有效 batch 为 4；输出写入 `outputs/qwen35-08b-full`。

`config.lora.server.json` 是可直接使用的 LoRA 替代方案：rank 8、alpha 32、
`all-linear`、学习率 `1e-4`、micro batch 4，输出写入 `outputs/qwen35-08b-lora`。
该配置当前也不冻结三个组件，因此 LoRA 会挂到 LLM、ViT 和 aligner；若后续需要缩小
覆盖范围，可修改该文件中的冻结项：

| LoRA 覆盖策略 | llm | vit | aligner |
|---|---:|---:|---:|
| 只在语言侧挂 LoRA | false | true | true |
| 三部分都挂 LoRA | false | false | false |
| 只训练视觉侧与 projector | true | false | false |

无论哪套配置，三个组件全为 `true` 都会被启动器拒绝。

三组对照由 `data.response_format` 控制，并使用独立的 prepared/output 目录。A/B 配置
直接继承对应规模的 C 配置，因此除 target、target 对应的 loss 权重以及 prepared/output
目录外，数据源、split、seed、模型、全参微调策略和训练超参数都与 C 相同：

| 实验 | 配置 | tuner | assistant 目标 |
|---|---|---|---|
| Exp-A Action-only | `config.action.10k.server.json` | 全参数 | `<action>` |
| Exp-B Plan + Action | `config.plan.10k.server.json` | 全参数 | `<plan>` + `<action>` |
| Exp-C State + Plan + Action | `config.cot.10k.server.json` | 全参数 | `<state>` + `<plan>` + `<action>` |

100K 版本分别是 `config.action.100k.server.json`、`config.plan.100k.server.json` 和
`config.cot.100k.server.json`，继承关系与 10K 完全相同。旧的无规模后缀配置仍保留为
240 条开发数据的快速测试配置，不用于正式三组比较。

`<plan>` 是高层任务分解，`<action>` 才保存原始 primitive action。Exp 0 evaluator 优先只从
`<action>...</action>` 抽取动作，同时保留对旧 `<plan>` action 输出的兼容回退。测试时使用
`exp0/config.cot.server.json`，强制模型只输出 `<action>`。

## 生成 CoT 数据

构建脚本直接使用 scene graph、`spatial_facts` 与 simulator action trajectory，通过规则生成
task-relevant state 和高层 subgoals，不调用 LLM：

```bash
.venv-train/bin/python -m exp0.generate_cot_data --overwrite
.venv-train/bin/python -m exp0.generate_cot_data --validate-only
```

默认读取 `exp0/data/samples.jsonl`，写入 `exp0/data_cot/samples_cot.jsonl` 和
`generation_report.json`。验证会检查 image/instruction/state/plan/action 完整性、动作语法、
plan-action 一致性、图像存在性、`sim_verified=true`，并报告三段平均 token 长度。

## 直接读取 raw simulator 数据

`training/cot_data.py` 现在支持两种显式输入路径：旧 CoT 数据继续使用默认的 `cot` 格式；
generator 原始输出可在配置中设置：

```json
"source_format": "raw_simulator"
```

raw adapter 严格保留 `gold.plan_actions` 的 simulator-verified primitive action 顺序，使用
`extract_task_relevant_state(raw_row)` 提取任务相关状态，并统一通过
`abstract_subgoals(actions)` 生成高层 plan。raw row 自带的 `subgoals` 不作为训练 truth。
adapter 只生成内部 `CotSample`，之后仍复用同一套 `_swift_row()` ms-swift serialization；
内部 plan 不带编号，最终 rendered `<plan>` 才添加编号。

当前开发与自动测试使用 repo 现有的小样本和 synthetic raw fixture 验证
`raw simulator adapter → prepare → validate`，不依赖也不会创建未来的 5000 条数据。

`expected_source_samples` 是可选的部署保护。配置后，`prepare` 会要求 source 中的非空 JSONL
样本数完全一致。每次 `validate`（以及训练前的自动 validation）还会比较当前 source SHA-256
与 prepared manifest；源文件变化后必须重新执行 `prepare --overwrite`。

### 未来 5000 Pilot

`config.cot.pilot5000.server.json` 只为未来数据部署预留。当前 repo 不包含
`exp0/new5000_data/samples.jsonl`，因此现在不能用该配置成功执行 `prepare`。等外部生成任务完成，
并将完整 5000 条数据和图像放到约定目录后，依次执行：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.cot.pilot5000.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.pilot5000.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.cot.pilot5000.server.json check-runtime
.venv-train/bin/python -m training.cli \
  --config training/config.cot.pilot5000.server.json show-command
.venv-train/bin/python -m training.cli \
  --config training/config.cot.pilot5000.server.json train
```

### 69,879 条原始数据中的 10,000 条训练

`config.cot.10k.server.json` 复用已成功的 500 条 full-CoT 全参数配置。它按固定 seed 从 8 个
generation shard 中精确选择 10,000 条，并且以完整 scene 为选择单位；随后仍按 scene 和
counterfactual group 做 90/10 train/val 切分。训练跑 1 个 epoch，step 0 先执行一次 val，
之后每 250 optimizer steps 做一次 val 并保存 checkpoint，最多保留 10 个 checkpoint。

正式 A/B 对照分别使用 `config.action.10k.server.json` 与 `config.plan.10k.server.json`。
二者直接 `extends` C 的 `config.cot.10k.server.json`；因此 C 后续若修改 batch、epoch、
learning rate、eval/save cadence 等公共训练设置，A/B 会自动继承，不需要手工同步三份配置。

`config.cot.format-only.10k.server.json` 是不可省略的格式归因对照。它复用完全相同的
10K 输入、并查集 split、超参和真实 validation，但在 **train split 内**把完整
`state/plan/action` 标签三元组做确定性无自配错排。三段一起移动以维持合法结构和
plan/action 内部一致性；每个标签三元组恰好使用一次，且不会回配给原样本。该 arm 的
validation 分数代表仅靠格式适配和数据先验可获得的收益，不能用主实验分数代替。

```bash
.venv-train/bin/python -m training.select_raw_subset \
  --shard-root exp0/new100k_shard_data \
  --shards 8 \
  --output-dir exp0/new100k_10k_data \
  --count 10000 \
  --seed 42 \
  --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.10k.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.10k.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.cot.10k.server.json train
```

服务器空闲后，format-only 必须单独准备、验证并训练：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.cot.format-only.10k.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.format-only.10k.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.cot.format-only.10k.server.json train
```

需要后台运行并保留统一日志时，使用：

```bash
nohup bash training/run_cot10k_server.sh \
  > run_logs/cot10k_train.log 2>&1 < /dev/null &
echo $! > run_logs/cot10k_train.pid
```

## 服务器使用顺序

先将本仓库同步到 `/root/qwen35-08b-spatial-action-ft`。安装脚本会复用服务器
Python 3.12 环境中已有的 CUDA 13.2 / PyTorch 2.13，不会重装 PyTorch：

```bash
cd /root/qwen35-08b-spatial-action-ft
bash training/setup_server.sh
```

准备并核验 Exp-A（Action-only）数据：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.action.10k.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.action.10k.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.action.10k.server.json show-command
.venv-train/bin/python -m training.cli \
  --config training/config.action.10k.server.json train
```

准备、核验并训练 Exp-B（Plan + Action）：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.plan.10k.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.plan.10k.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.plan.10k.server.json show-command
.venv-train/bin/python -m training.cli \
  --config training/config.plan.10k.server.json train
```

Exp-C（State + Plan + Action）对应命令为：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.cot.10k.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.10k.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.cot.10k.server.json train
```

切换到 LoRA 时无需重新准备数据：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.lora.server.json show-command
.venv-train/bin/python -m training.cli \
  --config training/config.lora.server.json train
```

full-CoT 的 LoRA 备选配置为 `training/config.lora.cot.server.json`；它使用独立的
`prepared/lora-cot` 与输出目录，不会覆盖全参实验，因此首次运行也需要执行一次
`prepare --overwrite`。

若训练被中断，把 `training.resume_from_checkpoint` 改为某个 `checkpoint-*` 目录；启动器会
先验证目录存在，再把它传给 ms-swift，恢复 optimizer、随机数状态和数据进度。

`prepare` 默认不覆盖既有结果；源数据变化后需显式使用 `prepare --overwrite`。
每次准备都会写 `prepared/manifest.json`，记录源数据 SHA-256、随机种子和两个 split 的
sample id，便于复现实验。

## 当前超参定位

两套配置的 `max_steps=300` 都是短程重测/Pilot 起点，而不是 10 万条主实验的最终训练
时长。主实验开始前应把训练终止方式改为经 Pilot 确定的 epoch/step，其余参数可继续复用。
