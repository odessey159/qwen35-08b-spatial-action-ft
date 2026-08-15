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
- full-CoT 使用 `state=0.3 / plan=0.3 / action=0.4` 的非二值 `loss_scale`；plan-only 使用
  `plan=0.4 / action=0.6`。这两种配置关闭 Liger kernel，以保留分段权重。
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

三组对照由 `data.response_format` 控制，并使用独立的 prepared/output 目录：

| 实验 | 配置 | tuner | assistant 目标 |
|---|---|---|---|
| Exp-A action | `config.action.server.json` | 全参数 | `<action>` |
| Exp-B full-CoT | `config.cot.server.json` | 全参数 | `<state>` + `<plan>` + `<action>` |
| Exp-C plan-only | `config.plan.server.json` | 全参数 | `<plan>` + `<action>` |
| full-CoT LoRA 备选 | `config.lora.cot.server.json` | LoRA | 与 full-CoT 相同 |

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

## 服务器使用顺序

先将本仓库同步到 `/root/qwen35-08b-spatial-action-ft`。安装脚本会复用服务器
Python 3.12 环境中已有的 CUDA 13.2 / PyTorch 2.13，不会重装 PyTorch：

```bash
cd /root/qwen35-08b-spatial-action-ft
bash training/setup_server.sh
```

准备并核验 Exp-A 数据：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.action.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.action.server.json validate
.venv-train/bin/python -m training.cli check-runtime
```

先审阅完整命令，再训练 Exp-A：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.action.server.json show-command
.venv-train/bin/python -m training.cli \
  --config training/config.action.server.json train
```

full-CoT 的主方案使用全参数微调：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.cot.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.server.json validate
.venv-train/bin/python -m training.cli \
  --config training/config.cot.server.json train
```

Exp-C plan-only 消融只需把上述 full-CoT 配置换成 `training/config.plan.server.json`。

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
