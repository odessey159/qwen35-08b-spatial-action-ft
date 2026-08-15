# Qwen3.5-0.8B 训练程序

该目录把已有空间规划样本转换为 ms-swift 标准多模态格式，并在服务器上启动
Qwen3.5-0.8B 的 bf16 SFT。默认方案为全参数微调，同时保留 LoRA 作为替代配置。
当前服务器适配目标为：

- RTX 4090 24GB，CUDA 可用且支持 bf16
- Python 3.12：`/usr/local/miniconda3/envs/py312/bin/python`
- 模型：`/model/ModelScope/Qwen/Qwen3.5-0.8B`
- ms-swift 4.4.2

## 已实现的约束

- 同时接受当前 Exp 0 schema（`instruction + gold + meta`）和总纲 schema
  （`prompt + plan_actions + plan_nl + _meta`）。
- 输出为 ms-swift 的 `messages + images` JSONL；图像使用运行转换程序的机器上的绝对路径。
- 按 `scene_id` 和 `counterfactual_group` 的连通分组切分，避免同场景或同一反事实对跨越
  train/validation。
- prompt 与零样本 A 条件一致，response 固定为 `<plan>...</plan>` 加自然语言计划。
- 全参和 LoRA 都启用 Liger CE、Flash Attention 2、非 thinking 前缀与 bf16。
- LoRA 替代方案使用 `all-linear`，覆盖 Qwen3.5 的混合注意力线性层。
- 不使用 QLoRA，也不手写 multimodal collator/mRoPE position ids。

## 两套训练配置

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

## 服务器使用顺序

先将本仓库同步到 `/root/qwen35-08b-spatial-action-ft`。安装脚本会复用服务器
Python 3.12 环境中已有的 CUDA 13.2 / PyTorch 2.13，不会重装 PyTorch：

```bash
cd /root/qwen35-08b-spatial-action-ft
bash training/setup_server.sh
```

准备并核验数据：

```bash
.venv-train/bin/python -m training.cli prepare
.venv-train/bin/python -m training.cli validate
.venv-train/bin/python -m training.cli check-runtime
```

默认执行全参数微调。先审阅完整命令，再训练：

```bash
.venv-train/bin/python -m training.cli show-command
.venv-train/bin/python -m training.cli train
```

切换到 LoRA 时无需重新准备数据：

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.lora.server.json show-command
.venv-train/bin/python -m training.cli \
  --config training/config.lora.server.json train
```

若训练被中断，把 `training.resume_from_checkpoint` 改为某个 `checkpoint-*` 目录；启动器会
先验证目录存在，再把它传给 ms-swift，恢复 optimizer、随机数状态和数据进度。

`prepare` 默认不覆盖既有结果；源数据变化后需显式使用 `prepare --overwrite`。
每次准备都会写 `prepared/manifest.json`，记录源数据 SHA-256、随机种子和两个 split 的
sample id，便于复现实验。

## 当前超参定位

两套配置的 `max_steps=300` 都是短程重测/Pilot 起点，而不是 10 万条主实验的最终训练
时长。主实验开始前应把训练终止方式改为经 Pilot 确定的 epoch/step，其余参数可继续复用。
