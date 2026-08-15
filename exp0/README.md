# Exp 0 诊断检测代码

该目录严格实现总纲 §6.1，不改变实验条件。代码不会自动训练模型。

## 输入数据

可以运行 ProcTHOR 生成器自动制作 240 条数据，也可以将
`data/samples.example.jsonl` 复制为 `data/samples.jsonl` 后人工填写
200–300 条真实样本。每条必须包含：

- 正确图像 `image`
- 不相关图像 `wrong_image`
- 指令 `instruction`
- 标准动作序列和自然语言计划 `gold`
- 正确场景图 `scene_graph`
- 正确空间事实 `spatial_facts`
- 正确子目标 `subgoals`
- `sim_verified=true`

图像路径相对于 `samples.jsonl` 所在目录。

### 自动生成数据

生成器使用规则 planner，并在 AI2-THOR 中实际执行每条计划。最终数据由
ProcTHOR 主分片与 iTHOR `HeatObject` 分片合并而成：

- ProcTHOR train split 中固定随机抽取 30 个场景
- 672×672 单张图像，任务相关物体必须全部可见
- 总计 240 条
- 40 组容器状态反事实对，共 80 条，占 33.3%
- 其余 160 条按 pickup、clean、heat、toggle、slice、open/close 分组

单进程冒烟测试可在安装 `requirements-generator.txt` 后执行：

```powershell
python -m exp0.generate_data
```

无可用 Vulkan 图形栈的 Linux 服务器可在 Xvfb 中改用 Linux64 构建：

```bash
AI2THOR_PLATFORM=Linux64 xvfb-run -a python -m exp0.generate_data
```

服务器实测的完整稳定生成流程如下。ProcTHOR 使用 AI2-THOR 5.0；由于该构建中的
ProcTHOR 食物资产不是 `cookable`，24 条 heat 数据使用独立 AI2-THOR 4.3 iTHOR
厨房生成。两个环境不会混用。

```bash
# ProcTHOR：前三个分片使用固定 30 场景配置，第四片扩大搜索范围以补足 slice。
for i in 0 1 2; do
  AI2THOR_PLATFORM=Linux64 xvfb-run -a .venv/bin/python -m exp0.generate_data \
    --config exp0/generator_config.procthor_no_heat.json \
    --shard-index "$i" --shard-count 4 --overwrite
done
AI2THOR_PLATFORM=Linux64 xvfb-run -a .venv/bin/python -m exp0.generate_data \
  --config exp0/generator_config.procthor_no_heat_retry.json \
  --shard-index 3 --shard-count 4 --overwrite

# iTHOR heat：独立 4.3 环境，串行运行避免多个 Unity 软件渲染进程互相阻塞。
python3 -m venv .venv-ithor
.venv-ithor/bin/pip install ai2thor==4.3.0 "numpy<2" Pillow
for i in 0 1; do
  AI2THOR_PLATFORM=Linux64 xvfb-run -a .venv-ithor/bin/python -m exp0.generate_data \
    --config exp0/generator_config.ithor43_heat_retry.json \
    --shard-index "$i" --shard-count 2 --overwrite
done
for i in 2 3; do
  AI2THOR_PLATFORM=Linux64 xvfb-run -a .venv-ithor/bin/python -m exp0.generate_data \
    --config exp0/generator_config.ithor43_heat_parallel.json \
    --shard-index "$i" --shard-count 4 --overwrite
done

# 先合并 24 条 heat，再合并最终 240 条。
.venv/bin/python -m exp0.merge_shards --shard-root exp0/heat_shard_data \
  --shards 4 --output-dir exp0/shard_data/shard_4 --seed 43 \
  --expected-count 24 --overwrite
.venv/bin/python -m exp0.merge_shards --shard-root exp0/shard_data \
  --shards 5 --output-dir exp0/data --seed 42 \
  --expected-count 240 --overwrite

.venv/bin/python -m exp0.cli --config exp0/config.server.json validate
.venv/bin/python -m exp0.validate_generated_data --dataset-dir exp0/data
```

重新生成会删除对应输出分片中的旧图像，因此必须显式添加 `--overwrite`。

## 五条件实现

- `D`：正确图像 + 正确指令 + 正确子目标
- `A_prime`：不相关图像 + 正确指令
- `A`：正确图像 + 正确指令
- `B_natural` / `B_json` / `B_triples`：三种场景图序列化 + 指令，无图像
- `C`：正确图像 + 指令 + 正确空间事实，不提供子目标

B 是一个概念条件，但按总纲分别运行三种序列化，因此每条样本实际产生 7 次推理。

## 使用顺序

安装依赖后，依次执行：

```powershell
python -m exp0.cli validate
python -m exp0.cli infer
python -m exp0.cli evaluate
```

推理支持断点续跑；只有明确需要清空旧结果时才使用：

```powershell
python -m exp0.cli infer --overwrite
```

## 输出

结果写入 `outputs/`：

- `predictions.jsonl`：模型原始输出和解析动作
- `scored_predictions.json`：逐样本得分
- `metrics.json`：条件汇总和差值
- `metrics_by_condition.csv`：表格结果
- `report.md`：按总纲规则生成的瓶颈判读

默认只检查动作名称是否越界。若需要检查 84 类物体词表，将完整词表填入 `config.json` 的 `allowed_objects`。

如果 A/B/C 全部触发 floor effect，报告只提示进入约 300 step 短 LoRA；短 LoRA 训练不属于本目录的零样本推理代码。
训练数据转换、服务器环境安装和 ms-swift 启动入口见 `../training/README.md`。
