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

生成器会把样本流式追加到 `samples.jsonl`。中断后再次运行同一命令会从已有文件续跑；只有需要推倒重来时才加 `--overwrite`。

一卡多进程时按 house 分片，每个进程一个 Controller：

```bash
AI2THOR_PLATFORM=CloudRendering python -m exp0.generate_data \
  --config exp0/generator_config.json --workers 4
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

生成与合并阶段都会拒绝同一图像内容和 instruction 对应多个 gold plan 的数据。
`generation_report.json` 还会记录 `instruction_collision`、`action_distribution` 和
`goto_schema`。其中 `goto_schema` 只统计 atomic 任务：存在可见父容器时必须导航到父容器；
没有可见父容器时允许回退到目标对象，并计入 `object_location`。

重新生成并清空对应输出分片时，必须显式添加 `--overwrite`。不加该参数时，已有 `samples.jsonl` 会作为断点续跑。

## 构建训练用 CoT 数据

下面的命令从 scene graph、`spatial_facts` 和 simulator action trajectory 生成
task-relevant state、高层 plan 与原始 primitive action；整个过程不调用 LLM：

```bash
python -m exp0.generate_cot_data --overwrite
```

默认输出为 `exp0/data_cot/samples_cot.jsonl`，assistant 格式为
`<state> + <plan> + <action>`；结构化 `oracle` 字段用于逐条验证三段内容。
同目录的 `generation_report.json` 记录样本数与三段平均 token 长度。

## 五条件实现

- `D`：正确图像 + 正确指令 + 正确子目标
- `A_prime`：不相关图像 + 正确指令
- `A`：正确图像 + 正确指令
- `B_natural` / `B_json` / `B_triples`：三种场景图序列化 + 指令，无图像
- `C`：正确图像 + 指令 + 正确空间事实，不提供子目标

B 是一个概念条件，但按总纲分别运行三种序列化，因此每条样本实际产生 7 次推理。

## 输出格式与打分

**所有格式规则都写在 system turn 里，user turn 只放题目本身。** 第一轮把规则连同
一个字面占位符 `ActionName(Object)` 放在 user turn 里，0.8B 把它当成模板照抄：D 条件
61.3% 的输出里含字面量 `ActionName`，全部输出里 90.9% 把规则中的整句复读了回来。规则
挪到 system turn 之后，任何原样复述都不会再被当成答案。

模型被要求输出两段：

```
<plan>
GotoLocation(Fridge)
OpenObject(Fridge)
</plan>
<summary>
走到 Fridge 前，把 Fridge 打开。
</summary>
```

**primitive action sequence 是主指标**（`primary_metric = action_seq_em`）。embodied CoT
输出时解析器优先读取 `<action>`，旧实验则兼容回退到 `<plan>`。词表封闭，精确匹配无歧义，
不需要 judge。打分走宽松解析器：接受
`GotoLocation Fridge`（无括号）、大小写漂移、行首编号和列表符号，并丢弃词表外的动作名
——第一轮 99.3% 的动作行是无括号形式，被严格正则整体判死。严格解析结果仍然保留，但只
作为 `strict_*` 格式合规率上报，不代表能力。

**`<summary>` 是旧双输出协议的次指标。** 使用 `exp0/config.cot.server.json` 评估新模型时，
模型只输出 `<action>`，诊断结论只使用 action 主指标。旧协议中 `nl_plan_match` = 全部 gold
步骤在自然语言里按顺序出现，且
物体名齐全，判定用同义词表（`schema.ACTION_NL_KEYWORDS`），所以"把门拉开"和"打开"都
算对，不需要 SBERT 或 LLM judge。`scored_predictions.json` 里保留了 `pred_nl` 和
`gold_nl`，想事后接 judge 随时可以。

`<summary>` 这个闭合标签保留着，不是为了打分，是为了**把两段输出切干净**：第一轮没有
标签时，`</plan>` 之后的内容有 90.9% 是复读的提示词，无论拿它做评测还是做训练目标都是
噪声。有了标签，训练时想给自然语言那一段降权或直接 mask 掉 loss 也有明确边界。

`system_prompt` 里带一个格式示例，用的是 `BathtubBasin` / `WateringCan`——这两个物体
不出现在 240 条诊断数据的任何场景里，示例的 3 步长度也不等于任何 gold 计划长度（gold
是 1/2/4/6 步）。因此"照抄示例"会被 `example_echo_rate` 抓到，而不会悄悄抬高分数。

若重跑后 D 的结构合法率仍然上不去，把 `config.json` 的 `model.format_demo_as_turns`
改成 `true`：格式示例会从 system turn 里的一段文字变成一轮真实的 user/assistant 对话。
小模型对"示范过的 assistant 轮"服从度高得多。该示例不含任何场景，只教排版，且七个条件
完全一致，不会改变条件之间的可比性。

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

- `predictions.jsonl`：模型原始输出、宽松解析出的动作、抽出的 `pred_nl`
- `scored_predictions.json`：逐样本得分，含 `pred_nl` / `gold_nl`，可直接喂给 LLM judge
- `metrics.json`：条件汇总、切片汇总和差值
- `metrics_by_condition.csv`：表格结果
- `report.md`：三张表（动作序列 / 自然语言 / 格式合规）加瓶颈判读

`report.md` 会按 `all` / `counterfactual` / `non_counterfactual` 以及每个 task_group
分别报主指标和 `A − A′`。这一点是必需的：本数据集有 81.7% 的样本只看指令文本就能答对，
`A − A′` 只在 counterfactual 切片上才有解释力，池化后的那个数会被稀释到看不出东西。

物体词表已填入 `config.json` 的 `allowed_objects`（`ALFRED_OBJECT_CLASSES ∪
ALFRED_RECEPTACLE_CLASSES`，去重后 75 类；总纲写的"58+26=84"没有扣除两个集合的重叠）。
`object_vocab_violation_rate` 因此不再是 `null`。

**判读会在 D 不达标时短路。** 总纲 §6.1 的判读表规定 D 低时其余条件的对比无意义，
`diagnose()` 现在真的会停在那里，只返回一条结论。上一版会继续追加"模型没有有效使用图像"
这类结论，紧跟在"其余条件先别看"后面。

`diagnosis_thresholds` 是 `d_min_score=0.6`、`floor_score=0.15`。旧的 `0.8 / 0.25`
里，`floor_score=0.25` 对一个随机基线为 0 的精确匹配指标来说几乎必然触发，起不到判别
作用。这两个数是可调的判定口径，不是测量结果。

如果 A/B/C 全部触发 floor effect，报告只提示进入约 300 step 短 LoRA；短 LoRA 训练不属于本目录的零样本推理代码。
训练数据转换、服务器环境安装和 ms-swift 启动入口见 `../training/README.md`。
