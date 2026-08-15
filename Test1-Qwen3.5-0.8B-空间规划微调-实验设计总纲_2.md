# Test 1:Qwen3.5-0.8B 空间规划能力微调 — 实验设计总纲

> 汇总自 2026-08-14 的方案讨论。所有数字均来自官方 config、模型卡或论文原文,可追溯至文末来源。
> **v2 修订**:数据方案改为**自建 VQA 为主**,ALFRED 降级为仅提供词表与任务模板;新增反事实配对设计、单图 planning VQA 数据集调研(附录 A)、双变量 Pilot 设计。
> **v3 修订**:Exp 0 诊断实验重写为**五条件检测**(推理时消融,非训练)。新增完整 oracle 下限检查 D 与错误图像基线 A′;把"空间事实 oracle"与"子目标 oracle"拆开;补充序列化格式混淆变量的处理;明确仅在全部贴地板时才升级为短 LoRA 重测。
> **v4 修订**:**场景域拍板为 B(单房间可见范围),不再做域选型 Pilot**。Exp 1 收窄为图像来源与渲染分辨率两个变量;§4.2 补入单房间约束;§4.6、§9 相应调整。
> **v5 修订**:任务从 `image + instruction → action plan` 改为**结构化 embodied CoT**;用 `<state>` / `<subgoal>` / `<plan>` 分别监督视觉空间理解、高层任务分解与原子动作执行;主实验改为 action-only / subgoal / full embodied CoT 三组递进消融。
> 状态:Exp 0 零样本诊断已完成;embodied reasoning 训练数据与训练接口待按 v5 规格统一。

---

## 0. 一句话概括

微调 Qwen3.5-0.8B,使其在给定**单张室内场景图 + 一条目标指令**时,学习从**视觉空间理解→任务分解→动作执行**的完整 embodied reasoning 链路,并输出可执行动作序列;通过 FT 前后与中间监督消融对比,量化空间 grounding 与多步规划的提升幅度。

---

## 1. 任务设定

| 项 | 取值 |
|---|---|
| **数据格式** | **Embodied reasoning**:(单图, 单条 prompt, state, subgoal, action) 五元组。无视频、无多帧、无动作历史;仿真轨迹仅用于生成与验证标签 |
| **输入** | 单张室内场景渲染图 + 一条明确目标指令("从冰箱拿瓶水"级别) |
| **输入不含** | 任何坐标、bbox、深度、场景图等结构化位置信息 |
| **输出** | **Structured embodied CoT**:`<state>` 任务相关空间事实 + `<subgoal>` 高层任务分解 + `<plan>` 原子动作序列 |
| **能力范围** | 定性空间关系(上/下/左/右/远/近、在……里/上)+ 任务状态识别 + 高层任务分解 + 原子动作执行 |
| **明确排除** | 定量距离估计、意图推断、机器人本体接入、真机成功率评估 |
| **项目定位** | 纯 VLM 的 embodied reasoning 能力测试。任务形式为具身式规划,但不接本体、不做 VLA |
| **训练方式** | LoRA |
| **数据规模** | 约 10 万条 |
| **场景域** | **已拍板:单房间可见范围**(目标物体与容器均在画面内),不做域选型 |
| **数据来源** | **自建为主**,公开数据集作可选补充(见 §4.6) |
| **算力** | 用户自有 GPU |

### 1.1 输出格式

```
<state>
Apple is inside Fridge.
Fridge is closed.
Table is the destination.
</state>

<subgoal>
Acquire the apple.
Move the apple to the table.
Place the apple.
</subgoal>

<plan>
GotoLocation(Fridge)
OpenObject(Fridge)
PickupObject(Apple)
GotoLocation(Table)
PutObject(Apple,Table)
</plan>
```

**三个区块的监督职责**:

- `<state>`:只输出与当前任务有关、可从仿真器状态验证的空间事实与物体状态,监督视觉空间理解。
- `<subgoal>`:输出与具体动作 API 解耦的高层任务分解,监督多步规划。
- `<plan>`:输出封闭 action vocabulary 内的可执行原子动作,继续作为主评测对象。

不再输出自然语言 summary;新的中间标签必须 simulator-grounded,而不是对 `<plan>` 的自由复述。

> 上例仅展示三个 block 的序列化格式。正式数据仍必须满足 §4.2 的可见性约束:若关闭的 Fridge 使 Apple 无法从当前图像观测,则不得把 `Apple is inside Fridge.` 作为该图的训练标签。

### 1.2 动作词表(沿用 ALFRED 定义,不用其数据)

**高层子目标 7 类**:`GotoLocation` `PickupObject` `PutObject` `SliceObject` `CleanObject` `HeatObject` `ToggleObject`

**补充(显式开关门)**:`OpenObject` `CloseObject`(取自 ALFRED 低层动作集)

**物体词表**:58 object classes + 26 receptacle classes = **84 个,严格封闭**

理由:已验证覆盖 7 种家务任务类型;封闭词表使精确匹配自动打分可行;与 AI2-THOR 资产命名天然对齐。

### 1.3 为什么砍掉定量能力是对的

SpatialVLM 报告:定量距离估计**仅 37.2% 的答案落在 ground truth 的 0.5x–2x 区间**。这是连大模型都做不好的方向,0.8B 更无胜算。定性任务可做成受限输出,自动打分容易,可学性高一个量级。

---

## 2. 模型事实(全部核实自官方 config.json 与文档)

### 2.1 架构

**关键点:Qwen3.5-0.8B 不是纯文本模型,是 VLM。** `architectures = ["Qwen3_5ForConditionalGeneration"]`

**视觉侧**

| 项 | 值 |
|---|---|
| ViT 层数 | 12 |
| hidden_size | 768 |
| patch_size | 16 |
| spatial_merge_size | 2 |
| out_hidden_size | 1024 |
| 特殊 token | image 248056 / video 248057 / vision_start 248053 / vision_end 248054 |

**语言侧**

| 项 | 值 |
|---|---|
| 层数 | 24 |
| hidden_size | 1024 |
| intermediate_size | 3584 |
| vocab_size | 248,320 |
| tie_word_embeddings | **true** |
| 词嵌入参数 | 254.3M(**占总量约 32%**) |
| transformer 主体 | ≈ 0.55B |
| max_position_embeddings | 262,144 |
| MTP | `mtp_num_hidden_layers = 1` |
| License | Apache 2.0 |

**混合注意力层排布**(`full_attention_interval = 4`)

- **18 层 linear_attention(Gated DeltaNet)**:16 key heads / 16 value heads,head_dim 128,`linear_conv_kernel_dim 4`,`mamba_ssm_dtype float32`
- **6 层 full_attention**(索引 **3 / 7 / 11 / 15 / 19 / 23**):Q 8 头、KV 2 头,head_dim 256,`attn_output_gate true`

**位置编码**:mRoPE,`mrope_section [11,11,10]`,`mrope_interleaved true`,`partial_rotary_factor 0.25`,`rope_theta 1e7`

### 2.2 三个直接影响实现的结论

**(1) LoRA target_modules 是陷阱。** 照抄稠密 Qwen3 的 `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` **只会命中 6/24 层**,18 层 GatedDeltaNet 完全没训到 → "微调前后提升幅度"这个结论本身会失真。
→ **用 `all-linear`**(ms-swift 官方推荐),并在日志里核对实际命中的层数。

**(2) 显存瓶颈在 logits,不在权重。** 248K 词表使 LM head 输出成为主要占用:

| batch × seq | logits bf16 | CE 升 fp32 额外 |
|---|---|---|
| 1 × 2048 | 0.95 GiB | 1.89 GiB |
| 4 × 2048 | 3.79 GiB | 7.58 GiB |
| 8 × 2048 | 7.58 GiB | 15.16 GiB |

→ **必须用 liger-kernel 融合 CE 或 chunked CE**,否则 batch 开不大。

**(3) 不要手写 collator。** position id 走 mRoPE,一律通过 processor / chat template 生成输入。

---

## 3. 训练配置

### 3.1 依赖栈

| 包 | 要求 | 备注 |
|---|---|---|
| transformers | **5.x 强制** | v4 不工作;ms-swift 要求 `>=5.9` |
| flash-linear-attention | `>=0.4.2` | 缺失会**静默**回退到慢速 PyTorch 实现 |
| causal-conv1d | 建议装 | 同上 |
| flash-attn | `2.8.3` | 配 `--attn_impl flash_attention_2` |
| liger-kernel | 必需 | 融合 CE,见 §2.2(2) |
| peft / trl | 最新 | — |

**坑**:
- **不要用 QLoRA 4-bit**。Unsloth 明确指出 Qwen3.5(dense 与 MoE)量化误差高于常规。
- Triton Mamba kernel **首次编译耗时明显**,别以为卡死。

### 3.2 显存参考

| 方案 | 显存 | 来源 |
|---|---|---|
| bf16 LoRA | ≈ 3 GB | Unsloth |
| 全参微调 | ≈ 12 GB(约 4×) | Unsloth |

→ 24GB 单卡即可做全参微调。本次既定用 LoRA,则务必固定并记录 rank 与 target_modules 这两个混淆变量。

### 3.3 超参起点(ms-swift 官方 Qwen3.5 示例)

```
--tuner_type lora
--lora_rank 8 --lora_alpha 32
--target_modules all-linear
--learning_rate 1e-4
--per_device_train_batch_size 4
--max_length 2048
--group_by_length true
--torch_dtype bfloat16
--use_liger_kernel true
--attn_impl flash_attention_2
--add_non_thinking_prefix true
--loss_scale ignore_empty_think
```

结构化 embodied CoT 比 action-only 的目标序列更长,`max_length` 需按 `<state> + <subgoal> + <plan>` 的长度分布留余量,并分实验记录 assistant token 数。

### 3.4 Intermediate Supervision:Embodied CoT

**训练目标**:让模型在同一个 causal LM 目标内显式学习三段串联链路。

输入:

```
image + instruction
```

输出:

```
<state>
task-relevant spatial understanding
</state>

<subgoal>
high-level task decomposition
</subgoal>

<plan>
primitive action execution
</plan>
```

第一版不引入额外 loss weighting。概念上将 assistant token 的监督分为:

\[
L = L_{state} + L_{subgoal} + L_{action}
\]

实现上仍直接使用 ms-swift 的 causal LM loss:屏蔽 system / user 侧 token,对 assistant 输出中 `<state>`、`<subgoal>`、`<plan>` 的所有 token 统一计算交叉熵,不新增 loss head、分段权重或自定义 trainer。

**实现契约**:`training/data.py` 中当前 assistant 侧的 `<plan> + <summary>` 序列化需统一为本节的 `<state> + <subgoal> + <plan>`;三组训练消融从同一份 canonical sample 按需选择区块,不各自生成一套标签。

---

## 4. 数据方案:自建 embodied reasoning 数据为主

### 4.1 为什么必须自建

系统调研了 40+ 个候选数据集(完整结果见**附录 A**)。结论:

**没有现成数据集同时提供"单图 + 一句 prompt → 可仿真验证的 state + subgoal + plan"。**即使放宽到只要求多步 plan,严格匹配也只有 4 个,且没有一个能直接用于本实验:

| 数据集 | 规模 | 致命问题 |
|---|---|---|
| ECoT (Bridge V2) | ~250 万 | **域不对** — tabletop / WidowX 机械臂第三人称,非家居室内 |
| EmbodiedBench EB-Alfred/Habitat | 12.9 GB | 标签是**模型 rollout**(含失败),非 gold;许可未声明 |
| RREP | 4,245 / 43,898 | **规模不够**;Gemini 蒸馏标签 |
| EgoThink planning 子集 | **100 条** | 评测级规模,无法训练 |

**ALFRED 为什么不能用**(v1 曾把它列为第一层,已推翻):

1. **它是轨迹数据不是 VQA。** 转成 (初始帧, goal, 高层序列) 只得 **8,055 个独立场景**;25,743 条语言标注只是同一场景的三种措辞。离 10 万差一个数量级。取中间帧扩样本则同 episode 各帧共享 plan 后缀,有效样本数远低于名义数。
2. **信息不充分。** 初始帧看不到冰箱在哪(可能在另一房间),plan 第一步却是 `GotoLocation(Fridge)` → 不可约标签噪声。
3. **最致命:plan 大多能从 goal 文本猜出,不用看图。** "Rinse off a mug and place it in the coffee maker" → Goto/Pickup/Clean/Put 由 7 种任务模板决定,视觉贡献极小。训出的涨幅会主要来自语言侧模板记忆,而非空间推理 —— 整个实验信噪比会很低。

→ **ALFRED 只保留 §1.2 的词表与任务模板,数据自建。**

### 4.2 生成管线

用 AI2-THOR / ProcTHOR 直接生成 embodied reasoning 五元组。标签生成链为:

```text
AI2-THOR / ProcTHOR
        ↓
Environment state
        ↓
Task-relevant spatial facts
        ↓
High-level planner
        ↓
Subgoals
        ↓
Execution trajectory
        ↓
Primitive actions
```

具体流程:

1. 在 ProcTHOR 场景中把相机置于**某个房间内**的某视角,渲染一张图。
2. **从该视角当前可见的物体集合中选取目标与容器** ← 关键步骤,保证信息充分。
3. 从仿真器 environment state 确定性导出与当前指令相关的空间关系、容器开关、物体所在位置等 `<state>` 事实。
4. 用 high-level planner 生成与低层 API 解耦的 `<subgoal>` 任务分解,再由 PDDL planner / 规则编译为候选 primitive action sequence。
5. **在仿真器中执行并验证成功**,从成功 execution trajectory 记录最终 `<plan>`;失败样本丢弃。
6. 输出 `(图, prompt, state, subgoal, action)` canonical sample,其中 action 字段在模型输出中序列化为 `<plan>`。

**场景域约束(v4 拍板)**:限定为**单房间可见范围**。生成时必须满足:

- 目标物体与所需容器**全部在当前视角画面内**(可见性由仿真器的 `visible` / `isVisible` 属性判定,不靠面积阈值猜)
- plan 不含跨房间导航;`GotoLocation` 的目标必须是画面内可见对象
- 若某视角下找不到满足条件的任务,**丢弃该视角重采样**,不要降级成"目标不可见"的样本

理由:目标不可见会引入**不可约标签噪声**——标签是对的,但输入信息不足以推出它(这正是 ALFRED 不能用的原因之一,见 §4.1)。单房间可见范围保证视觉必须参与,是本实验成立的前提。

**得到的性质**:

- 天然单图单 prompt 格式,轨迹只用于导出与验证监督信号,不作为模型输入
- **样本独立** — 每张图是独立场景配置 + 独立视角
- **信息充分** — 目标可见,视觉必须参与
- **标签可验证正确** — 不只是精确,而是仿真器跑通过的
- **规模不受限** — ProcTHOR-10K:10,000 房屋(+1k val +1k test)、1,633 个可交互物体实例 / 108 类、16 种房型、每房 1–10 间、18 组语义资产组;材质可随机化(40+ 纯色、122 墙面纹理、55 地面材质)。10 万条轻松

### 4.3 核心设计:反事实配对(counterfactual pairs)

**这是现成数据集给不了、而可控生成独有的能力,也是本实验成败的关键。**

同一句 prompt,两张或多张图,场景配置不同导致正确 state 与 plan 不同。例如对指令"把水瓶放进冰箱":

| 场景配置 | 关键 `<state>` | 正确 `<plan>` |
|---|---|---|
| 水瓶在桌上,冰箱**开着** | `WaterBottle is on Table.` `Fridge is open.` | `GotoLocation(Table)` `PickupObject(WaterBottle)` `GotoLocation(Fridge)` `PutObject(WaterBottle,Fridge)` |
| 水瓶在桌上,冰箱**关着** | `WaterBottle is on Table.` `Fridge is closed.` | `GotoLocation(Table)` `PickupObject(WaterBottle)` `GotoLocation(Fridge)` `OpenObject(Fridge)` `PutObject(WaterBottle,Fridge)` `CloseObject(Fridge)` |
| 水瓶在台面上,冰箱**开着** | `WaterBottle is on CounterTop.` `Fridge is open.` | `GotoLocation(CounterTop)` `PickupObject(WaterBottle)` `GotoLocation(Fridge)` `PutObject(WaterBottle,Fridge)` |

**模型不看图就必然答错一部分。这等于在数据层面消除语言先验捷径**,而不是靠事后用 format-only 对照组去扣。

建议:**至少 30% 的样本以反事实对的形式出现**(同 prompt、不同配置、不同 plan),并在评测集中单独切出一个 counterfactual 子集报分——这个子集上的分数最能反映真实空间推理能力。

### 4.4 样本 schema 建议

```json
{
  "image": "scene_00123_view_02.png",
  "prompt": "把桌上的水瓶放进冰箱",
  "state_facts": [
    "WaterBottle is on Table.",
    "Fridge is closed.",
    "Fridge is the destination."
  ],
  "subgoals": [
    "Acquire the water bottle.",
    "Move the water bottle to the fridge.",
    "Make the fridge accessible.",
    "Place the water bottle."
  ],
  "plan_actions": ["GotoLocation(Table)", "PickupObject(WaterBottle)",
                   "GotoLocation(Fridge)", "OpenObject(Fridge)",
                   "PutObject(WaterBottle,Fridge)", "CloseObject(Fridge)"],
  "_meta": {
    "scene_id": "procthor_00123",
    "counterfactual_group": "cf_0451",
    "target_visible": true,
    "receptacle_state": "closed",
    "spatial_relations": [["Fridge","right_of","Agent"], ["WaterBottle","on","Table"]],
    "plan_length": 6,
    "sim_verified": true
  }
}
```

`state_facts`、`subgoals`、`plan_actions` 都是 assistant 侧监督标签;`_meta` 不进训练损失,只用于**数据验证、评测归因与切片分析**。三组实验必须从这份统一 schema 派生,保证它们只在 supervision 上不同。

### 4.5 规模与配比

| 组成 | 占比建议 | 说明 |
|---|---|---|
| 自建 embodied reasoning 数据(含反事实对) | 主体 | §4.2–4.3;同时提供 state / subgoal / action 三层标签 |
| **空间关系辅助任务** | 混入 | EmbSpatial 方法:用相机参数 + 3D 坐标直接导出 above/below/left/right/close/far,**不依赖检测器**。先例:EmbSpatial-SFT **仅 25,000 条**就让 MiniGPT-v2 提升 **34.25 个百分点** |
| **通用多模态指令数据** | 按比例混入 | 防灾难性遗忘。EgoPlan-IT 先例:5 万 in-domain + 5 万辅助 + **164K 通用**。纯 in-domain 10 万条会遗忘 |

### 4.6 可选补充数据集(非主力)

| 数据集 | 规模 | 用法 | 代价 |
|---|---|---|---|
| **ShareRobot** (BAAI/RoboBrain) | **1,027,990** planning QA | 输入是帧区间 `<image 0-25>`,**只取每区间最后一帧即可转成单图格式** —— 量最大的可转换来源 | OXE 机器人视角(非家居)、348 GB、**许可未声明** |
| **ECoT** (Bridge V2) | ~250 万 | **域不匹配(tabletop,已被 v4 排除)**,仅作 `PLAN:` 字段的格式参考;**MIT 许可** | 非单房间家居场景;Gemini 1.0 蒸馏标签 |
| **EmbodiedBench EB-Alfred** | 12.9 GB | AI2-THOR 室内家居,有 `executable_plan`;需按 `success` flag 过滤 | 模型 rollout 标签,许可未声明 |
| **RREP** | 4,245 / 43,898 | AI2-THOR 单图 + `executable_plan`,可作少量补充 | 规模小,许可未声明 |
| **EmbSpatial-SFT** | 25,000 | 空间关系辅助任务,已验证有效 | 无 plan 输出 |

### 4.7 中间监督标签的质量约束

任务现在是三段能力串联:**视觉定位与空间关系提取 → 高层任务分解 → 原子动作生成**。因此中间标签不再只是归因用 metadata,而是进入 causal LM loss 的正式监督信号。

生成时必须满足:

- **State groundedness**:每条 `state_facts` 都可从当帧可见物体、仿真器属性或准确 3D 关系确定性导出;只保留解决当前指令所需的事实。
- **Subgoal abstraction**:`subgoals` 表达高层意图与阶段边界,不应只把 action API 逐行翻译成自然语言;同时必须足以解释后续 plan。
- **Action executability**:`plan_actions` 必须处于封闭动作词表内,并由仿真器完整执行成功。
- **Cross-layer consistency**:`state_facts → subgoals → plan_actions` 必须逻辑一致;反事实组内的差异必须能追溯到一个明确环境状态变化。

`_meta.spatial_relations`、原始 scene graph 与 execution trace 仍不计入损失,用于自动校验和切片归因。EmbodiedBench 的失败分类可继续借用:**感知错误 / 推理错误 / 规划错误**。

---

## 5. 评测设计

### 5.1 三轨评测

| 轨 | 用途 | 内容 |
|---|---|---|
| **In-domain** | 测有效性 | 自建 held-out(场景不重叠) |
| **Counterfactual** | **测真实视觉依赖** | 自建反事实子集(同 prompt / 不同配置),**最关键的一轨** |
| **OOD** | 测泛化 | 公开 benchmark(见 §5.2) |

只有 in-domain 一轨的话结论无法解释。

### 5.2 候选公开 benchmark

| Benchmark | 规模 | 参考分数 | 人类 | 随机 | 说明 |
|---|---|---|---|---|---|
| **EmbSpatial-Bench** ⭐ | 3,640 MCQ / 2,181 图 | Qwen-VL-Max **49.11%** | 90.33% | 25% | 六种定性关系;图像来自 MP3D + AI2-THOR + ScanNet;QA 由 3D 标注自动导出 |
| **Spatial-DISE** | 559 MCQ | 开源均 **26.2%**;Qwen2.5-VL-7B 微调后 47.0% | 76.8% | 24.8% | 2×2 象限;**已知跨象限迁移极小** |
| **MFE-ETP** | 100 planning cases | — | — | — | 输出格式高度对齐:`navigate_to(freezer), open(freezer), place_inside(apple, freezer), close(freezer)`;多帧输入但可作参考;**MIT** |
| ViewSpatial-Bench | — | — | — | — | 多视角空间定位,备选 |
| LRR-Bench | 每任务 200 | **3D 任务接近 0**;GPT-4o Camera Movement 仅 16% | ~90% | — | **headroom 最大但对 0.8B 大概率学不动**,不建议作主指标 |

⭐ **EmbSpatial-Bench 为首选**:纯定性、四选一可自动打分、与训练数据同域、且已有 +34.25 点的可涨性实证。

### 5.3 打分方式

- **动作序列(主指标)**:保持原 benchmark 定义,同时报 **sequence-level exact match (Action EM)** 与 **step-level match**。三组训练实验的主结论只根据 `<plan>` 评测,确保与 action-only baseline 直接可比。
- **State(诊断指标)**:对归一化后的任务相关事实报 precision / recall / F1,并单独统计物体属性与空间关系。
- **Subgoal(诊断指标)**:报归一化后的 subgoal step match 与顺序一致率,用于区分感知改善和规划改善,不取代 action 主指标。
- **格式与可执行性**:加报三个区块的结构合法率、动作/物体词表越界率,以及对预测 `<plan>` 的仿真器执行成功率(若评测环境支持)。

---

## 6. 研究假设与实验设计

### 6.1 研究假设

#### H1:intermediate embodied reasoning 是当前主要瓶颈

模型当前的主要瓶颈不是已知子目标后的 action generation,而是从图像恢复任务相关状态并形成高层规划的中间链路。

依据是已完成的 Exp 0 零样本诊断:子目标 oracle 条件 D 的 Action EM 为 **83.77%**,而 image condition A 仅 **0.88%**、三种结构化场景描述 B 仅 **0.88%–2.19%**、image + spatial facts 条件 C 仅 **1.75%**。这表明模型在给定高层任务结构后能生成动作,但无法稳定自行完成感知到规划的映射。

#### H2:subgoal supervision 提升多步规划

在图像、指令、数据分布和训练配置相同时,增加 `<subgoal>` 监督应当比 action-only SFT 获得更高的 Action EM 与 step-level match,改善应在长 plan 和 counterfactual 子集上更明显。

#### H3:spatial state supervision 进一步提升视觉空间推理

在 subgoal supervision 之上增加 simulator-grounded `<state>` 监督,应当进一步提升视觉条件下的 Action EM、step-level match 与 counterfactual 一致性,并在 state fact F1 上体现直接改善。

### 6.2 总体实验矩阵

| # | 实验 | assistant 输出 | 目的 | 状态/规模 |
|---|---|---|---|---|
| **Exp 0** | **瓶颈诊断** | oracle 推理时条件 | 建立 H1,定位感知/规划/动作生成瓶颈 | **已完成**:5 个语义条件,228 条;其中 B 含 3 种序列化 |
| **Shared Pilot** | **图像来源与分辨率** | 固定 action-only | 为三组主实验选定共享图像配置,不参与 supervision 结论 | 每组 2k 条短 LoRA |
| **Exp 1** | **Action-only SFT** | `<plan>` | baseline:`image + instruction → action` | 约 10 万条 |
| **Exp 2** | **Subgoal supervision** | `<subgoal> → <plan>` | 验证 H2:planning supervision 是否有效 | 与 Exp 1 同样本 |
| **Exp 3** | **Full embodied CoT** | `<state> → <subgoal> → <plan>` | 验证 H3:视觉 grounding + planning 联合监督 | 与 Exp 1 同样本 |
| **Aux** | **format-only 对照** | 与对应实验同格式、错误标签 | 量出仅学会输出格式的涨幅 | 资源允许时运行,不占 Exp 1–3 编号 |

### 6.3 Exp 0 — 诊断检测(零样本阶段已完成)

**性质:推理时消融检测,不训练。** 所有 oracle 信息仅用于诊断,不进入最终 pipeline。

**要回答的问题**:Qwen3.5-0.8B 的 vision tower 只有 12 层 / hidden 768,而它现在要**独自承担全部空间信息提取**(输入不含任何坐标)。链路上的瓶颈在哪一段?这直接决定 LoRA 是否需覆盖 vision tower。

#### 五条件阶梯(按执行顺序)

| 顺序 | 条件 | 模型输入 | 诊断能力 | 读什么 |
|---|---|---|---|---|
| 1 | **D 完整 oracle** | 正确场景图 + 正确子目标 | 纯结构生成 | **下限检查**。D 低 → 输出格式/词表/结构合法性有问题,其余条件先别看 |
| 2 | **A′ 错误图像** | **不相关场景图** + 正确指令 | 语言先验基线 | `A − A′` = **视觉的真实贡献量** |
| 3 | **A 完整任务** | 图像 + 指令 | 感知 + reasoning + planning | 全链路基线 |
| 4 | **B 感知 oracle** | 场景图/结构化描述 + 指令(**无图**) | reasoning + planning | 去掉感知后能恢复多少 |
| 5 | **C 空间事实 oracle** | 图像 + 指令 + **正确空间事实**(**不给子目标**) | 感知后的 reasoning + planning | 感知损失有多少能被空间事实补回来 |

#### 两条关键的设计约束

**(1) 空间事实 oracle 与子目标 oracle 必须拆开,不能一起给。**

这个任务里 reasoning 与 planning 本来就难分离——"看到冰箱关着 → 所以要先开门"这个推断**本身就是 planning**。在封闭词表下,子目标 → 动作序列几乎是查表。**同时给出正确子目标等于给出答案**,该条件会贴近满分,测出的是"能不能照抄"而非"会不会规划"。

因此:
- **C 只给空间事实**(冰箱在右前方 / 冰箱是关着的 / 水在冰箱里)
- **D 才给子目标**,且 D 的定位是**下限检查**而非能力测量

**(2) 场景图的序列化格式是隐藏混淆变量。**

B 高于 A,可能只是因为场景图写法恰好对模型口味,不代表感知是瓶颈。
→ **B 至少用三种序列化格式各测一次**(自然语言 / JSON / 三元组),取一致的结论;若三者分歧大,在结论中明确标注这是 confound。

#### 判读规则

| 观察 | 结论 | 行动 |
|---|---|---|
| **D 低** | 输出层就不行 | 先修词表/格式/结构合法率,其余条件的对比无意义 |
| **A ≈ A′** | **模型压根没在用图** | 视觉是死的 → LoRA 必须覆盖 vision tower,甚至考虑冻结 LLM 只训 ViT + projector。此时 oracle 阶梯可不必跑完 |
| **B ≫ A** | 感知瓶颈 | LoRA 覆盖 vision tower |
| **B ≈ A 且两者都低** | reasoning / planning 瓶颈 | LoRA 语言侧即可 |
| **C 接近 B** | 感知损失可被空间事实完全补偿 | 感知是唯一瓶颈 |
| **C 显著低于 B** | 图像的存在反而干扰 | 检查视觉 token 与文本事实的融合方式 |

#### 执行与升级路径

1. **零样本诊断已完成**:228 条样本;D / A′ / A / C 各一种输入,B 按自然语言 / JSON / 三元组三种序列化分别运行。
2. **结果**:D 的 Action EM 为 83.77%,A 为 0.88%,A′ 为 0.00%,B 为 0.88%–2.19%,C 为 1.75%。D 通过下限检查,但 A / B / C 全部贴地板。
3. **下一步升级**:各条件跑一次约 300 step 短 LoRA 后重测,用于检查条件间的可学性差异。

第 3 步是必要的兜底——base 模型在所有条件下都不会做,不代表这些条件之间没有可学性差异。零样本结果是 H1 的当前依据,但短 LoRA 重测不替代 Exp 1–3 的主消融。

这一诊断链后续只决定 LoRA 是否覆盖 vision tower,不改变 Exp 1–3 的 supervision 定义。

### 6.4 Shared Pilot — 图像来源与分辨率(不再选域)

> **v4:场景域直接定为 B(单房间可见范围),取消域选型。** 理由见 §4.2 的场景域约束——目标不可见会引入不可约标签噪声,全屋长程与 tabletop 均不满足本实验的输入设定。Pilot 只保留下面两个变量。

**变量一:图像来源**

| 来源 | 说明 | 获取摩擦 |
|---|---|---|
| **仿真渲染**(AI2-THOR / ProcTHOR) | 主力。3D 标注精确,可见性可判定,反事实对可控 | **最低**(`pip install ai2thor`) |
| 真实扫描(ScanNet / MP3D) | 泛化补充。需签协议邮件申请 | 中(数天),可并行发起 |
| 真实第一人称(Epic-Kitchens / Ego4D 系) | 泛化补充。无 3D gold,oracle 需另建 | 中 |

**变量二:渲染分辨率**

Qwen3.5 的 ViT 是 patch 16 + spatial_merge 2,低分辨率下剩余视觉 token 很少(ALFRED 默认仅 300×300,明显不够)。建议至少测 **300 / 512 / 768** 三档,确认空间细节是否够支撑定性关系判断。见 §7 风险 4。

**流程**:为避免和主消融交互,所有 Pilot 统一使用 Exp 1 action-only 输出。每组 2k 条(严格 held-out 200 条 + counterfactual 100 条)→ 各跑 ~500 step 短 LoRA → 测三轨(§5.1)→ **看斜率而非绝对分**,取斜率最大且 OOD 与 counterfactual 都不塌的组合。单卡约 30 分钟/次。

**注意**:两个变量应先各自单独扫,确认无强交互后再定组合;不要一上来跑全笛卡尔积。

### 6.5 Exp 1–3 的共享控制条件

- 使用同一批 canonical samples、同一 scene/counterfactual-group 切分、同一训练 seed 与同一评测集。
- 保持 Qwen3.5-0.8B、LoRA target / rank、optimizer、learning rate、batch size、训练 step 和图像配置不变;唯一主变量是 assistant 侧 supervision block。
- 三组的 `<plan>` gold 完全一致,action vocabulary 与 Action EM / step-level match 解析器完全一致。
- 因 Exp 2 / Exp 3 的 assistant 序列更长,除了固定样本数与 step,还要报告实际 supervised token 数和训练时间,防止把额外 token 预算当成 supervision 结构的收益。

### 6.6 Exp 1 — Action-only SFT

Baseline:

```text
image + instruction
        ↓
<plan>
primitive action sequence
</plan>
```

本组不输出 `<state>` 或 `<subgoal>`,用于复现旧的直接 action supervision,是 Exp 2 和 Exp 3 的统一 baseline。

### 6.7 Exp 2 — Subgoal Supervision

```text
image + instruction
        ↓
<subgoal>
high-level task decomposition
</subgoal>
        ↓
<plan>
primitive action sequence
</plan>
```

本组只增加 planning supervision,不输出 `<state>`。`Exp 2 − Exp 1` 主要检验 H2。

### 6.8 Exp 3 — Full Embodied CoT

```text
image + instruction
        ↓
<state>
task-relevant spatial facts
</state>
        ↓
<subgoal>
high-level task decomposition
</subgoal>
        ↓
<plan>
primitive action sequence
</plan>
```

本组联合监督视觉 grounding、高层规划与动作执行。`Exp 3 − Exp 2` 主要检验 H3;`Exp 3 − Exp 1` 衡量完整 embodied CoT 的总体收益。

### 6.9 主实验判读

- H2 的主检验是 Exp 2 相对 Exp 1 的 Action EM / step-level match 改变,并分别报 in-domain、counterfactual、OOD 和 plan-length 切片。
- H3 的主检验是 Exp 3 相对 Exp 2 的动作指标改变;state fact F1 和 subgoal 指标只用于解释改变来自哪一层。
- 对同一 held-out sample 做配对统计,报告 bootstrap 置信区间;不只比较单一 seed 的点估计。

### 6.10 format-only 辅助对照

该对照不再占用 Exp 3 编号。在资源允许时,对相应实验使用**相同 block 格式但在训练 split 内打乱 state / subgoal / action 标签**的数据微调,量出"仅学会输出格式"可能带来的涨幅。反事实配对(§4.3)用于消除语言先验捷径,format-only 对照用于量化输出格式适配,两者职责不同。

---

## 7. 风险清单

| # | 风险 | 依据 | 对策 |
|---|---|---|---|
| 1 | **Floor effect** | Spatial-DISE:开源 VLM 平均 26.2%,随机 24.8%,人类 76.8%;EgoPlan-Bench:GPT-4V 仅 37.98% | 三轨评测 + format-only 对照;0.8B 基线大概率贴随机线,单看绝对分无意义 |
| 2 | **域特化,跨域不迁移** | Spatial-DISE:微调后象限特化,跨象限迁移极小;Vlaser:域外具身推理提升**不迁移**到闭环,人工标注域内数据远优于互联网规模预训练数据 | 训练数据必须覆盖多种空间关系模式;eval 必须含 OOD 轨 |
| 3 | **模型靠语言先验走捷径** | ALFRED 分析(§4.1 第 3 条) | **反事实配对(§4.3)** + format-only 对照 |
| 4 | **渲染分辨率不足** | Qwen3.5 ViT 为 patch 16 + spatial_merge 2;ALFRED 默认仅 300×300 | 提高渲染分辨率;在 Pilot 中作为变量测 |
| 5 | **灾难性遗忘** | EgoPlan-IT 配方中通用数据占比很高 | 按比例混入通用多模态指令数据 |
| 6 | **0.8B 三区块输出合法性不稳定** | Exp 0 已验证 `<plan>` 格式,但 `<state> + <subgoal> + <plan>` 尚未验证 | 词表严格封闭;先小样本验证三区块结构合法率再放量;评测加报分区块合法率 |
| 7 | **LoRA 只命中 6/24 层** | config.json 层排布 | `all-linear` + 日志核对命中层数 |
| 8 | **logits 显存爆炸** | vocab 248,320 | liger 融合 CE |
| 9 | **仿真渲染 sim-to-real gap** | Vlaser 域内/域外结论 | 材质光照随机化;OOD 轨用真实图像 benchmark 检验 |

---

## 8. 环境边界(本次讨论所用容器实测)

- 无 GPU(2 vCPU / 7GB RAM)
- 无 GL / Vulkan(`glxinfo`、`vulkaninfo` 均不存在;有 Xvfb 但无用)
- **pip 无法安装新包** — `ai2thor`、`prior`、甚至 `cowsay` 均报 "No matching distribution found",索引为受限白名单

→ AI2-THOR / ProcTHOR 渲染与整条数据管线**必须在自有 GPU 机器上运行**。

---

## 9. 仍待决定

~~场景域~~ — **已拍板:B 单房间可见范围**(v4)

1. **渲染分辨率** — 在 Pilot 中作为变量,建议扫 300 / 512 / 768
2. **反事实对占比** — 建议 ≥30%
3. **通用数据混入比例** — EgoPlan-IT 是 5万 : 5万 : 164K,可作起点
4. ~~**是否显式保留 `OpenObject` / `CloseObject`**~~ — **已拍板:保留**,action vocabulary 沿用 §1.2;反事实对依赖显式开关门动作。
5. **LoRA rank** — ms-swift 示例是 8;若想减少混淆变量可改用全参微调,0.8B 只需约 12GB
6. **A′ 错误图像的构造方式** — 当前 exp0 代码用"不相关图像";另一方案是复用反事实对(同 prompt、配对图、参考答案仍用原图 plan),更严格但需先有反事实数据。**未决,影响已写好的 exp0 代码**
7. **State / subgoal 标签语言与归一化规则** — 本文示例用英文,当前 Exp 0 中间字段主要为中文;正式生成 10 万条数据前必须固定一种序列化与评测 normalization。

---

## 附录 A:单图 planning VQA 数据集调研

筛选标准(需全部满足):输入为**一张静态图** + 一句自然语言指令;输出为**多步动作序列**(非单个动作、非选择题、非 bbox);公开可下载。

### A.1 严格匹配

| 数据集 | 规模 | 域 | 输出格式 | 标签来源 | 许可 |
|---|---|---|---|---|---|
| **ECoT** (Bridge V2) | ~250 万 transitions / 6 万轨迹 | Tabletop,WidowX 第三人称 | CoT 内含 `PLAN:` 枚举步骤 | Prismatic-7B + Grounding DINO + **Gemini 1.0** | **MIT** |
| **EmbodiedBench EB-Alfred / EB-Habitat** | 12.9 GB(行数未公布) | AI2-THOR 室内 / Habitat | JSON:`language_plan` + `executable_plan` | **模型 rollout**(含失败,有 success flag) | 未声明 |
| **RREP** | 4,245 SFT / 43,898 RFT | AI2-THOR 室内 | `executable_plan` + reasoning | Gemini-2.0-flash 蒸馏 | 未声明 |
| **EgoThink** planning 子集 | **100** | 第一人称 Ego4D | 散文式有序动作 | 人工(6 标注 3 复核) | Apache-2.0 |

ECoT 样本示例:

```
TASK: Place the watermelon on the towel.
PLAN: 1. Move to watermelon 2. Firmly grasp it 3. Move to towel 4. Place watermelon on towel.
SUBTASK: Move to the watermelon. SUBTASK REASONING: ...
```

### A.2 近失(可转换或可借鉴)

| 数据集 | 规模 | 失格原因 | 备注 |
|---|---|---|---|
| **ShareRobot** (BAAI) | **1,027,990** planning QA / 51,403 episodes × 30 帧 | 输入是帧区间 | **取最后一帧即可转严格匹配**;Gemini 分解 + 3 人复核;348 GB;许可未声明 |
| **Can-Do** | 400 | **未发布**("Coming Soon") | 格式最干净:人工标注有序 plan;18.5% 真实照片 + DALL·E 3 合成 |
| **MFE-ETP** | 100 planning | 多帧输入 | 输出格式高度对齐;人工标注;**MIT** |
| **PCA-EVAL / PCA-Bench** | 300 / +7,510 训练 | 输出是**单个动作** | 输入形式正确;MIT |
| **RoboInter-VQA** | 928,819 train | 8 帧视频 | DROID + RH20T |
| **RoboVQA** | 829,502 | 16 帧视频 | 论文自述 video VLM 比 single-image 错误率低 19% |
| **EgoCOT** (EmbodiedGPT) | Ego4D 派生 | 8 连续帧 | 输出确为编号 plan |
| **EgoPlan-IT / Bench** | 50,000 / 3,355+1,584 | 视频 + 单个动作 / MCQ | BSD-3-Clause |
| **MuEP** | 15,247 episodes | 需仿真器交互 | ALFWorld + AI2-THOR |
| **WAP** | 80,875 | **未发布** + 多帧历史 | — |
| **LLaRA** | 8k–660k | 单步输出 | 输入形式正确 |
| **VLABench** | 100 tasks / 1,600 轨迹 | 需 MuJoCo,4 视角 | — |
| **Emma-X-GCOT** | 6 万轨迹 | 输出是单步子任务 | MIT |
| **RoboBench** | 6,092 QA | MCQ | CC BY-SA 4.0 |
| **AsgardBench** | 108 task instances | 需仿真器 | Apache-2.0 |
| **FoMER Bench** | 1,112 | 多帧 + MCQ | — |
| **RoboBrain 2.0 / 2.5** | — | **训练数据未发布** | 模型权重 Apache-2.0 |
| **ALFRED / TEACh / PARTNR / LoTa-Bench** | — | 轨迹 / 对话 / 仿真 / 无图像 | 见 §4.1 |

### A.3 不符合

OpenEQA(短答案)、SQA3D(3D 场景输入)、M3CoT(MCQ,非动作域)、EmbodiedScan(3D 感知)、EO-Bench(全 MCQ)、EmbodiedAgentInterface(符号状态,无图像)。

### A.4 未能核实

- Vlaser-6M 的 `eb-alfred.jsonl` / `eb-habitat.jsonl` 分片格式(与严格匹配 #2 同源,可能也是严格匹配)
- PCA-Bench 精确实例数
- MMRo 的 planning 答案是开放式还是 MCQ
- ShareRobot / EgoCOT / EmbodiedBench 轨迹数据集的许可

---

## 来源

- [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json) · [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5) · [Unsloth Qwen3.5 微调指南](https://unsloth.ai/docs/models/qwen3.5/fine-tune) · [ms-swift Qwen3.5 最佳实践](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html)
- [ALFRED](https://arxiv.org/pdf/1912.01734) · [ProcTHOR](https://ar5iv.labs.arxiv.org/html/2206.06994) · [AI2-THOR](https://ai2thor.allenai.org/publications/)
- [EmbSpatial-Bench](https://aclanthology.org/2024.acl-short.33.pdf) · [Spatial-DISE](https://arxiv.org/html/2510.13394v3) · [LRR-Bench](https://arxiv.org/html/2507.20174v1) · [VGSI](https://arxiv.org/abs/2104.05845) · [MFE-ETP](https://arxiv.org/abs/2407.05047)
- [ECoT](https://arxiv.org/abs/2407.08693) · [ECoT 数据](https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge) · [ShareRobot](https://huggingface.co/datasets/BAAI/ShareRobot) · [RoboBrain](https://arxiv.org/abs/2502.21257) · [RREP](https://arxiv.org/abs/2505.22050) · [EmbodiedBench](https://arxiv.org/abs/2502.09560) · [EgoThink](https://arxiv.org/abs/2311.15596) · [Can-Do](https://arxiv.org/abs/2409.14277) · [PCA-Bench](https://arxiv.org/abs/2402.15527)
- [SpatialVLM](https://spatial-vlm.github.io/) · [Vlaser](https://arxiv.org/pdf/2510.11027) · [EgoPlan-Bench](https://arxiv.org/html/2312.06722v2) · [RoboVQA](https://arxiv.org/abs/2311.00899) · [EmbodiedGPT / EgoCOT](https://openreview.net/pdf?id=IL5zJqfxAa) · [ERQA-Plus](https://arxiv.org/html/2606.17639v2)
