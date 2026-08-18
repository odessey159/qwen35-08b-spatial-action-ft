# Qwen3.5-0.8B Spatial Action Fine-Tuning

This repository implements a **simulator-verified embodied planning** pipeline for **Qwen3.5-0.8B**: generate indoor household tasks in AI2-THOR / ProcTHOR, convert them to structured chain-of-thought (`<state>` / `<plan>` / `<action>`), full-parameter fine-tune the 0.8B vision-language model with ms-swift, and evaluate visual grounding with counterfactual open/closed pairs plus a swapped-image (A′) control.

The runnable training and evaluation stack is written for a Linux GPU server at `/root/qwen35-08b-spatial-action-ft`. This checkout also contains generated datasets, prepared JSONL, checkpoints, and evaluation reports from those runs.

The current root README is the technical source of truth for what is in the tree. Older operational notes remain in `exp0/README.md` and `training/README.md`; where those notes disagree with code, configs, or result files, **the implementation and machine-readable artifacts win**.

---

## Overview

**Problem.** A small multimodal model is asked to map a **single indoor RGB image** plus a **Chinese instruction** to a **closed-vocabulary primitive action sequence**. Many such instructions are solvable from text alone. The interesting cases are those where the correct plan depends on a visual state (especially whether a receptacle is open or closed).

**Approach.**

1. A rule-based generator samples ProcTHOR (and, for the 240-sample diagnostic set, iTHOR kitchens) scenes, renders one 672×672 image, executes the gold plan in AI2-THOR, and keeps only `sim_verified=true` rows.
2. Counterfactual pairs hold scene, camera, instruction, and objects fixed while flipping a receptacle between **open** and **closed**, which changes the gold action sequence.
3. Training labels are **not** LLM-written. Task-relevant state is extracted from the scene graph; high-level plan text is a deterministic abstraction of the primitive actions; the action block is the simulator-verified `gold.plan_actions`.
4. Fine-tuning is **full-parameter SFT** of Qwen3.5-0.8B (LLM, ViT, and aligner unfrozen) via ms-swift 4.4.2, with per-section `loss_scale` on `<state>` / `<plan>` / `<action>`.
5. Evaluation includes teacher-forced section losses, free-generation action exact match, counterfactual **pair exact**, and an A vs A′ image-swap gap.

```text
ProcTHOR / iTHOR (AI2-THOR)
        ↓
exp0.generate_data  (rule planner + simulator execution)
        ↓
samples.jsonl + images/   [optional: merge_shards / clean_raw_shards]
        ↓
training.cot_data prepare  (raw_simulator → ms-swift JSONL)
        ↓
swift sft  (full FT, bf16)
        ↓
generate_cpu_predictions / evaluate_in_domain_predictions
        + compare_counterfactual_predictions
        + evaluate_section_losses
```

A design note (`Test1-Qwen3.5-0.8B-空间规划微调-实验设计总纲_2.md`) originally described LoRA, tags `<state>` / `<subgoal>` / `<plan>`, and an ALFRED object count of 58+26=84. **The code does not follow those three items.** Implemented tags are `<state>` / `<plan>` / `<action>`; the completed 10K and 100K runs are **full** fine-tunes; the object vocabulary in `exp0/generate_data.py` unions to **75** types (see below).

---

## Key Features

- Rule-based, simulator-verified data generation (no LLM in the label path).
- Two counterfactual pair kinds, both stored as `task_group=counterfactual_put` with exactly two rows per group.
- Embodied CoT serialization with optional section loss weights.
- Connected-component train/val split on `scene_id` and `counterfactual_group`.
- Configured A/B/C response-format ablations (action-only / plan+action / full CoT). **Only the full-CoT 10K and 100K runs have training outputs in this checkout.**
- A format-only (`permuted_triplet`) control is implemented in code and config; the on-disk 100K prepared copy is **not** actually permuted (see [Known Issues](#known-issues--limitations)).
- Zero-shot Exp 0 diagnostic protocol with seven prompt conditions (D, A, A′, B×3, C).
- Section-level teacher-forced loss, in-domain free-generation metrics, and CF pair scoring.

---

## Repository Structure

```text
.
├── README.md                          # this file
├── requirements-generator.txt         # AI2-THOR 5.0 generator deps
├── requirements-exp0.txt              # local Exp 0 inference deps
├── requirements-exp0-server.txt       # server Exp 0 inference deps
├── requirements-training.txt          # ms-swift SFT deps
├── exp0/                              # generation, schema, Exp 0 eval
│   ├── generate_data.py               # AI2-THOR / ProcTHOR generator
│   ├── generate_cot_data.py           # rule CoT builder (state/plan/action)
│   ├── subgoal_abstraction.py         # primitive actions → high-level plan
│   ├── merge_shards.py
│   ├── validate_generated_data.py
│   ├── evaluation.py / inference.py / cli.py / prompts.py / schema.py
│   ├── generator_config*.json
│   ├── data/                          # diagnostic 240-sample set
│   ├── data_cot/                      # CoT rewrite of the 240-sample set
│   ├── new100k_shard_data/            # 8 shards, 100000 raw rows
│   ├── new100k_clean_data/            # 99489 cleaned rows (training source)
│   ├── new100k_10k_data/              # 10000-row whole-scene subset
│   └── new5000_shard_data/            # 5000-row shard set (no merged dir)
├── training/                          # ms-swift prepare / train / eval
│   ├── cli.py                         # prepare | validate | train | ...
│   ├── cot_data.py / data.py / launcher.py / runtime.py
│   ├── config.cot.10k.server.json
│   ├── config.cot.100k.server.json
│   ├── select_raw_subset.py / clean_raw_shards.py
│   ├── evaluate_section_losses.py
│   ├── evaluate_in_domain_predictions.py
│   ├── compare_counterfactual_predictions.py
│   ├── evaluate_embspatial.py         # EmbSpatial-Bench MCQ (no scores in this checkout)
│   ├── evaluate_relaxed_action_metrics.py
│   ├── generate_cpu_predictions.py
│   └── prepared/cot-10k, cot-100k, ...
├── tests/                             # unittest modules
├── scripts/                           # server bash helpers (100K eval, format-only, val-5k, EmbSpatial)
├── models/Qwen3.5-0.8B-original/      # local copy of the base VLM
├── outputs/                           # training runs and EVA reports
└── passkey.txt                        # gitignored server credential file
```

Additional Chinese design / progress files in the repo root (`Test1-...md`, a PDF) are historical notes. They are not the training contract.

---

## Dataset / Data Generation

### Simulator

| Item | Value | Source |
|---|---|---|
| Engine | AI2-THOR `Controller` | `exp0/generate_data.py` |
| Default house source | ProcTHOR-10k via `prior.load_dataset` | generator configs |
| ProcTHOR revision | `439193522244720b86d8c81cde2e51e3a4d150cf` | `generator_config.json`, `generator_config.procthor_100k.json` |
| Diagnostic heat scenes | iTHOR kitchens (`FloorPlan1/3/4` in the merged 240-set) | `exp0/data/generation_report.json` |
| Platform | `AI2THOR_PLATFORM` or `render.platform`; default `CloudRendering`; also `Linux64` | `generate_data.py` |
| Generator package pin | `ai2thor==5.0.0` | `requirements-generator.txt` |
| iTHOR 4.3 heat env (documented recipe) | `ai2thor==4.3.0` in a separate venv | `exp0/README.md` |

ProcTHOR foods are treated as not `cookable` under AI2-THOR 5.0 in this project, so the 24 diagnostic heat rows were generated in a separate AI2-THOR 4.3 iTHOR environment and merged. The 5K/100K ProcTHOR configs set `"heat": 0`.

### Render / spatial settings (non-smoke configs)

From `exp0/generator_config.json` and `exp0/generator_config.procthor_100k.json`:

| Field | Value |
|---|---|
| Image size | 672 × 672 PNG |
| FOV | 90 |
| `visibility_distance` | 5.0 |
| `grid_size` | 0.25 |
| `rotate_step_degrees` | 45 |
| `camera_horizons` | `[0, 30]` |
| `camera_rotations` | 8 compass headings |
| Spatial axis / close / far | 0.15 m / 1.5 m / 3.0 m |

Smoke generation uses 300×300 (`generator_config.smoke.json`).

### Task groups

Atomic tasks are two-step plans: `GotoLocation(...)` then one manipulation. Gold action names are closed:

`GotoLocation`, `PickupObject`, `PutObject`, `SliceObject`, `CleanObject`, `HeatObject`, `ToggleObject`, `OpenObject`, `CloseObject`.

Simulator mapping that is not 1-1 with the gold name:

- Heat gold action is `HeatObject`; the controller step is `CookObject`.
- Toggle uses `ToggleObjectOn` / `ToggleObjectOff` in the simulator and `ToggleObject` in the gold plan.

`GotoLocation` goes to a visible parent receptacle when one exists, otherwise to the object. `exp0/data_quality.py` records that choice as `goto_schema`.

Object arguments are restricted to the union of `ALFRED_OBJECT_CLASSES` (56 types) and `ALFRED_RECEPTACLE_CLASSES` (26 types) in `generate_data.py`. Overlap is 7 types; the union is **75**, matching `allowed_objects` in `exp0/config.json`. The design note’s “58+26=84” does not match these sets.

### Counterfactual construction

A **counterfactual group** is exactly **two** rows with a shared `meta.counterfactual_group` id (`cf_XXXX`, later namespaced as `shard_{i}_cf_XXXX`). Incomplete pairs are rolled back.

Both pair kinds are stored as `task_group=counterfactual_put`.

**Put pair** (`put_pair`):

- Held fixed: scene, camera pose, target, destination, instruction `把 {target} 放进 {destination}。`
- Changed: destination open vs closed
- Open gold: Goto source → Pickup → Goto dest → Put (length 4)
- Closed gold: same plus `OpenObject` before Put and `CloseObject` after (length 6)
- Images: `*_open.png` / `*_closed.png`

**Open-state pair** (`open_pair`):

- Held fixed: scene, camera, openable receptacle, instruction `确保 {type} 是打开的。`
- Open gold: `GotoLocation` only (length 1)
- Closed gold: `GotoLocation` + `OpenObject` (length 2)

`training/select_raw_subset.py` and the CF evaluators require every retained group to have exactly two members. Pair scoring further requires the two gold action sequences to differ (“gold-discriminative”).

### Generator CLI

```bash
python -m exp0.generate_data
python -m exp0.generate_data --config exp0/generator_config.json --overwrite
AI2THOR_PLATFORM=CloudRendering python -m exp0.generate_data --config exp0/generator_config.json --workers 4
python -m exp0.generate_data --config ... --shard-index 0 --shard-count 8 --overwrite
```

`--workers` cannot be combined with `--shard-index` / `--shard-count`. Without `--overwrite`, an existing `samples.jsonl` is resumed. Target size is `sum(group_quotas) + 2 * counterfactual_pairs`.

### Requested vs on-disk counts

| Dataset | Requested | On disk | Notes |
|---|---:|---:|---|
| Diagnostic `exp0/data/` | 240 | **240** | Merged ProcTHOR + iTHOR heat; 26 scenes; CF ratio 1/3 |
| CoT rewrite `exp0/data_cot/` | 240 | **240** | From the diagnostic set; `max_state_facts=12` |
| `new108` / `new120` shards | 108 | **108** | Config name says 120; quotas sum to 108 |
| `mixed228_data/` | 108 + 120 | **228** | Seed `20260815`; heat count in mix is 12 |
| `new5000_shard_data/` | 5000 | **5000** jsonl across 8 shards | Merged `exp0/new5000_data/` is **not** present |
| `new100k_shard_data/` | 100000 | **100000** jsonl | Shard `generation_report.json` files often still `complete: false` |
| `new100k_clean_data/` | — | **99489** clean / **511** failed | Training source for CoT-100K |
| `new100k_10k_data/` | 10000 | **10000** | Whole-scene subset, seed 42, from an **earlier 69879-row snapshot** |
| ProcTHOR val-5k dirs | 5000 | **empty** in this checkout | Scripts exist; data was not retained here |

Diagnostic group counts (`exp0/data/generation_report.json`):

| Group | Count |
|---|---:|
| `counterfactual_put` | 80 (40 pairs) |
| `pickup` | 32 |
| `clean` | 32 |
| `heat` | 24 |
| `toggle` | 24 |
| `slice` | 24 |
| `open_close` | 24 |

100K requested quotas (`generator_config.procthor_100k.json`, seed **1042**, 10000 train houses):

`2×18520 CF pairs + 14820 pickup + 14820 clean + 0 heat + 11120 toggle + 11100 slice + 11100 open_close = 100000`.

Cleaning (`exp0/new100k_clean_data/cleaning_report.json`):

| Field | Value |
|---|---:|
| `input_samples` | 100000 |
| `clean_samples` | 99489 |
| `failed_samples` | 511 |
| `same_image_and_instruction_map_to_multiple_gold_plans` | 510 |
| `preprocess_no_relevant_state` | 1 |
| `collision_inputs` | 255 |
| `counterfactual_groups` retained | **18265** |
| `samples_sha256` | `f33c8d109aed4cf4c3058b2f2c3abfad7149e770e7969bbb1755402b7b4504bd` |

Rejected attempts recorded from shard reports (not present as rows) include large `insufficient_visible_objects` (91934) and `simulation_failed:counterfactual_put` (14822) counts. Those are generation failures, not the 511 post-clean drops.

10K subset (`exp0/new100k_10k_data/selection_manifest.json`):

| Field | Value |
|---|---:|
| `seed` | 42 |
| `available_samples` | **69879** (incomplete shard snapshot at selection time) |
| `selected_samples` | 10000 |
| `whole_scene_selection` | true |
| `selected_scenes` | 914 |
| `selected_counterfactual_groups` | 1256 |
| Groups in subset | clean 2190, CF put 2512, open_close 1574, pickup 2194, slice 679, toggle 851 (**no heat**) |

The 10K set is therefore **not** a subset of the later cleaned 99489 file. Shard SHA-256 values in the 10K selection manifest do not match those in the 100K cleaning report.

---

## Data Format

### Raw generator row (diagnostic / 100K shards)

Fields written by `save_sample` in `exp0/generate_data.py`:

```json
{
  "sample_id": "exp0_0001",
  "image": "images/exp0_0001.png",
  "wrong_image": "images/exp0_0107.png",
  "instruction": "清洁 Plate。",
  "gold": {
    "plan_actions": ["GotoLocation(Plate)", "CleanObject(Plate)"],
    "plan_nl": "前往 Plate，然后清洁 Plate。"
  },
  "scene_graph": {"objects": [], "relations": []},
  "spatial_facts": ["Plate 位于左侧 Agent。"],
  "subgoals": ["前往 Plate。", "清洁 Plate。"],
  "meta": {
    "scene_id": "procthor_train_03771",
    "counterfactual_group": null,
    "task_group": "clean",
    "target_visible": true,
    "receptacle_state": null,
    "plan_length": 2,
    "sim_verified": true,
    "camera_pose": {},
    "required_object_ids": []
  }
}
```

The first real diagnostic row is a `clean` task on `Plate` in `procthor_train_03771` (`exp0/data/samples.jsonl`). `exp0/data/samples.example.jsonl` is a **hand-written** illustration and is not generator output.

### CoT rewrite (`exp0/generate_cot_data.py`)

Default: `--input exp0/data/samples.jsonl` → `exp0/data_cot/samples_cot.jsonl`. Variants: `cot` (default), `plan-only`, `action-only`. `--max-state-facts` default **12**.

Assistant text for `cot`:

```text
<state>
Pencil is below Agent.
Pencil is far from Agent.
</state>

<plan>
1. Acquire Pencil.
</plan>

<action>
GotoLocation(Pencil)
PickupObject(Pencil)
</action>
```

State comes from `extract_task_relevant_state` (scene graph / attributes / relations, capped at 12 facts). Plan comes from `abstract_subgoals(gold.plan_actions)`, **not** from the raw row’s `subgoals`. That abstraction drops `GotoLocation`, and also drops `OpenObject`/`CloseObject` when another manipulation is present. The rendered `<plan>` is numbered; the internal plan list is not.

Token-length estimates on the 240 CoT file (`exp0/data_cot/generation_report.json`, deterministic unicode estimator): average state 25.250, plan 6.213, action 8.800 tokens.

### Prepared ms-swift JSONL (`training/cot_data.py`)

User turn: `"<image>\n目标指令：{instruction}"`. Images are **absolute paths** on the machine that ran `prepare`.

When `section_loss_weights` is set, each section is a separate assistant message with a scalar `loss_scale`. Example from `training/prepared/cot-100k/train.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "你是一个室内家务动作规划助手。..."},
    {"role": "user", "content": "<image>\n目标指令：拿起Pencil。"},
    {"role": "assistant", "content": "<state>\nPencil is below Agent.\nPencil is far from Agent.\n</state>\n\n", "loss_scale": 0.3},
    {"role": "assistant", "content": "<plan>\n1. Acquire Pencil.\n</plan>\n\n", "loss_scale": 0.3},
    {"role": "assistant", "content": "<action>\nGotoLocation(Pencil)\nPickupObject(Pencil)\n</action>", "loss_scale": 0.4}
  ],
  "images": ["/root/qwen35-08b-spatial-action-ft/exp0/new100k_shard_data/shard_0/images/exp0_0001.png"]
}
```

`new100k_clean_data/` stores JSONL only; PNG files remain under `new100k_shard_data/shard_*/images/`.

Legacy non-CoT `training/data.py` (no `response_format`) still targets:

```text
<plan> …primitive actions… </plan>
<summary> …gold.plan_nl… </summary>
```

That is the Exp 0 diagnostic contract, not the 10K/100K training contract.

---

## Model

| Item | Value | Source |
|---|---|---|
| Name | Qwen3.5-0.8B (post-trained VLM) | `models/Qwen3.5-0.8B-original/README.md` |
| Architecture | `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5` | model `config.json` |
| Parameters | 0.8B | model card |
| Hidden size / layers / FFN | 1024 / 24 / 3584 | model card |
| Vocab | 248320 | model card |
| Native context | 262144 | model card |
| Vision | ViT 12 layers, hidden 768, patch 16, merge 2, out 1024 | design note §2.1; not re-derived here from every config field |
| Hybrid attention | Gated DeltaNet + sparse full attention | model card |
| Server training path | `/model/ModelScope/Qwen/Qwen3.5-0.8B` | `training/config.server.json`, run `args.json` |
| Local weight copy | `models/Qwen3.5-0.8B-original/` | used as step-0 base in EVA |
| Model license file | Apache-2.0 | `models/Qwen3.5-0.8B-original/LICENSE` |

`training/runtime.py` requires `config.json` architectures to contain `"Qwen3_5"`, Python ≥ 3.12, CUDA, and bf16.

---

## Training

Entry point:

```bash
python -m training.cli --config training/config.cot.100k.server.json prepare --overwrite
python -m training.cli --config training/config.cot.100k.server.json validate
python -m training.cli --config training/config.cot.100k.server.json check-runtime
python -m training.cli --config training/config.cot.100k.server.json show-command
python -m training.cli --config training/config.cot.100k.server.json train
```

If `data.response_format` is set, `prepare`/`validate` use `training/cot_data.py`; otherwise `training/data.py`. `train` always validates, then launches `swift sft`.

### Split

Union-find over `split_group_fields` (default `scene_id`, `counterfactual_group`). Components are shuffled with `data.seed` (42) and packed until validation is nearest to `round(N * 0.1)`, with at least one val example and not all components in val.

Prepared manifests:

| Prepared dir | Source | Total | Train | Val |
|---|---|---:|---:|---:|
| `training/prepared/cot-10k` | `new100k_10k_data` | 10000 | 9000 | 1000 |
| `training/prepared/cot-100k` | `new100k_clean_data` | 99489 | 89538 | 9951 |
| `training/prepared/cot-smoke500` | `samples.smoke500.jsonl` | 500 | 452 | 48 |

100K val contains **1860** complete gold-discriminative CF pairs (3720 rows, 600 scenes) per `outputs/.../counterfactual-comparison/REPORT.md`. 10K val contains **106** such pairs per the 10K `DETAILED_EVA.md`. Audits report **0** train/val scene overlap and **0** CF-group overlap.

### Response-format ablations (configured)

| Arm | Config | Assistant target | Section weights |
|---|---|---|---|
| Exp-A action-only | `config.action.{10k,100k}.server.json` | `<action>` | `{action: 1.0}` |
| Exp-B plan+action | `config.plan.{10k,100k}.server.json` | `<plan>` + `<action>` | `{plan: 3/7, action: 4/7}` ≈ 0.42857 / 0.57143 |
| Exp-C full CoT | `config.cot.{10k,100k}.server.json` | `<state>` + `<plan>` + `<action>` | `{state: 0.3, plan: 0.3, action: 0.4}` |
| Format-only | `config.cot.format-only.{10k,100k}.server.json` | same as C, train labels permuted | same as C |

A/B configs `extends` the corresponding C config, so batch, epoch, LR, split, and model are inherited. The older unsuffixed `config.plan.server.json` uses **0.4 / 0.6**, not 3:4. That file is the 240-sample / `max_steps=300` path, not the 10K/100K comparison.

**This checkout has training outputs only for Exp-C 10K, Exp-C 100K, and a 500-row smoke run.** There are no `outputs/qwen35-08b-plan-*` or `outputs/qwen35-08b-action-*` directories. Format-only 100K has an incomplete run directory (see below).

### Fine-tuning method (completed CoT runs)

| Item | 10K CoT | 100K CoT |
|---|---|---|
| Tuner | `full` | `full` |
| Freeze LLM / ViT / aligner | false / false / false | same |
| LoRA | not used | not used |
| `torch_dtype` | bfloat16 | bfloat16 |
| `attn_impl` | `flash_attention_2` | `flash_attention_2` |
| `use_liger_kernel` | false (required with section weights) | false |
| `max_length` | 2048 | 2048 |
| `add_non_thinking_prefix` | true | true |
| `loss_scale` | `ignore_empty_think` | `ignore_empty_think` |
| `is_binary_loss_scale` | false | false |
| Learning rate | `1e-5` | `1e-5` |
| `warmup_ratio` | 0.05 | 0.05 |
| Per-device train batch | 2 | 8 |
| Grad accum | 2 | 1 |
| Effective batch (1 GPU) | 4 | 8 |
| `num_train_epochs` | 1.0 | 1.0 |
| `max_steps` | unset (epoch-controlled) | unset |
| Actual optimizer steps | **2250** | **11193** |
| Eval / save every | 250 | 1000 |
| `eval_on_start` | true | true |
| `save_total_limit` | 10 | 10 |
| Grad checkpointing | true | false |
| Seed / data_seed | 42 / 42 | 42 / 42 |
| `CUDA_VISIBLE_DEVICES` | `0` | `0` |

Optimizer / scheduler are **not** set in the JSON configs. The launcher forwards `optim` only if present. The completed runs’ `args.json` records:

| Item | Recorded value |
|---|---|
| `optim` | `adamw_torch_fused` |
| `lr_scheduler_type` | `cosine` |
| `weight_decay` | 0.1 |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.95 |
| `max_grad_norm` | 1.0 |
| `warmup_steps` | 0 (ratio 0.05 still set) |

Those are ms-swift/Transformers defaults as logged by the trainer, not values authored in `training/config.*.json`.

LoRA alternative configs (`config.lora.server.json`, `config.lora.cot.server.json`): rank **8**, alpha **32**, `target_modules: ["all-linear"]`, LR `1e-4`, micro-batch 4, accum 1, `max_steps` 300. They also leave llm/vit/aligner unfrozen. **No LoRA run directory is present under `outputs/`.** The launcher rejects a freeze policy that freezes all three of llm, vit, and aligner. QLoRA is not implemented.

### Shell runners (server)

| Script | Action |
|---|---|
| `training/setup_server.sh` | Create `.venv-train` from Python 3.12, install training deps, `check-runtime` |
| `training/run_cot10k_server.sh` | validate → check-runtime → show-command → train (`config.cot.10k.server.json`) |
| `training/run_cot100k_server.sh` | same for 100K; **refuses** if `outputs/qwen35-08b-cot-100k` is already non-empty |
| `scripts/retry_cot100k_format_only.sh` | train `config.cot.format-only.100k.server.json` |

---

## Evaluation

There are two evaluators. They are **not** interchangeable.

### 1. Exp 0 diagnostic (`exp0/evaluation.py`)

Zero-shot / prompt-condition study on the 240 (or mixed 228) rows. Primary metric: `action_seq_em`.

**Conditions** (`exp0/prompts.py`):

| ID | Image | User content |
|---|---|---|
| D | correct | instruction + gold `subgoals` |
| A′ | `wrong_image` | instruction only |
| A | correct | instruction only |
| B_natural / B_json / B_triples | none | instruction + serialized scene graph |
| C | correct | instruction + `spatial_facts` |

Each sample yields 7 inferences. Generation in `exp0/config.json`: `do_sample=false`, `max_new_tokens=512`, seed 42. Server config uses `attn_implementation: sdpa`.

**Action metrics (lenient):**

- `action_seq_em`: lenient-parsed action list **exactly equals** gold. Parser prefers `<action>`, then `<plan>`. Accepts missing parentheses, case drift, list markers; drops names outside the allowed set.
- `action_step_match`: positional matches / `max(len(pred), len(gold), 1)` (length mismatch penalizes the denominator).
- `action_parsed`: whether any lenient action was recovered.

**Strict format metrics** (`parse_plan`): tags present, every non-empty line parseable, non-empty action list. Reported as compliance, not as the primary capability score.

**NL metrics** (legacy `<summary>` protocol): `nl_plan_match` requires ordered action-keyword hits and full object-name recall using `schema.ACTION_NL_KEYWORDS`.

Diagnosis thresholds (`exp0/config.json`): `d_min_score=0.6`, `near_gap=0.03`, `large_gap=0.1`, `floor_score=0.15`. If D is below `d_min_score`, `diagnose()` stops and does not interpret A/B/C.

`exp0/evaluation.py` comments that 81.7% of **that diagnostic set** is solvable from instruction text alone; this README does not recompute that figure.

CLI:

```bash
python -m exp0.cli --config exp0/config.server.json validate
python -m exp0.cli --config exp0/config.server.json infer
python -m exp0.cli --config exp0/config.server.json evaluate
```

`infer` resumes `outputs/predictions.jsonl` unless `--overwrite`. Fine-tuned CoT models are intended to be scored with `exp0/config.cot.server.json` (`response_format: action`).

### 2. In-domain CoT evaluation (10K / 100K)

**Free generation** (`training/generate_cpu_predictions.py`): greedy (`do_sample=False`), default `max_new_tokens=256`. `--image-mode a-prime` swaps images: CF pairs use the **other member’s** image; non-CF rows use persisted `wrong_image` when present, else a hash-stable other-scene image.

**Action exact** (`training/evaluate_in_domain_predictions.py`) uses **strict** `parse_plan` (not the Exp 0 lenient reader). `action_sequence_exact` is predicted tuple == gold tuple.

**Step position match recall / precision:** count of positional equalities divided by total gold steps (recall) or predicted steps (precision), summed over the slice.

**State facts:** lines inside `<state>`, whitespace-normalized, casefolded, trailing `.`/`。` stripped; precision/recall/F1 over that set. `state_exact` is set equality.

**CF pair exact:** both members of a complete pair have `action_exact`. Same-action rate is both members emitting the identical predicted sequence.

**Structure:** `STRICT_COT` regex (state/plan/action tags in order) **and** `parsed.structure_valid`.

**Text oracle:** train-fitted map from normalized instruction → majority gold action sequence, scored on val.

**A − A′:** mean per-example difference of action exact, with scene-cluster percentile bootstrap (1000 resamples) when reported.

**Teacher-forced section losses** (`training/evaluate_section_losses.py`): CE on state/plan/action. `full_*` includes XML/newlines/EOS; `body_*` is inside the tags; `format_*` is full minus body. The script’s weighted diagnostic is **not** the Swift trainer `eval_loss` (different reduction). `--checkpoint-steps 0` requires `--base-model-dir`. Default `--dtype` is `bfloat16`; default `--device` is `cpu`.

### 3. Additional evaluators (code present; no scores in this checkout)

`training/evaluate_embspatial.py` answers EmbSpatial-Bench multiple-choice items with a single letter A–D (`PROMPT_VERSION = embspatial-generation-v1`). `scripts/run_embspatial_pre_post.sh` downloads `Phineas476/EmbSpatial-Bench` to `data/embspatial-bench/embspatial_bench.json`, **requires exactly 3640 rows**, and compares the base model to `checkpoint-11193`. That dataset directory and `ood-embspatial-bench/` outputs are **not** in this checkout.

`training/evaluate_relaxed_action_metrics.py` re-scores before/after prediction files with a looser action parse (lenient names/arity, Chinese NL keywords, rejection of conditional/negated plans). No relaxed-metric report is present under `outputs/`.

---

## Experiments

### Exp 0 — zero-shot diagnostic (240)

Purpose: bottleneck diagnosis before SFT. Model: `Qwen/Qwen3.5-0.8B` (local config) or `/model/ModelScope/Qwen/Qwen3.5-0.8B` (server). Dataset: `exp0/data/samples.jsonl`, 200–300 sample gate (actual 240). Outputs: `exp0/outputs/`.

Observed `action_seq_em` is **0%** on every condition (`exp0/outputs/report.md`). Structure is low except D (58.75%). The report flags a floor effect. This run used the **legacy** `<plan>`+`<summary>` contract.

### Exp 0 mixed228, new prompt (228)

`exp0/outputs_mixed228_new_prompt/`. Dataset `mixed228_data` (108 new + 120 of the old 240). Primary `action_seq_em`:

| Condition | action_seq_em | step match | strict structure |
|---|---:|---:|---:|
| D | 83.77% | 90.23% | 100% |
| A | 0.88% | 2.56% | 99.12% |
| A′ | 0.00% | 0.51% | 98.68% |
| B_json | 2.19% | 18.86% | 99.56% |
| B_natural | 1.75% | 19.53% | 98.25% |
| B_triples | 0.88% | 18.48% | 99.56% |
| C | 1.75% | 14.56% | 98.25% |

A − A′ = 0.88%. Oracle condition D is high; A/B/C remain near floor under that prompt.

### Smoke-500

Config `training/config.cot.smoke500.server.json`. Source: 500-row smoke JSONL. Train 452 / val 48. `max_steps=3`. Output: `outputs/qwen35-08b-cot-smoke500/v0-20260815-130829/` (`checkpoint-3` only). Eval loss at step 3: 0.24714084; token acc 0.824295. This is a launcher smoke test, not a result table.

### CoT-10K — `outputs/qwen35-08b-cot-10k/v0-20260816-022322`

Full CoT, 1 epoch, 2250 steps, val 1000. Checkpoints: 250, 500, …, 2250.

Trainer full-val (from `checkpoint-2250/trainer_state.json`):

| step | eval_loss | eval_token_acc |
|---:|---:|---:|
| 0 | 0.438532263 | 0.737151 |
| 250 | 0.023159448 | 0.972723 |
| 2250 | 0.008684068 | 0.987711 |

Split audit (10K `DETAILED_EVA.md`): 0 DSU leakage; 822 train scenes / 92 val scenes; 106 complete CF pairs in val; train-fitted text oracle **72.60%** (coverage 99.30%).

Free-generation CF coverage in that report is a **1/106** CPU pair. That is not a formal 10K CF score.

### CoT-100K — `outputs/qwen35-08b-cot-100k/v0-20260816-102046`

Full CoT, 1 epoch, 11193 steps, train 89538 / val 9951. Wall time **9098.61 s**; peak VRAM **23.03 GiB** (`DETAILED_EVA.md`). Best logged val loss: **0.005916716** at step 11000; final 11193 is 0.005917093. `save_total_limit=10` removed checkpoints 1000 and 2000. Remaining: 3000 … 11000 and 11193.

Hardware assumed by `training/setup_server.sh`: RTX 4090 (`TORCH_CUDA_ARCH_LIST=8.9`), CUDA 13.2, Python 3.12 at `/usr/local/miniconda3/envs/py312/bin/python`. Config `minimum_gpu_memory_gib=20`.

---

## Results

Interpretations from reports are listed after the tables. Do not mix n=8, n=200, and full-val numbers.

### Trainer full validation (teacher forcing)

| Run | n | step0 loss | final loss | step0 token acc | final token acc |
|---|---:|---:|---:|---:|---:|
| CoT-10K | 1000 | 0.438532 | 0.008684 | 73.72% | 98.77% |
| CoT-100K | 9951 | 0.428906 | 0.005917 | 73.97% | 99.19% |

100K curve (`checkpoint-11193/trainer_state.json`):

| step | eval_loss | eval_token_acc |
|---:|---:|---:|
| 0 | 0.428906053 | 0.739738 |
| 1000 | 0.018265830 | 0.977421 |
| 5000 | 0.007204033 | 0.990169 |
| 11000 | 0.005916716 | 0.991907 |
| 11193 | 0.005917093 | 0.991883 |

10K vs 100K losses are **non-paired** (different val splits).

### CoT-100K free generation — full val (correct image A vs A′)

Source: `outputs/qwen35-08b-cot-100k/v0-20260816-102046/in-domain-full-val/`. n=9951, 752 scenes, step **11193**, scene-cluster bootstrap 1000.

| Metric | A (correct image) | A′ (swapped image) |
|---|---:|---:|
| Action sequence exact | 91.59% (CI 90.74–92.41) | 37.55% (36.11–39.07) |
| CF pair exact | 79.14% (76.61–81.42) | 0.22% |
| Step position match recall | 96.65% | 71.72% |
| State fact P / R / F1 | 87.60 / 88.21 / 87.90% | F1 53.33% in `metrics.json` |
| State exact | 48.23% | 4.59% |
| P(action exact \| state exact) | 100% | 100% |
| P(action exact \| state wrong) | 83.75% | 34.55% |
| Strict structure | 100% | 100% |
| Placeholder copy | 0% | 0% |
| Train-fitted text oracle (A slice) | 62.54% (cov 99.96%) | same oracle |

A − A′ action exact = **+54.03%** (CI +52.43 to +55.53).

Selected A slices (`in-domain-full-val/REPORT.md`):

| Slice | n | Action exact | State F1 | Text oracle |
|---|---:|---:|---:|---:|
| cf:no | 6231 | 94.32% | 89.07% | 74.42% |
| cf:yes | 3720 | 87.02% | 85.77% | 42.63% |
| plan_length:2 | 7476 | 94.54% | 89.59% | 62.03% |
| plan_length:4 | 615 | 74.15% | 81.07% | 55.45% |
| task_group:pickup | 1481 | 99.86% | 92.30% | 99.73% |
| Microwave | 68 | 73.53% | 88.46% | 48.53% |

### CoT-100K counterfactual pairs — 1860 / 3720 (formal CF report)

Source: `counterfactual-comparison/REPORT.md` (binomial CIs in that file). Gold actions differ in all 1860 pairs.

| Model | Sample exact | Pair exact | Same-action rate | Structure |
|---|---:|---:|---:|---:|
| step0 | 0.00% | 0.00% | 100.00% | 0.00% |
| step11193 | **87.10%** (3240/3720) | **79.25%** (1474/1860) | **14.84%** (276/1860) | 100% |

Pair outcomes at 11193: both correct 1474, one correct 292, both wrong 94.

State slice sample exact at 11193: closed 91.08%, open 83.12%.

Error types at 11193: `missing_open_for_closed` 76, `redundant_open_for_open` 215, `navigation_or_target_error` 188, `other_action_error` 1.

**Do not average with in-domain CF pair exact 79.14%.** That number is the CF slice of the in-domain scorer (`0.870161` sample exact on 3720; val audit `pair_exact=0.791398`, one/zero members 293/95). The dedicated CF report / `eva-audit-complete-cf-step11193` uses 1474/1860 = 0.792473 and 292/94. Same run, slightly different aggregation paths.

Cluster-bootstrap CIs (`counterfactual-comparison-cluster-bootstrap/REPORT.md`): sample exact 85.51–88.55; pair exact 76.73–81.45 (point estimates unchanged).

### CoT-100K n=200 seed 42 (medium-strength; do not overwrite full val)

Teacher-forced (`section-loss-eval-gpu-bf16-n200-seed42/`, RTX 4090 bf16):

| Metric | step0 | step11193 |
|---|---:|---:|
| State body token acc | 66.44% | 98.55% |
| Plan body token acc | 67.93% | 100% |
| Action body token acc | 82.95% | 100% |
| Plan/action body EM | 0 | 1.0 |
| State body EM | 0 | 0.56 |

Free-gen random 200 (`pre-post-comparison-random200-seed42/`): action exact 0 → 92.50%; state F1 0 → 88.62%; state exact 0 → 51.00%; structure 0 → 100%; text oracle 62.00% both sides. That subset contains only **1** complete CF pair.

CF 100 pairs seed 42 (`counterfactual-comparison-cf100-seed42-n200/`): after sample exact 87.00% (CI 81.50–92.50); pair exact 81.00% (73.73–88.42); same-action 10.00%. A **different** cf100-n200 draw (53 scenes, no seed42 in the folder name) reports after sample exact 93.00% / pair exact 88.00%. Those two 200-row CF tables are not the same sample.

### Section-loss pilots (not formal)

Fixed 8 val rows, seed 42. 100K GPU bf16: plan/action body EM 8/8 from step 3000; state format loss **rises** vs step0 (0.405031 → 2.818685 at 3000). 10K CPU fp32: similar pattern. These support the report claim that early loss drop is not “mostly XML format,” but they are n=8.

Step0 full-val section CE (9951, `section-loss-eval-gpu-bf16-step0-full/`): state/plan/action body losses 1.452780 / 2.000356 / 0.876851; body EM 0/9951 on all three sections.

---

## Installation

### Generator (AI2-THOR 5.0)

```bash
pip install -r requirements-generator.txt
```

Pins: `ai2thor==5.0.0`, `prior`, `numpy<2`, `Pillow`. A Linux server without Vulkan typically uses `AI2THOR_PLATFORM=Linux64 xvfb-run -a`.

iTHOR heat (diagnostic 24 rows) is documented as a **separate** venv with `ai2thor==4.3.0`.

### Training (intended server)

`training/setup_server.sh` expects:

- Python 3.12 at `/usr/local/miniconda3/envs/py312/bin/python`
- CUDA 13.2 at `/usr/local/cuda-13.2`
- GPU with bf16 and ≥ 20 GiB (config); the documented target is RTX 4090 24GB
- Repo at `/root/qwen35-08b-spatial-action-ft`

```bash
cd /root/qwen35-08b-spatial-action-ft
bash training/setup_server.sh
```

`requirements-training.txt`: `ms-swift==4.4.2`, `transformers>=5.9,<6`, `qwen_vl_utils>=0.0.14`, `decord`, `peft`, `liger-kernel`, `tensorboard`. The setup script also installs `flash-linear-attention>=0.4.2`, `causal-conv1d` (commit `3a4c88e599cd7dec333cac727bd59f2a41a8aad5`, patched to `sm_89` only), and `flash-attn==2.8.3`. PyTorch is taken from the system site packages; this repo does not pin a torch version.

`training/runtime.py` requires **exactly** ms-swift 4.4.2.

### Exp 0 inference

`requirements-exp0.txt` (local) includes `flash-attn==2.8.3`. `requirements-exp0-server.txt` omits flash-attn (server Exp 0 config uses sdpa).

---

## Usage

### 1. Generate diagnostic data (240)

See `exp0/README.md` for the multi-shard ProcTHOR + iTHOR heat recipe that produced `exp0/data/`. Minimal single-process:

```bash
python -m exp0.generate_data --config exp0/generator_config.json
python -m exp0.validate_generated_data --dataset-dir exp0/data
python -m exp0.cli --config exp0/config.server.json validate
```

### 2. Generate / inspect CoT labels (240)

```bash
python -m exp0.generate_cot_data --overwrite
python -m exp0.generate_cot_data --validate-only
```

Large-scale 10K/100K training **does not** require this file. Those runs set `"source_format": "raw_simulator"` and rebuild state/plan from `gold.plan_actions` inside `training/cot_data.py`.

### 3. Clean 100K shards and prepare training JSONL

```bash
.venv-train/bin/python -m training.clean_raw_shards \
  --shard-root exp0/new100k_shard_data \
  --output-dir exp0/new100k_clean_data \
  --shards 8 --overwrite

.venv-train/bin/python -m training.cli \
  --config training/config.cot.100k.server.json prepare --overwrite
.venv-train/bin/python -m training.cli \
  --config training/config.cot.100k.server.json validate
```

10K subset from shards (historical command; it will not reproduce the committed 10K file from the **current** 100000-row shards, because that subset was drawn from 69879 rows):

```bash
.venv-train/bin/python -m training.select_raw_subset \
  --shard-root exp0/new100k_shard_data \
  --shards 8 \
  --output-dir exp0/new100k_10k_data \
  --count 10000 --seed 42 --overwrite
```

### 4. Train

```bash
.venv-train/bin/python -m training.cli \
  --config training/config.cot.100k.server.json train
```

Or `bash training/run_cot100k_server.sh` on a **fresh** output directory.

To resume, set `training.resume_from_checkpoint` in the JSON to an existing `checkpoint-*` path. `prepare` does not overwrite unless `--overwrite`.

### 5. Evaluate a checkpoint

Section losses:

```bash
python -m training.evaluate_section_losses \
  --run-dir outputs/qwen35-08b-cot-100k/v0-20260816-102046 \
  --base-model-dir models/Qwen3.5-0.8B-original \
  --val-file training/prepared/cot-100k/val.jsonl \
  --checkpoint-steps 0 11193 \
  --output-dir outputs/qwen35-08b-cot-100k/v0-20260816-102046/section-loss-eval \
  --device cuda --dtype bfloat16
```

Free-generation + in-domain metrics (arguments are those the parsers actually expose):

```bash
python -m training.generate_cpu_predictions \
  --checkpoint outputs/qwen35-08b-cot-100k/v0-20260816-102046/checkpoint-11193 \
  --val-file training/prepared/cot-100k/val.jsonl \
  --raw-data exp0/new100k_clean_data/samples.jsonl \
  --manifest training/prepared/cot-100k/manifest.json \
  --output /tmp/preds.jsonl \
  --selection all --image-mode correct \
  --max-new-tokens 256 --device cuda --dtype bfloat16 --overwrite

python -m training.evaluate_in_domain_predictions \
  --raw-data exp0/new100k_clean_data/samples.jsonl \
  --val-file training/prepared/cot-100k/val.jsonl \
  --manifest training/prepared/cot-100k/manifest.json \
  --predictions /tmp/preds.jsonl \
  --output-dir /tmp/in-domain
```

Optional `--a-prime-predictions` adds the A′ column.

CF comparison:

```bash
python -m training.compare_counterfactual_predictions \
  --raw-data exp0/new100k_clean_data/samples.jsonl \
  --manifest training/prepared/cot-100k/manifest.json \
  --baseline-predictions preds-step0.jsonl \
  --candidate-predictions preds-step11193.jsonl \
  --output-dir /tmp/cf-compare
```

### 6. Tests

```bash
python -m unittest discover -s tests
```

Modules cover generation, Exp 0 scoring, CoT prepare/validate, section-loss eval, CF/pre-post comparison, shard cleaning, and EVA audits.

---

## Reproducibility

**Present**

- Config JSON with `extends`, seeds (generation 42 / 1042 / 2042; split 42; 10K subset 42; n=200 eval 42), and SHA-256 of the 100K cleaned source and prepared manifests.
- Trainer `args.json` + `trainer_state.json` for the completed 10K and 100K runs.
- Evaluation reports and metrics JSON under `outputs/`.
- Unittest coverage of core transforms.

**Missing or incomplete**

- No project-level `LICENSE`.
- No pinned PyTorch / CUDA wheel in a lockfile; server setup reuses site-packages CUDA 13.2.
- `full_determinism` is false in the 100K `args.json`.
- 10K subset was drawn from a 69879-row incomplete snapshot; re-running `select_raw_subset` on the current 8×100000 shards will not recreate `new100k_10k_data`.
- ProcTHOR held-out val-5k data dirs (`exp0/procthor_val_5k_*`) are empty shells in this checkout. `scripts/retry_procthor_val_5k_eval.sh` would write `outputs/.../in-domain-heldout-procthor-val5k/`; that output directory is not present here.
- EmbSpatial-Bench (`data/embspatial-bench/`) and `ood-embspatial-bench/` scores are not present.
- `exp0/new5000_data/samples.jsonl` is absent, so `config.cot.pilot5000.server.json` cannot `prepare` as written (shards exist under `new5000_shard_data/`).
- Format-only 100K: config and retry script exist; `outputs/qwen35-08b-cot-100k-format-only/v1-20260817-105816/` has TensorBoard `runs/` but **no** `checkpoint-11193` in this tree. `DETAILED_EVA.md` treats format-only scores as missing.
- Exp-A / Exp-B 10K and 100K were not trained in this checkout.
- Large prediction JSONL for the full 3720 CF generations is documented as remaining on the server (`DETAILED_EVA.md`).
- Image paths inside prepared JSONL are absolute `/root/...` paths.

---

## Known Issues / Limitations

1. **Implemented CoT tags ≠ design-doc tags.** Code uses `<state>` / `<plan>` / `<action>`. The design note uses `<state>` / `<subgoal>` / `<plan>`.
2. **Design note vs runs:** the outline’s default tuner is LoRA; completed 10K/100K jobs are full FT.
3. **Object vocabulary:** design note 58+26=84; `generate_data.py` union is 75. `exp0/README.md` already records this.
4. **Text shortcut is large.** 100K train-fitted text oracle is 62.54%; 10K is 72.60%. Token accuracy in the 98–99% range is not visual success.
5. **CF pair exact < sample exact.** 79.25% vs 87.10% on 1860 pairs. Residual same-action rate 14.84% means some open/closed pairs still get one shared action sequence.
6. **Two CF pair-exact numbers** (79.25% vs 79.14%) from dedicated CF vs in-domain scorers. Do not blend them.
7. **Format-only data on disk is not permuted.** `training/prepared/cot-100k-format-only/train.jsonl` has the same size as `prepared/cot-100k/train.jsonl` (107,592,404 bytes) and the same first 8 MiB SHA-256. Its manifest has no `training_label_mode`. Current `validate` would treat missing mode as `aligned` and reject the format-only config until `prepare --overwrite`. Causal “format vs content” attribution is therefore **not** available from this checkout.
8. **A/B response-format ablations** are configured and unit-tested for inheritance, but not trained here.
9. **No heat tasks** in 10K/100K ProcTHOR data (`heat: 0`). Only the 240 diagnostic set includes 24 iTHOR heat rows.
10. **10K is not a subsample of cleaned 100K.**
11. **Exp 0 lenient vs in-domain strict parsers** differ. Legacy Exp 0 also scored `<plan>` as actions; CoT scoring prefers `<action>`.
12. **`config.plan.server.json` weights (0.4/0.6)** disagree with 10K/100K plan weights (3/7, 4/7).
13. **Trainer `eval_loss` ≠ section-loss diagnostic.** Documented in `DETAILED_EVA.md`.
14. **State format loss can increase** while body loss falls (n=8 and n=200 section evals).
15. **`run_cot100k_server.sh` will not relaunch** on top of the existing output directory.
16. **`passkey.txt`** is gitignored and used by `.codex_incremental_sync.ps1` for SSH. Do not commit it.
17. **Shard generation reports lag JSONL** (`complete: false` while 100000 rows exist). Cleaning used the JSONL counts.
18. **`samples.example.jsonl`** uses placeholder objects (`WaterBottle`) that are not in the generator’s ALFRED type sets.
19. **EmbSpatial-Bench / relaxed-action evaluators** are implemented but have no result files in this checkout.

---

## Future Work

Only items the repo itself still marks as open:

- Train and evaluate the **format-only** arm on the same 9951 val / 1860 CF pairs (`DETAILED_EVA.md`, `scripts/retry_cot100k_format_only.sh`).
- Generate and evaluate the **ProcTHOR val-5k** held-out set (`scripts/retry_procthor_val_5k_*.sh`; local dirs empty).
- Run `scripts/run_embspatial_pre_post.sh` if an OOD spatial MCQ number is required; the script exists, the 3640-row dataset and scores do not.
- Run Exp-A / Exp-B at 10K/100K if the three-way ablation is still required (`training/README.md`).
- Merge `new5000_shard_data` into `new5000_data` if the pilot5000 config is to be used.

This README does not add an independent roadmap.

---

## License

There is **no repository-level LICENSE**. The bundled Qwen3.5-0.8B files are Apache-2.0 (`models/Qwen3.5-0.8B-original/LICENSE`). Dataset and code licensing for this project are unspecified in-tree.

---

## Acknowledgements / References

Software and assets actually used:

- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (local copy under `models/`)
- [ms-swift 4.4.2](https://github.com/modelscope/ms-swift)
- [AI2-THOR](https://github.com/allenai/ai2thor) 5.0.0 (generator) and 4.3.0 (documented iTHOR heat path)
- [ProcTHOR-10k](https://github.com/allenai/procthor) via `prior`, revision `439193522244720b86d8c81cde2e51e3a4d150cf`
- ALFRED **class names and task templates only** — ALFRED trajectories are not used as training data
- EmbSpatial-Bench (`Phineas476/EmbSpatial-Bench`) is referenced by `scripts/run_embspatial_pre_post.sh` (expected 3640 items); the JSON is not vendored in this checkout

Operational details for Exp 0 generation and ms-swift flags: `exp0/README.md`, `training/README.md`. Formal 100K writeup: `outputs/qwen35-08b-cot-100k/v0-20260816-102046/DETAILED_EVA.md`.
