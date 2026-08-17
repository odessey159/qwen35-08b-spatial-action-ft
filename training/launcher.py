from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .common import TrainingConfigError, require_mapping, resolve_path
from .cot_data import uses_embodied_format, validate_cot_prepared_dataset
from .data import validate_prepared_dataset


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _option(command: list[str], name: str, value: Any) -> None:
    command.extend((f"--{name}", str(value)))


def _resolved_freeze_policy(config: dict[str, Any]) -> dict[str, bool]:
    policy = require_mapping(config.get("freeze_policy"), "freeze_policy")
    result: dict[str, bool] = {}
    unresolved: list[str] = []
    for name in ("llm", "vit", "aligner"):
        value = policy.get(name)
        if value is None:
            unresolved.append(name)
        elif isinstance(value, bool):
            result[name] = value
        else:
            raise TrainingConfigError(f"freeze_policy.{name} must be true, false, or null")
    if unresolved:
        raise TrainingConfigError(
            "Freeze policy is intentionally unresolved. Set true/false for: "
            + ", ".join(f"freeze_policy.{name}" for name in unresolved)
        )
    if all(result.values()):
        raise TrainingConfigError("Freeze policy freezes LLM, ViT, and aligner; nothing would train")
    return result


def build_environment(config: dict[str, Any]) -> dict[str, str]:
    server = require_mapping(config.get("server"), "server")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(server.get("cuda_visible_devices", "0"))
    environment["IMAGE_MAX_TOKEN_NUM"] = str(server.get("image_max_token_num", 1024))
    environment["PYTORCH_CUDA_ALLOC_CONF"] = str(
        server.get("pytorch_cuda_alloc_conf", "expandable_segments:True")
    )
    return environment


def build_swift_command(config: dict[str, Any], base_dir: Path) -> list[str]:
    freeze = _resolved_freeze_policy(config)
    data_config = require_mapping(config.get("data"), "data")
    server = require_mapping(config.get("server"), "server")
    model = require_mapping(config.get("model"), "model")
    training = require_mapping(config.get("training"), "training")
    weighted_sections = training.get("section_loss_weights") is not None
    if weighted_sections and training.get("use_liger_kernel", True):
        raise TrainingConfigError(
            "Non-binary section_loss_weights require training.use_liger_kernel=false"
        )
    tuner_type = str(training.get("tuner_type", "full"))
    if tuner_type not in {"full", "lora"}:
        raise TrainingConfigError("training.tuner_type must be 'full' or 'lora'")

    prepared_dir = resolve_path(base_dir, str(data_config.get("prepared_dir", "prepared")))
    train_path = prepared_dir / "train.jsonl"
    val_path = prepared_dir / "val.jsonl"
    swift_executable = str(server.get("swift_executable", "swift"))
    if Path(swift_executable).is_absolute() and not Path(swift_executable).is_file():
        raise FileNotFoundError(f"swift executable does not exist: {swift_executable}")
    if not Path(swift_executable).is_absolute() and shutil.which(swift_executable) is None:
        raise FileNotFoundError(f"swift executable is not on PATH: {swift_executable}")

    command = [swift_executable, "sft"]
    _option(command, "model", model["path"])
    _option(command, "tuner_type", tuner_type)
    _option(command, "dataset", train_path)
    if val_path.is_file() and val_path.stat().st_size:
        _option(command, "val_dataset", val_path)
    _option(command, "load_from_cache_file", _bool_text(training.get("load_from_cache_file", True)))
    _option(command, "add_non_thinking_prefix", _bool_text(training.get("add_non_thinking_prefix", True)))
    _option(command, "loss_scale", training.get("loss_scale", "ignore_empty_think"))
    if weighted_sections:
        _option(command, "is_binary_loss_scale", "false")
    _option(command, "torch_dtype", training.get("torch_dtype", "bfloat16"))
    max_steps = training.get("max_steps", 300)
    num_train_epochs = training.get("num_train_epochs")
    if max_steps is None and num_train_epochs is None:
        raise TrainingConfigError("Set training.max_steps or training.num_train_epochs")
    if max_steps is not None and num_train_epochs is not None:
        raise TrainingConfigError(
            "Set only one of training.max_steps and training.num_train_epochs"
        )
    if max_steps is not None:
        if isinstance(max_steps, bool) or int(max_steps) <= 0:
            raise TrainingConfigError("training.max_steps must be a positive integer")
        _option(command, "max_steps", int(max_steps))
    else:
        if isinstance(num_train_epochs, bool) or float(num_train_epochs) <= 0:
            raise TrainingConfigError("training.num_train_epochs must be positive")
        _option(command, "num_train_epochs", float(num_train_epochs))
    _option(command, "per_device_train_batch_size", int(training.get("per_device_train_batch_size", 4)))
    _option(command, "per_device_eval_batch_size", int(training.get("per_device_eval_batch_size", 4)))
    _option(command, "gradient_accumulation_steps", int(training.get("gradient_accumulation_steps", 1)))
    default_learning_rate = 1e-5 if tuner_type == "full" else 1e-4
    _option(command, "learning_rate", training.get("learning_rate", default_learning_rate))
    if tuner_type == "lora":
        lora = require_mapping(config.get("lora"), "lora")
        _option(command, "lora_rank", int(lora.get("rank", 8)))
        _option(command, "lora_alpha", int(lora.get("alpha", 32)))
        target_modules = lora.get("target_modules", ["all-linear"])
        if not isinstance(target_modules, list) or not target_modules:
            raise TrainingConfigError("lora.target_modules must be a non-empty list")
        command.append("--target_modules")
        command.extend(str(value) for value in target_modules)
    for name, value in freeze.items():
        _option(command, f"freeze_{name}", _bool_text(value))

    simple_options = {
        "group_by_length": training.get("group_by_length", True),
        "use_liger_kernel": training.get("use_liger_kernel", True),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "max_length": int(training.get("max_length", 2048)),
        "warmup_ratio": training.get("warmup_ratio", 0.05),
        "eval_on_start": training.get("eval_on_start", False),
        "eval_steps": int(training.get("eval_steps", 50)),
        "save_steps": int(training.get("save_steps", 50)),
        "save_total_limit": int(training.get("save_total_limit", 2)),
        "logging_steps": int(training.get("logging_steps", 5)),
        "dataset_num_proc": int(training.get("dataset_num_proc", 4)),
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 4)),
        "seed": int(data_config.get("seed", 42)),
        "data_seed": int(data_config.get("seed", 42)),
    }
    for name, value in simple_options.items():
        _option(command, name, _bool_text(value) if isinstance(value, bool) else value)
    for name in (
        "eval_strategy",
        "save_strategy",
        "save_only_model",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        "optim",
    ):
        if name in training and training[name] is not None:
            value = training[name]
            _option(command, name, _bool_text(value) if isinstance(value, bool) else value)
    _option(command, "attn_impl", training.get("attn_impl", "flash_attention_2"))
    _option(command, "report_to", training.get("report_to", "tensorboard"))
    _option(command, "output_dir", model["output_dir"])
    resume_from_checkpoint = training.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        checkpoint_path = resolve_path(base_dir, str(resume_from_checkpoint))
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")
        _option(command, "resume_from_checkpoint", checkpoint_path)
    return command


def format_command(command: list[str], environment: dict[str, str]) -> str:
    prefixes = [
        f"CUDA_VISIBLE_DEVICES={shlex.quote(environment['CUDA_VISIBLE_DEVICES'])}",
        f"IMAGE_MAX_TOKEN_NUM={shlex.quote(environment['IMAGE_MAX_TOKEN_NUM'])}",
        "PYTORCH_CUDA_ALLOC_CONF="
        + shlex.quote(environment["PYTORCH_CUDA_ALLOC_CONF"]),
    ]
    return " ".join(prefixes + [shlex.join(command)])


def run_training(config: dict[str, Any], base_dir: Path) -> None:
    if uses_embodied_format(config):
        validate_cot_prepared_dataset(config, base_dir)
    else:
        validate_prepared_dataset(config, base_dir)
    command = build_swift_command(config, base_dir)
    subprocess.run(command, env=build_environment(config), check=True)
