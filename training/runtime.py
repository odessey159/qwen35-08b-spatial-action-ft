from __future__ import annotations

import importlib.metadata
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .common import TrainingConfigError, require_mapping


REQUIRED_DISTRIBUTIONS = {
    "ms-swift": "4.4.2",
    "transformers": "5.9",
    "qwen-vl-utils": "0.0.14",
    "decord": None,
    "peft": None,
    "liger-kernel": None,
    "flash-linear-attention": "0.4.2",
    "causal-conv1d": None,
    "flash-attn": "2.8.3",
}

REQUIRED_IMPORTS = {
    "swift": "ms-swift",
    "transformers": "transformers",
    "qwen_vl_utils": "qwen-vl-utils",
    "decord": "decord",
    "peft": "peft",
    "liger_kernel": "liger-kernel",
    "fla": "flash-linear-attention",
    "causal_conv1d": "causal-conv1d",
    "flash_attn": "flash-attn",
}


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_runtime(config: dict[str, Any]) -> dict[str, Any]:
    if sys.version_info < (3, 12):
        raise TrainingConfigError(
            f"Python 3.12+ is required by the Qwen3.5 FLA stack; found {sys.version.split()[0]}"
        )
    try:
        import torch
    except ImportError as exc:
        raise TrainingConfigError("PyTorch is not installed in this environment") from exc
    if not torch.cuda.is_available():
        raise TrainingConfigError("CUDA is not available to PyTorch")
    if not torch.cuda.is_bf16_supported():
        raise TrainingConfigError("The configured GPU does not support bfloat16")

    installed: dict[str, str] = {}
    for distribution, minimum in REQUIRED_DISTRIBUTIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TrainingConfigError(f"Required package is missing: {distribution}") from exc
        installed[distribution] = version
        if minimum and _version_tuple(version) < _version_tuple(minimum):
            raise TrainingConfigError(
                f"{distribution}>={minimum} is required; found {version}"
            )
    if installed["ms-swift"] != "4.4.2":
        raise TrainingConfigError(
            f"This launcher is pinned to ms-swift 4.4.2; found {installed['ms-swift']}"
        )

    for module, distribution in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise TrainingConfigError(
                f"Package '{distribution}' is installed but import '{module}' failed: {exc}"
            ) from exc

    server = require_mapping(config.get("server"), "server")
    model = require_mapping(config.get("model"), "model")
    swift_executable = Path(str(server.get("swift_executable", "swift")))
    if swift_executable.is_absolute():
        if not swift_executable.is_file():
            raise FileNotFoundError(f"swift executable does not exist: {swift_executable}")
    elif shutil.which(str(swift_executable)) is None:
        raise FileNotFoundError(f"swift executable is not on PATH: {swift_executable}")
    model_path = Path(str(model.get("path", "")))
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    architectures = model_config.get("architectures", [])
    if not any("Qwen3_5" in str(value) for value in architectures):
        raise TrainingConfigError(f"Unexpected model architectures: {architectures}")

    properties = torch.cuda.get_device_properties(0)
    minimum_memory = float(server.get("minimum_gpu_memory_gib", 20))
    memory_gib = properties.total_memory / (1024**3)
    if memory_gib < minimum_memory:
        raise TrainingConfigError(
            f"At least {minimum_memory:g} GiB GPU memory is configured; found {memory_gib:.2f} GiB"
        )
    expected_cuda = server.get("expected_torch_cuda")
    if expected_cuda and str(torch.version.cuda) != str(expected_cuda):
        raise TrainingConfigError(
            f"Expected PyTorch CUDA {expected_cuda}; found {torch.version.cuda}"
        )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": properties.name,
        "gpu_memory_gib": round(memory_gib, 2),
        "packages": installed,
        "model": str(model_path),
        "architectures": architectures,
    }
