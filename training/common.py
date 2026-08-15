from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrainingConfigError(ValueError):
    """Raised when the training configuration is incomplete or invalid."""


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TrainingConfigError(f"Config must contain a JSON object: {config_path}")
    return config, config_path.parent


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"'{name}' must be a JSON object")
    return value

