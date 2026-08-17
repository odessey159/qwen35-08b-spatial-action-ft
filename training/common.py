from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrainingConfigError(ValueError):
    """Raised when the training configuration is incomplete or invalid."""


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and "$replace" in value:
            if set(value) != {"$replace"}:
                raise TrainingConfigError(
                    f"Config replacement for '{key}' must contain only '$replace'"
                )
            result[key] = value["$replace"]
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def _read_config(config_path: Path, seen: set[Path]) -> dict[str, Any]:
    if config_path in seen:
        raise TrainingConfigError(f"Circular config inheritance: {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TrainingConfigError(f"Config must contain a JSON object: {config_path}")
    parent_value = config.get("extends")
    if parent_value is None:
        return config
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise TrainingConfigError(f"'extends' must be a non-empty path: {config_path}")
    parent_path = Path(parent_value).expanduser()
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_path = parent_path.resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent config does not exist: {parent_path}")
    return _merge_config(_read_config(parent_path, seen), config)


def load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.resolve()
    config = _read_config(config_path, set())
    return config, config_path.parent


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"'{name}' must be a JSON object")
    return value
