from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .evaluation import evaluate
from .inference import run_inference
from .schema import read_json, read_jsonl, validate_samples


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = config_path.resolve()
    config = read_json(config_path)
    parent = config.get("extends")
    if parent is not None:
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("config.extends must be a non-empty path")
        parent_path = Path(parent).expanduser()
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        config = _merge_config(read_json(parent_path.resolve()), config)
    base_dir = config_path.parent
    dataset_path = (base_dir / config["dataset_path"]).resolve()
    output_dir = (base_dir / config["output_dir"]).resolve()
    return config, dataset_path, output_dir


def _validate(config: dict[str, Any], dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Copy samples.example.jsonl to samples.jsonl and fill it."
        )
    samples = read_jsonl(dataset_path)
    errors = validate_samples(
        samples=samples,
        dataset_dir=dataset_path.parent,
        allowed_actions=set(config["allowed_actions"]),
        allowed_objects=set(config.get("allowed_objects", [])),
        min_samples=int(config.get("min_samples", 200)),
        max_samples=int(config.get("max_samples", 300)),
    )
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Exp 0 dataset validation failed:\n{formatted}")
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B Exp 0 diagnostic runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Path to exp0 config JSON",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the 200-300 diagnostic samples")

    infer_parser = subparsers.add_parser("infer", help="Run all Exp 0 inference conditions")
    infer_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing prediction file instead of resuming it",
    )

    subparsers.add_parser("evaluate", help="Evaluate saved predictions and diagnose bottlenecks")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, dataset_path, output_dir = _load_config(args.config)
    samples = _validate(config, dataset_path)
    predictions_path = output_dir / "predictions.jsonl"

    if args.command == "validate":
        print(f"Validated {len(samples)} Exp 0 samples: {dataset_path}")
        return
    if args.command == "infer":
        run_inference(
            config=config,
            dataset_path=dataset_path,
            output_path=predictions_path,
            overwrite=bool(args.overwrite),
        )
        return
    if args.command == "evaluate":
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Predictions not found: {predictions_path}")
        evaluate(
            config=config,
            dataset_path=dataset_path,
            predictions_path=predictions_path,
            output_dir=output_dir,
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
