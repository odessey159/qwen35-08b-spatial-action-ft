from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config
from .data import prepare_dataset, validate_prepared_dataset
from .launcher import build_environment, build_swift_command, format_command, run_training
from .runtime import check_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B spatial-action ms-swift trainer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.server.json"),
        help="Training configuration JSON",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Convert and split source data")
    prepare.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("validate", help="Validate prepared train/validation JSONL")
    subparsers.add_parser("check-runtime", help="Check GPU, packages, and model files")
    subparsers.add_parser("show-command", help="Print the exact swift sft command")
    subparsers.add_parser("train", help="Validate data and launch swift sft")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, base_dir = load_config(args.config)
    if args.command == "prepare":
        result = prepare_dataset(config, base_dir, bool(args.overwrite))
    elif args.command == "validate":
        result = validate_prepared_dataset(config, base_dir)
    elif args.command == "check-runtime":
        result = check_runtime(config)
    elif args.command == "show-command":
        command = build_swift_command(config, base_dir)
        print(format_command(command, build_environment(config)))
        return
    elif args.command == "train":
        run_training(config, base_dir)
        return
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

