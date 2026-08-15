from __future__ import annotations

import argparse
from pathlib import Path

from .inference import _load_model
from .schema import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the configured Exp 0 model once")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    args = parser.parse_args()
    torch_module, processor, model = _load_model(read_json(args.config.resolve()))
    allocated_gib = torch_module.cuda.memory_allocated() / (1024**3)
    print(f"processor={processor.__class__.__name__}")
    print(f"model={model.__class__.__name__}")
    print(f"device={model.device}")
    print(f"allocated_gib={allocated_gib:.3f}")


if __name__ == "__main__":
    main()
