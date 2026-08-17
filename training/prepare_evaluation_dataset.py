from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from exp0.subgoal_abstraction import parse_primitive_action
from training.cot_data import (
    _normalize_raw_simulator_sample,
    _source_sha256,
    _swift_row,
)
from training.data import _read_jsonl, _write_json, _write_jsonl


ALLOWED_ACTIONS = [
    "GotoLocation",
    "PickupObject",
    "PutObject",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "ToggleObject",
    "OpenObject",
    "CloseObject",
]
SECTION_WEIGHTS = {"state": 0.3, "plan": 0.3, "action": 0.4}


def prepare_evaluation_dataset(
    source: Path,
    output_dir: Path,
    max_state_facts: int = 12,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    val_path = output_dir / "val.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if max_state_facts <= 0:
        raise ValueError("max_state_facts must be positive")
    if not overwrite and (val_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Evaluation outputs exist; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = _read_jsonl(source)
    counterfactual_counts = Counter(
        str(row.get("meta", {}).get("counterfactual_group"))
        for row in source_rows
        if row.get("meta", {}).get("counterfactual_group") not in (None, "")
    )
    incomplete_groups = {
        name for name, count in counterfactual_counts.items() if count != 2
    }
    rows = [
        row
        for row in source_rows
        if str(row.get("meta", {}).get("counterfactual_group")) not in incomplete_groups
    ]
    samples = [
        _normalize_raw_simulator_sample(
            row,
            index,
            source.parent,
            True,
            max_state_facts,
        )
        for index, row in enumerate(rows, start=1)
    ]
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation sample_id values are not unique")
    allowed = set(ALLOWED_ACTIONS)
    for sample in samples:
        unknown = {
            parse_primitive_action(action).name for action in sample.actions
        } - allowed
        if unknown:
            raise ValueError(f"{sample.sample_id}: unknown actions {sorted(unknown)}")
        if not sample.state:
            raise ValueError(f"{sample.sample_id}: CoT evaluation requires state facts")

    _write_jsonl(
        val_path,
        (_swift_row(sample, ALLOWED_ACTIONS, "cot", SECTION_WEIGHTS) for sample in samples),
    )
    manifest = {
        "source_dataset": str(source),
        "source_sha256": _source_sha256(source),
        "source_format": "raw_simulator",
        "response_format": "cot",
        "training_label_mode": "aligned",
        "section_loss_weights": SECTION_WEIGHTS,
        "evaluation_only": True,
        "source_samples_before_incomplete_cf_filter": len(source_rows),
        "dropped_incomplete_counterfactual_groups": len(incomplete_groups),
        "dropped_incomplete_counterfactual_samples": len(source_rows) - len(rows),
        "total_samples": len(samples),
        "train_samples": 0,
        "validation_samples": len(samples),
        "train_sample_ids": [],
        "validation_sample_ids": sample_ids,
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare every raw row as held-out CoT eval")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-state-facts", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_evaluation_dataset(
        args.source,
        args.output_dir,
        max_state_facts=args.max_state_facts,
        overwrite=bool(args.overwrite),
    )
    print(
        f"prepared_evaluation={manifest['validation_samples']} "
        f"output={args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
