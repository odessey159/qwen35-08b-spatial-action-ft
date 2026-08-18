from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from training.evaluate_section_losses import load_jsonl
from training.generate_cpu_predictions import select_counterfactual_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prediction_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("prediction must use LABEL=PATH")
    return label, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a deterministic prefix of complete counterfactual pairs"
    )
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--prediction",
        type=_prediction_argument,
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")
    output_dir = args.output_dir.resolve()
    outputs = [output_dir / "val.jsonl", output_dir / "manifest.json"] + [
        output_dir / f"predictions-{label}.jsonl" for label, _ in args.prediction
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist (use --overwrite): {existing}")

    prepared_rows = load_jsonl(args.val_file.resolve())
    raw_rows = load_jsonl(args.raw_data.resolve())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_selected = select_counterfactual_rows(prepared_rows, raw_rows, manifest, 0)
    complete_pairs = [all_selected[index : index + 2] for index in range(0, len(all_selected), 2)]
    if args.max_pairs > len(complete_pairs):
        raise ValueError(
            f"Requested {args.max_pairs} pairs but only {len(complete_pairs)} are available"
        )
    pair_indices = sorted(
        random.Random(args.sample_seed).sample(range(len(complete_pairs)), args.max_pairs)
    )
    selected = [row for index in pair_indices for row in complete_pairs[index]]
    selected_ids = [sample_id for _, sample_id, _, _ in selected]
    if len(selected_ids) != args.max_pairs * 2:
        raise ValueError(
            f"Requested {args.max_pairs} pairs but selected {len(selected_ids) // 2}"
        )
    selected_id_set = set(selected_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "val.jsonl", [row for _, _, _, row in selected])
    subset_manifest = dict(manifest)
    subset_manifest.update(
        {
            "validation_sample_ids": selected_ids,
            "validation_samples": len(selected_ids),
            "evaluation_subset": {
                "strategy": "random_complete_counterfactual_pairs",
                "seed": args.sample_seed,
                "pair_count": args.max_pairs,
                "sample_count": len(selected_ids),
                "source_manifest": str(args.manifest.resolve()),
            },
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(subset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for label, prediction_path in args.prediction:
        by_id = {
            str(row.get("sample_id")): row
            for row in load_jsonl(prediction_path.resolve())
            if str(row.get("sample_id")) in selected_id_set
        }
        missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                f"{prediction_path} is missing {len(missing)} selected rows: {missing[:10]}"
            )
        _write_jsonl(
            output_dir / f"predictions-{label}.jsonl",
            [by_id[sample_id] for sample_id in selected_ids],
        )

    selection = {
        "pair_count": args.max_pairs,
        "sample_count": len(selected_ids),
        "seed": args.sample_seed,
        "source_pair_indices": pair_indices,
        "sample_ids": selected_ids,
        "groups": [group for _, _, group, _ in selected[::2]],
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"selected_pairs={args.max_pairs} selected_samples={len(selected_ids)}")
    print(output_dir)


if __name__ == "__main__":
    main()
