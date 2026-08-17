from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.clean_raw_shards import clean_raw_shards


class CleanRawShardsTests(unittest.TestCase):
    def _row(self, sample_id: str, image: str, plan: list[str]) -> dict:
        return {
            "sample_id": sample_id,
            "image": image,
            "instruction": "Pick up Apple.",
            "gold": {"plan_actions": plan},
            "spatial_facts": ["Apple is visible."],
            "meta": {
                "scene_id": "scene_1",
                "sim_verified": True,
                "target_visible": True,
                "image_sha256": "same-image",
            },
        }

    def test_filters_explicit_failures_and_all_ambiguous_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shards" / "shard_0"
            images = shard / "images"
            images.mkdir(parents=True)
            for name in ("a.png", "b.png", "c.png"):
                (images / name).write_bytes(b"image")
            rows = [
                self._row("a", "images/a.png", ["PickupObject(Apple)"]),
                self._row("b", "images/b.png", ["GotoLocation(Apple)", "PickupObject(Apple)"]),
                self._row("c", "images/c.png", ["PickupObject(Apple)"]),
            ]
            rows[2]["status"] = "failed"
            with (shard / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            (shard / "generation_report.json").write_text(
                json.dumps({"sample_count": 3, "instruction_collision": 1, "rejections": {}}),
                encoding="utf-8",
            )

            report = clean_raw_shards(root / "shards", root / "clean", 1, False)

            self.assertEqual(report["input_samples"], 3)
            self.assertEqual(report["clean_samples"], 0)
            self.assertEqual(report["failed_samples"], 3)
            self.assertEqual(report["collision_inputs"], 1)
            self.assertEqual(
                report["failure_counts"][
                    "same_image_and_instruction_map_to_multiple_gold_plans"
                ],
                2,
            )

    def test_keeps_verified_non_ambiguous_sample_and_resolves_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shards" / "shard_0"
            images = shard / "images"
            images.mkdir(parents=True)
            (images / "a.png").write_bytes(b"image")
            row = self._row("a", "images/a.png", ["PickupObject(Apple)"])
            (shard / "samples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (shard / "generation_report.json").write_text(
                json.dumps({"sample_count": 1, "rejections": {"simulation_failed": 4}}),
                encoding="utf-8",
            )

            report = clean_raw_shards(root / "shards", root / "clean", 1, False)

            self.assertEqual(report["clean_samples"], 1)
            cleaned = json.loads((root / "clean" / "samples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(cleaned["sample_id"], "shard_0_a")
            self.assertTrue(Path(cleaned["image"]).is_absolute())
            self.assertEqual(
                report["generation_rejections_not_present_in_samples"],
                {"simulation_failed": 4},
            )

    def test_accepts_complete_unreported_tail_after_finalization_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shards" / "shard_0"
            images = shard / "images"
            images.mkdir(parents=True)
            rows = []
            for index in range(2):
                name = f"{index}.png"
                (images / name).write_bytes(str(index).encode())
                row = self._row(str(index), f"images/{name}", ["PickupObject(Apple)"])
                row["meta"]["image_sha256"] = f"image-{index}"
                rows.append(row)
            with (shard / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            (shard / "generation_report.json").write_text(
                json.dumps(
                    {
                        "sample_count": 1,
                        "expected_sample_count": 2,
                        "complete": False,
                        "rejections": {},
                    }
                ),
                encoding="utf-8",
            )

            report = clean_raw_shards(root / "shards", root / "clean", 1, False)

            self.assertEqual(report["input_samples"], 2)
            self.assertEqual(report["clean_samples"], 2)
            self.assertEqual(report["source_reports"][0]["unreported_tail_samples"], 1)

    def test_filters_sample_that_cannot_produce_cot_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shards" / "shard_0"
            images = shard / "images"
            images.mkdir(parents=True)
            (images / "a.png").write_bytes(b"image")
            row = self._row("a", "images/a.png", ["PickupObject(Apple)"])
            row.pop("spatial_facts")
            (shard / "samples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (shard / "generation_report.json").write_text(
                json.dumps({"sample_count": 1, "rejections": {}}), encoding="utf-8"
            )

            report = clean_raw_shards(root / "shards", root / "clean", 1, False)

            self.assertEqual(report["clean_samples"], 0)
            self.assertEqual(report["failure_counts"], {"preprocess_no_relevant_state": 1})


if __name__ == "__main__":
    unittest.main()
