from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.select_raw_subset import select_raw_subset


class SelectRawSubsetTests(unittest.TestCase):
    def _make_shards(self, root: Path) -> None:
        for shard_index in range(2):
            shard_dir = root / f"shard_{shard_index}"
            images_dir = shard_dir / "images"
            images_dir.mkdir(parents=True)
            rows = []
            for scene_index, scene_size in enumerate((2, 3, 4)):
                for row_index in range(scene_size):
                    sample_id = f"exp0_{len(rows) + 1:04d}"
                    image = images_dir / f"{sample_id}.png"
                    image.write_bytes(b"png")
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "image": f"images/{image.name}",
                            "instruction": "拿起 Apple。",
                            "gold": {
                                "plan_actions": [
                                    "GotoLocation(Apple)",
                                    "PickupObject(Apple)",
                                ]
                            },
                            "meta": {
                                "scene_id": f"scene_{shard_index}_{scene_index}",
                                "counterfactual_group": (
                                    f"cf_{scene_index}" if row_index < 2 else None
                                ),
                                "task_group": "pickup",
                                "target_visible": True,
                                "sim_verified": True,
                            },
                        }
                    )
            (shard_dir / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (shard_dir / "generation_report.json").write_text(
                json.dumps({"sample_count": len(rows)}), encoding="utf-8"
            )

    def test_exact_deterministic_subset_preserves_whole_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_root = root / "shards"
            self._make_shards(shard_root)
            output = root / "selected"
            manifest = select_raw_subset(
                shard_root, 2, output, sample_count=7, seed=42, overwrite=False
            )
            rows = [
                json.loads(line)
                for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 7)
            self.assertEqual(manifest["selected_samples"], 7)
            self.assertEqual(len({row["sample_id"] for row in rows}), 7)
            self.assertTrue(all(Path(row["image"]).is_absolute() for row in rows))

            selected_scene_counts = {}
            for row in rows:
                scene = row["meta"]["scene_id"]
                selected_scene_counts[scene] = selected_scene_counts.get(scene, 0) + 1
                if row["meta"].get("counterfactual_group"):
                    self.assertTrue(
                        row["meta"]["counterfactual_group"].startswith("shard_")
                    )
            self.assertTrue(all(count in {2, 3, 4} for count in selected_scene_counts.values()))

            second = root / "selected-again"
            second_manifest = select_raw_subset(
                shard_root, 2, second, sample_count=7, seed=42, overwrite=False
            )
            self.assertEqual(manifest["output_sha256"], second_manifest["output_sha256"])

    def test_complete_unreported_atomic_tail_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_root = root / "shards"
            self._make_shards(shard_root)
            report_path = shard_root / "shard_0" / "generation_report.json"
            report_path.write_text(json.dumps({"sample_count": 8}), encoding="utf-8")
            manifest = select_raw_subset(
                shard_root,
                2,
                root / "selected",
                sample_count=7,
                seed=42,
                overwrite=False,
            )
            self.assertEqual(manifest["sources"][0]["unreported_tail_samples"], 1)


if __name__ == "__main__":
    unittest.main()
