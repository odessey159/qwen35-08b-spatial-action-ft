from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from exp0.generate_cot_data import extract_task_relevant_state, parse_response_sections
from exp0.subgoal_abstraction import abstract_subgoals
from training.common import TrainingConfigError, load_config
from training.cot_data import (
    _normalize_raw_simulator_sample,
    prepare_cot_dataset,
    validate_cot_prepared_dataset,
)


class RawCotTrainingTests(unittest.TestCase):
    def _raw_row(self, index: int, scene_id: str) -> dict:
        sample_id = f"raw_{index}"
        return {
            "sample_id": sample_id,
            "image": f"images/{sample_id}.png",
            "wrong_image": "",
            "instruction": "把 Apple 放到 Table 上。",
            "gold": {
                "plan_actions": [
                    "GotoLocation(Fridge)",
                    "OpenObject(Fridge)",
                    "PickupObject(Apple)",
                    "GotoLocation(Table)",
                    "PutObject(Apple,Table)",
                ],
                "plan_nl": "unused raw plan text",
            },
            "scene_graph": {
                "objects": [
                    {"id": "Agent", "type": "Agent", "attributes": {"visible": True}},
                    {"id": "Apple", "type": "Apple", "attributes": {"visible": True}},
                    {
                        "id": "Fridge",
                        "type": "Fridge",
                        "attributes": {"visible": True, "is_open": False},
                    },
                    {"id": "Table", "type": "Table", "attributes": {"visible": True}},
                    {"id": "Vase", "type": "Vase", "attributes": {"visible": True}},
                ],
                "relations": [
                    {"subject": "Apple", "relation": "inside", "object": "Fridge"},
                    {"subject": "Table", "relation": "near", "object": "Fridge"},
                    {"subject": "Vase", "relation": "left_of", "object": "Agent"},
                ],
            },
            "spatial_facts": [
                "Apple 位于 Fridge 里面。",
                "Vase 位于 Agent 左侧。",
            ],
            "subgoals": ["1. WRONG PLAN"],
            "meta": {
                "scene_id": scene_id,
                "counterfactual_group": None,
                "task_group": "put",
                "target_visible": True,
                "sim_verified": True,
                "required_object_ids": ["Apple", "Fridge", "Table"],
                "spatial_relations": [
                    ["Apple", "inside", "Fridge"],
                    ["Table", "near", "Fridge"],
                ],
            },
        }

    def _write_source(self, root: Path, count: int = 2) -> tuple[Path, list[dict]]:
        images = root / "images"
        images.mkdir()
        rows = [self._raw_row(index, f"scene_{index}") for index in range(1, count + 1)]
        for row in rows:
            (root / row["image"]).write_bytes(b"png")
        source = root / "samples.jsonl"
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return source, rows

    def _config(self) -> dict:
        return {
            "data": {
                "source_dataset": "samples.jsonl",
                "source_format": "raw_simulator",
                "prepared_dir": "prepared",
                "response_format": "cot",
                "validation_ratio": 0.5,
                "split_group_fields": ["scene_id", "counterfactual_group"],
                "require_sim_verified": True,
                "seed": 42,
            },
            "training": {
                "section_loss_weights": {"state": 0.3, "plan": 0.3, "action": 0.4}
            },
            "allowed_actions": [
                "GotoLocation",
                "OpenObject",
                "PickupObject",
                "PutObject",
            ],
        }

    def test_raw_adapter_preserves_actions_and_derives_plan_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, rows = self._write_source(root, count=1)
            row = rows[0]
            sample = _normalize_raw_simulator_sample(row, 1, root, True)

            self.assertEqual(sample.actions, tuple(row["gold"]["plan_actions"]))
            self.assertEqual(sample.plan, tuple(abstract_subgoals(sample.actions)))
            self.assertEqual(sample.state, tuple(extract_task_relevant_state(row)))
            self.assertNotIn("WRONG PLAN", "\n".join(sample.plan))
            self.assertTrue(all(not item.lstrip().startswith(("1.", "2.")) for item in sample.plan))

    def test_raw_prepare_validate_and_rendered_cot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root)
            config = self._config()

            manifest = prepare_cot_dataset(config, root, overwrite=False)
            self.assertEqual(manifest["source_format"], "raw_simulator")
            self.assertEqual(manifest["total_samples"], 2)
            self.assertEqual(
                validate_cot_prepared_dataset(config, root), {"train": 1, "val": 1}
            )

            row = json.loads(
                (root / "prepared" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            assistants = row["messages"][2:]
            self.assertEqual([message["loss_scale"] for message in assistants], [0.3, 0.3, 0.4])
            rendered = "".join(message["content"] for message in assistants)
            self.assertIn("<state>", rendered)
            self.assertIn("<plan>\n1. Acquire Apple.\n", rendered)
            self.assertIn("<action>", rendered)
            self.assertEqual(
                parse_response_sections(rendered)["action"],
                self._raw_row(1, "scene_1")["gold"]["plan_actions"],
            )

    def test_expected_source_samples_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_source(root, count=2)
            config = self._config()
            config["data"]["expected_source_samples"] = 3
            with self.assertRaisesRegex(
                TrainingConfigError, "expected 3, found 2"
            ):
                prepare_cot_dataset(config, root, overwrite=False)

    def test_validate_rejects_stale_prepared_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self._write_source(root)
            config = self._config()
            prepare_cot_dataset(config, root, overwrite=False)
            with source.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(TrainingConfigError, "Prepared data is stale"):
                validate_cot_prepared_dataset(config, root)

    def test_raw_adapter_keeps_sim_verification_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, rows = self._write_source(root, count=1)
            unverified = copy.deepcopy(rows[0])
            unverified["meta"]["sim_verified"] = False
            with self.assertRaisesRegex(TrainingConfigError, "simulator verification"):
                _normalize_raw_simulator_sample(unverified, 1, root, True)

    def test_future_pilot_config_does_not_override_training_hyperparameters(self) -> None:
        training_dir = Path(__file__).resolve().parents[1] / "training"
        pilot, _ = load_config(training_dir / "config.cot.pilot5000.server.json")
        baseline, _ = load_config(training_dir / "config.cot.server.json")
        self.assertEqual(pilot["training"], baseline["training"])
        self.assertEqual(pilot["data"]["source_format"], "raw_simulator")
        self.assertEqual(pilot["data"]["expected_source_samples"], 5000)
        self.assertEqual(
            pilot["data"]["source_dataset"],
            "../exp0/new5000_data/samples.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
