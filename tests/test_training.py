from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from exp0.build_cot_dataset import build_cot_dataset
from training.common import TrainingConfigError, load_config
from training.data import prepare_dataset, validate_prepared_dataset
from training.launcher import build_swift_command


class TrainingDataTests(unittest.TestCase):
    def _config(self, root: Path) -> dict:
        fake_swift = root / "swift"
        fake_swift.write_text("", encoding="utf-8")
        return {
            "server": {"swift_executable": str(fake_swift)},
            "model": {"path": "/model/test", "output_dir": str(root / "output")},
            "data": {
                "source_dataset": "source.jsonl",
                "prepared_dir": "prepared",
                "validation_ratio": 0.5,
                "split_group_fields": ["scene_id", "counterfactual_group"],
                "require_sim_verified": True,
                "seed": 42,
            },
            "training": {"tuner_type": "lora", "max_steps": 3},
            "lora": {"rank": 8, "alpha": 32, "target_modules": ["all-linear"]},
            "freeze_policy": {"llm": False, "vit": True, "aligner": True},
            "allowed_actions": ["GotoLocation", "PickupObject"],
        }

    def test_prepare_supports_both_schemas_and_keeps_groups_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("one.png", "two.png", "three.png", "four.png"):
                (root / name).write_bytes(b"png")
            rows = [
                {
                    "sample_id": "a",
                    "image": "one.png",
                    "instruction": "拿杯子",
                    "gold": {
                        "plan_actions": ["GotoLocation(Table)", "PickupObject(Cup)"],
                        "plan_nl": "走到桌旁并拿起杯子。",
                    },
                    "meta": {"scene_id": "s1", "sim_verified": True},
                },
                {
                    "sample_id": "b",
                    "image": "two.png",
                    "prompt": "拿盘子",
                    "plan_actions": ["GotoLocation(Table)", "PickupObject(Plate)"],
                    "plan_nl": "走到桌旁并拿起盘子。",
                    "_meta": {"scene_id": "s1", "sim_verified": True},
                },
                {
                    "sample_id": "c",
                    "image": "three.png",
                    "prompt": "拿苹果",
                    "plan_actions": ["GotoLocation(CounterTop)", "PickupObject(Apple)"],
                    "plan_nl": "走到台面并拿起苹果。",
                    "_meta": {"scene_id": "s2", "sim_verified": True},
                },
                {
                    "sample_id": "d",
                    "image": "four.png",
                    "prompt": "拿刀",
                    "plan_actions": ["GotoLocation(CounterTop)", "PickupObject(Knife)"],
                    "plan_nl": "走到台面并拿起刀。",
                    "_meta": {"scene_id": "s2", "sim_verified": True},
                },
            ]
            source = root / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = self._config(root)
            manifest = prepare_dataset(config, root, overwrite=False)
            self.assertEqual(manifest["train_samples"], 2)
            self.assertEqual(manifest["validation_samples"], 2)
            train_ids = set(manifest["train_sample_ids"])
            self.assertTrue({"a", "b"}.issubset(train_ids) or {"a", "b"}.isdisjoint(train_ids))
            self.assertEqual(validate_prepared_dataset(config, root), {"train": 2, "val": 2})
            prepared_rows = [
                json.loads(line)
                for line in (root / "prepared" / "train.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            roles = [message["role"] for message in prepared_rows[0]["messages"]]
            self.assertEqual(roles, ["system", "user", "assistant"])
            self.assertIn("<summary>", prepared_rows[0]["messages"][0]["content"])
            self.assertIn("<image>", prepared_rows[0]["messages"][1]["content"])
            self.assertIn("<plan>", prepared_rows[0]["messages"][2]["content"])
            self.assertIn("<summary>", prepared_rows[0]["messages"][2]["content"])
            self.assertNotIn("ActionName", prepared_rows[0]["messages"][0]["content"])

    def test_launcher_blocks_unresolved_freeze_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config["freeze_policy"]["vit"] = None
            with self.assertRaisesRegex(TrainingConfigError, "intentionally unresolved"):
                build_swift_command(config, root)

    def test_full_cot_preparation_emits_intermediate_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(4):
                image = root / f"cot_{index}.png"
                image.write_bytes(b"png")
                rows.append(
                    {
                        "sample_id": f"cot_{index}",
                        "image": image.name,
                        "instruction": "拿起杯子",
                        "gold": {
                            "spatial_state": "Cup 位于 Table 上。",
                            "subgoals": "1. 到达杯子。\n2. 拿起杯子。",
                            "plan_actions": [
                                "GotoLocation(Table)",
                                "PickupObject(Cup)",
                            ],
                            "plan_nl": "前往 Table，然后拿起 Cup。",
                        },
                        "_meta": {
                            "scene_id": f"scene_{index // 2}",
                            "sim_verified": True,
                        },
                    }
                )
            (root / "cot.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = self._config(root)
            config["data"].update(
                {
                    "source_dataset": "cot.jsonl",
                    "prepared_dir": "prepared-cot",
                    "supervision_mode": "full_cot",
                }
            )
            prepare_dataset(config, root, overwrite=False)
            self.assertEqual(
                validate_prepared_dataset(config, root), {"train": 2, "val": 2}
            )
            row = json.loads(
                (root / "prepared-cot" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            assistant = row["messages"][2]
            content = assistant["content"]
            self.assertNotIn("loss_scale", assistant)
            self.assertLess(content.index("<state>"), content.index("<subgoal>"))
            self.assertLess(content.index("<subgoal>"), content.index("<plan>"))
            self.assertLess(content.index("<plan>"), content.index("<summary>"))

    def test_cot_builder_moves_simulator_annotations_into_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "images" / "sample.png").write_bytes(b"png")
            source = root / "samples.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_1",
                        "image": "images/sample.png",
                        "instruction": "打开抽屉",
                        "gold": {
                            "plan_actions": [
                                "GotoLocation(Drawer)",
                                "OpenObject(Drawer)",
                            ],
                            "plan_nl": "前往 Drawer，然后打开 Drawer。",
                        },
                        "spatial_facts": ["Drawer 靠近 Agent。", "Drawer 当前关闭。"],
                        "subgoals": ["前往 Drawer。", "打开 Drawer。"],
                        "meta": {"scene_id": "scene_1", "sim_verified": True},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "data_cot" / "train.jsonl"
            self.assertEqual(build_cot_dataset(source, output), 1)
            built = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                built["gold"]["spatial_state"],
                "Drawer 靠近 Agent。\nDrawer 当前关闭。",
            )
            self.assertEqual(
                built["gold"]["subgoals"], "1. 前往 Drawer。\n2. 打开 Drawer。"
            )
            self.assertEqual(built["image"], "../images/sample.png")
            self.assertTrue(built["_meta"]["sim_verified"])

    def test_launcher_emits_explicit_freeze_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            command = build_swift_command(config, root)
            joined = " ".join(command)
            self.assertIn("--target_modules all-linear", joined)
            self.assertIn("--freeze_llm false", joined)
            self.assertIn("--freeze_vit true", joined)
            self.assertIn("--freeze_aligner true", joined)

    def test_full_tuning_omits_lora_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config["training"] = {"tuner_type": "full", "max_steps": 3}
            config.pop("lora")
            config["freeze_policy"] = {"llm": False, "vit": False, "aligner": False}
            joined = " ".join(build_swift_command(config, root))
            self.assertIn("--tuner_type full", joined)
            self.assertIn("--learning_rate 1e-05", joined)
            self.assertNotIn("--lora_rank", joined)
            self.assertNotIn("--lora_alpha", joined)
            self.assertNotIn("--target_modules", joined)

    def test_cot_configs_keep_full_tuning_primary_and_lora_as_fallback(self) -> None:
        training_dir = Path(__file__).resolve().parents[1] / "training"
        full_config, _ = load_config(training_dir / "config.cot.server.json")
        lora_config, _ = load_config(training_dir / "config.lora.cot.server.json")
        self.assertEqual(full_config["training"]["tuner_type"], "full")
        self.assertEqual(lora_config["training"]["tuner_type"], "lora")
        for config in (full_config, lora_config):
            self.assertEqual(config["data"]["response_format"], "cot")
            self.assertEqual(
                config["data"]["source_dataset"],
                "../exp0/data_cot/samples_cot.jsonl",
            )
            self.assertEqual(
                config["training"]["section_loss_weights"],
                {"state": 0.3, "plan": 0.3, "action": 0.4},
            )
            self.assertFalse(config["training"]["use_liger_kernel"])


if __name__ == "__main__":
    unittest.main()
