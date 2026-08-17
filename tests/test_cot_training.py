from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from exp0.generate_cot_data import parse_response_sections
from training.common import load_config
from training.cot_data import prepare_cot_dataset, validate_cot_prepared_dataset
from training.launcher import build_swift_command


class CotTrainingTests(unittest.TestCase):
    def test_ablation_configs_inherit_each_c_run_except_target_specific_fields(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        training_dir = repository / "training"
        expected = {
            "action": ("action", {"action": 1.0}),
            "plan": (
                "plan_action",
                {
                    "plan": 0.42857142857142855,
                    "action": 0.5714285714285714,
                },
            ),
        }

        with tempfile.TemporaryDirectory() as temporary:
            fake_swift = Path(temporary) / "swift"
            fake_swift.write_text("", encoding="utf-8")
            for scale in ("10k", "100k"):
                c_config, base_dir = load_config(
                    training_dir / f"config.cot.{scale}.server.json"
                )
                c_config["server"]["swift_executable"] = str(fake_swift)

                for arm, (response_format, weights) in expected.items():
                    with self.subTest(scale=scale, arm=arm):
                        arm_config, _ = load_config(
                            training_dir / f"config.{arm}.{scale}.server.json"
                        )
                        arm_config["server"]["swift_executable"] = str(fake_swift)

                        self.assertEqual(arm_config["server"], c_config["server"])
                        self.assertEqual(
                            arm_config["freeze_policy"], c_config["freeze_policy"]
                        )
                        self.assertEqual(
                            arm_config["allowed_actions"], c_config["allowed_actions"]
                        )
                        self.assertEqual(
                            arm_config["data"]["source_dataset"],
                            c_config["data"]["source_dataset"],
                        )
                        self.assertEqual(
                            arm_config["data"]["source_format"],
                            c_config["data"]["source_format"],
                        )
                        self.assertEqual(
                            arm_config["data"]["response_format"], response_format
                        )
                        self.assertEqual(
                            arm_config["training"]["section_loss_weights"], weights
                        )

                        c_training = dict(c_config["training"])
                        arm_training = dict(arm_config["training"])
                        c_training.pop("section_loss_weights")
                        arm_training.pop("section_loss_weights")
                        self.assertEqual(arm_training, c_training)

                        c_command = build_swift_command(c_config, base_dir)
                        arm_command = build_swift_command(arm_config, base_dir)
                        for command in (c_command, arm_command):
                            for option in ("--dataset", "--val_dataset", "--output_dir"):
                                if option in command:
                                    position = command.index(option)
                                    del command[position : position + 2]
                        self.assertEqual(arm_command, c_command)

    def test_10k_config_evaluates_before_training_and_keeps_checkpoints(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, base_dir = load_config(
            repository / "training" / "config.cot.10k.server.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            fake_swift = Path(temporary) / "swift"
            fake_swift.write_text("", encoding="utf-8")
            config["server"]["swift_executable"] = str(fake_swift)
            command = build_swift_command(config, base_dir)
        self.assertIn("--eval_on_start", command)
        self.assertEqual(command[command.index("--eval_on_start") + 1], "true")
        self.assertNotIn("--max_steps", command)
        self.assertEqual(command[command.index("--num_train_epochs") + 1], "1.0")
        self.assertEqual(command[command.index("--eval_steps") + 1], "250")
        self.assertEqual(command[command.index("--save_steps") + 1], "250")
        self.assertEqual(command[command.index("--save_total_limit") + 1], "10")

    def test_format_only_arm_is_a_separate_permuted_triplet_run(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, _ = load_config(
            repository / "training" / "config.cot.format-only.10k.server.json"
        )
        self.assertEqual(config["data"]["training_label_mode"], "permuted_triplet")
        self.assertTrue(config["data"]["prepared_dir"].endswith("cot-10k-format-only"))
        self.assertTrue(config["model"]["output_dir"].endswith("cot-10k-format-only"))

    def test_benchmark_only_options_are_emitted_when_configured(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, base_dir = load_config(
            repository / "training" / "config.cot.10k.server.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            fake_swift = Path(temporary) / "swift"
            fake_swift.write_text("", encoding="utf-8")
            config["server"]["swift_executable"] = str(fake_swift)
            config["training"].update(
                {
                    "eval_strategy": "no",
                    "save_strategy": "no",
                    "save_only_model": True,
                    "dataloader_persistent_workers": True,
                    "dataloader_prefetch_factor": 4,
                    "optim": "adamw_torch_fused",
                }
            )
            command = build_swift_command(config, base_dir)
        for name, expected in {
            "--eval_strategy": "no",
            "--save_strategy": "no",
            "--save_only_model": "true",
            "--dataloader_persistent_workers": "true",
            "--dataloader_prefetch_factor": "4",
            "--optim": "adamw_torch_fused",
        }.items():
            self.assertEqual(command[command.index(name) + 1], expected)

    def test_100k_config_uses_benchmarked_batches_and_keeps_periodic_checkpoints(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, base_dir = load_config(
            repository / "training" / "config.cot.100k.server.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            fake_swift = Path(temporary) / "swift"
            fake_swift.write_text("", encoding="utf-8")
            config["server"]["swift_executable"] = str(fake_swift)
            command = build_swift_command(config, base_dir)
        expected = {
            "--per_device_train_batch_size": "8",
            "--per_device_eval_batch_size": "8",
            "--gradient_accumulation_steps": "1",
            "--gradient_checkpointing": "false",
            "--eval_on_start": "true",
            "--eval_steps": "1000",
            "--save_steps": "1000",
            "--save_total_limit": "10",
        }
        for name, value in expected.items():
            self.assertEqual(command[command.index(name) + 1], value)

    def test_three_experiments_prepare_and_validate(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        training_dir = repository / "training"
        cases = {
            "config.action.10k.server.json": ("action", 1, [1.0]),
            "config.cot.10k.server.json": ("cot", 3, [0.3, 0.3, 0.4]),
            "config.plan.10k.server.json": (
                "plan_action",
                2,
                [0.42857142857142855, 0.5714285714285714],
            ),
        }
        prepared_actions = {}
        prepared_inputs = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_swift = root / "swift"
            fake_swift.write_text("", encoding="utf-8")
            for filename, (response_format, block_count, scales) in cases.items():
                with self.subTest(config=filename):
                    config, base_dir = load_config(training_dir / filename)
                    output = root / response_format
                    config["server"]["swift_executable"] = str(fake_swift)
                    config["data"].update(
                        {
                            "source_dataset": "../exp0/data_cot/samples_cot.jsonl",
                            "source_format": "cot",
                            "prepared_dir": str(output),
                            "expected_source_samples": 240,
                        }
                    )
                    manifest = prepare_cot_dataset(config, base_dir, overwrite=False)
                    self.assertEqual(manifest["total_samples"], 240)
                    counts = validate_cot_prepared_dataset(config, base_dir)
                    self.assertEqual(counts["train"] + counts["val"], 240)

                    row = json.loads(
                        (output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
                    )
                    self.assertNotIn("格式示例", row["messages"][0]["content"])
                    assistant_messages = row["messages"][2:]
                    if scales:
                        self.assertEqual(len(assistant_messages), block_count)
                        self.assertTrue(
                            all(
                                message["role"] == "assistant"
                                and isinstance(message["content"], str)
                                for message in assistant_messages
                            )
                        )
                        self.assertEqual(
                            [message["loss_scale"] for message in assistant_messages],
                            scales,
                        )
                        content = "".join(
                            message["content"] for message in assistant_messages
                        )
                    else:
                        self.assertEqual(len(assistant_messages), 1)
                        self.assertIsInstance(assistant_messages[0]["content"], str)
                        self.assertNotIn("loss_scale", assistant_messages[0])
                        content = assistant_messages[0]["content"]
                    sections = parse_response_sections(content)
                    train_rows = [
                        json.loads(line)
                        for line in (output / "train.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
                    prepared_inputs[response_format] = [
                        (item["messages"][1]["content"], item["images"])
                        for item in train_rows
                    ]
                    prepared_actions[response_format] = [
                        parse_response_sections(
                            "".join(
                                message["content"]
                                for message in item["messages"][2:]
                            )
                        )["action"]
                        for item in train_rows
                    ]
                    self.assertTrue(sections["action"])
                    self.assertEqual(bool(sections["state"]), response_format == "cot")
                    self.assertEqual(
                        bool(sections["plan"]), response_format in {"cot", "plan_action"}
                    )

                    command = build_swift_command(config, base_dir)
                    if scales:
                        self.assertIn("--is_binary_loss_scale", command)
                        self.assertEqual(
                            command[command.index("--is_binary_loss_scale") + 1], "false"
                        )
                        self.assertEqual(
                            command[command.index("--use_liger_kernel") + 1], "false"
                        )

        self.assertEqual(prepared_inputs["action"], prepared_inputs["plan_action"])
        self.assertEqual(prepared_inputs["action"], prepared_inputs["cot"])
        self.assertEqual(prepared_actions["action"], prepared_actions["plan_action"])
        self.assertEqual(prepared_actions["action"], prepared_actions["cot"])

    def test_format_only_prepare_deranges_training_triplets_but_not_validation(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        config, base_dir = load_config(repository / "training" / "config.cot.server.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aligned_dir = root / "aligned"
            permuted_dir = root / "permuted"

            config["data"]["prepared_dir"] = str(aligned_dir)
            aligned_manifest = prepare_cot_dataset(config, base_dir, overwrite=False)

            config["data"]["prepared_dir"] = str(permuted_dir)
            config["data"]["training_label_mode"] = "permuted_triplet"
            permuted_manifest = prepare_cot_dataset(config, base_dir, overwrite=False)
            validate_cot_prepared_dataset(config, base_dir)

            self.assertEqual(
                aligned_manifest["train_sample_ids"], permuted_manifest["train_sample_ids"]
            )
            stats = permuted_manifest["training_label_permutation"]
            self.assertEqual(stats["identical_label_triplets_after_permutation"], 0)

            aligned_train = [
                json.loads(line)
                for line in (aligned_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            permuted_train = [
                json.loads(line)
                for line in (permuted_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["messages"][:2] for row in aligned_train],
                [row["messages"][:2] for row in permuted_train],
            )
            self.assertTrue(
                all(
                    left["messages"][2:] != right["messages"][2:]
                    for left, right in zip(aligned_train, permuted_train)
                )
            )
            self.assertEqual(
                (aligned_dir / "val.jsonl").read_text(encoding="utf-8"),
                (permuted_dir / "val.jsonl").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
