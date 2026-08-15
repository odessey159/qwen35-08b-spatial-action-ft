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
    def test_three_experiments_prepare_and_validate(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        training_dir = repository / "training"
        cases = {
            "config.action.server.json": ("action", 1, []),
            "config.cot.server.json": ("cot", 3, [0.3, 0.3, 0.4]),
            "config.plan.server.json": ("plan_action", 2, [0.4, 0.6]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_swift = root / "swift"
            fake_swift.write_text("", encoding="utf-8")
            for filename, (response_format, block_count, scales) in cases.items():
                with self.subTest(config=filename):
                    config, base_dir = load_config(training_dir / filename)
                    output = root / response_format
                    config["server"]["swift_executable"] = str(fake_swift)
                    config["data"]["prepared_dir"] = str(output)
                    manifest = prepare_cot_dataset(config, base_dir, overwrite=False)
                    self.assertEqual(manifest["total_samples"], 240)
                    counts = validate_cot_prepared_dataset(config, base_dir)
                    self.assertEqual(counts["train"] + counts["val"], 240)

                    row = json.loads(
                        (output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
                    )
                    assistant = row["messages"][2]
                    if scales:
                        self.assertEqual(len(assistant["content"]), block_count)
                        self.assertEqual(assistant["loss_scale"], scales)
                        content = "".join(block["text"] for block in assistant["content"])
                    else:
                        self.assertIsInstance(assistant["content"], str)
                        self.assertNotIn("loss_scale", assistant)
                        content = assistant["content"]
                    sections = parse_response_sections(content)
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


if __name__ == "__main__":
    unittest.main()
