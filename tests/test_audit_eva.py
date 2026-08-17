from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.audit_eva import (
    counterfactual_pair_audit,
    has_placeholder_copy,
    split_audit,
    structure_audit,
    text_oracle_audit,
)
from training.generate_cpu_predictions import select_all_rows, select_counterfactual_rows


def row(sample_id: str, scene: str, group: str | None, instruction: str, actions: list[str]):
    return {
        "sample_id": sample_id,
        "instruction": instruction,
        "gold": {"plan_actions": actions},
        "meta": {
            "scene_id": scene,
            "counterfactual_group": group,
            "sim_verified": True,
            "target_visible": True,
        },
    }


class AuditEvaTests(unittest.TestCase):
    def test_split_audit_detects_component_leakage(self) -> None:
        rows = [
            row("a", "scene-1", "shard_0_cf_1", "open", ["GotoLocation(Box)"]),
            row("b", "scene-1", "shard_0_cf_1", "open", ["OpenObject(Box)"]),
            row("c", "scene-2", None, "clean", ["CleanObject(Plate)"]),
        ]
        manifest = {
            "train_sample_ids": ["a", "c"],
            "validation_sample_ids": ["b"],
            "split_group_fields": ["scene_id", "counterfactual_group"],
        }
        audit = split_audit(rows, manifest)
        self.assertFalse(audit["merge_split_valid"])
        self.assertEqual(audit["cross_split_component_count"], 1)
        self.assertEqual(audit["scene_overlap_count"], 1)

    def test_text_oracle_distinguishes_ambiguous_instruction(self) -> None:
        rows = [
            row("a", "s1", None, "确保 Box 打开", ["GotoLocation(Box)"]),
            row("b", "s2", None, "确保 Box 打开", ["GotoLocation(Box)", "OpenObject(Box)"]),
            row("c", "s3", None, "清洁 Plate", ["GotoLocation(Plate)", "CleanObject(Plate)"]),
        ]
        manifest = {
            "train_sample_ids": ["a"],
            "validation_sample_ids": ["b", "c"],
        }
        audit = text_oracle_audit(rows, manifest)
        self.assertAlmostEqual(audit["filtered_text_deterministic_rate"], 0.5)
        self.assertAlmostEqual(audit["train_text_oracle_accuracy"], 0.0)
        self.assertAlmostEqual(audit["visual_lift_headroom_vs_train_oracle"], 1.0)
        self.assertAlmostEqual(audit["validation_text_bayes_oracle_accuracy"], 1.0)
        self.assertAlmostEqual(audit["irreducible_visual_ambiguity_rate"], 0.0)

    def test_placeholder_copy_detection(self) -> None:
        self.assertTrue(has_placeholder_copy("<plan>ActionName(Object)</plan>"))
        self.assertTrue(has_placeholder_copy("<action>动作名(物体名)</action>"))
        self.assertFalse(has_placeholder_copy("<action>OpenObject(Box)</action>"))

    def test_structure_and_counterfactual_pair_metrics(self) -> None:
        rows = [
            row("a", "s1", "shard_0_cf_1", "put cup", ["PickupObject(Cup)"]),
            row("b", "s2", "shard_0_cf_1", "put cup", ["OpenObject(Drawer)"]),
        ]
        manifest = {
            "train_sample_ids": ["train-only"],
            "validation_sample_ids": ["a", "b"],
        }
        predictions = [
            {
                "sample_id": "a",
                "prediction": (
                    "<state>visible</state><plan>same plan</plan>"
                    "<action>PickupObject(Cup)</action>"
                ),
            },
            {
                "sample_id": "b",
                "prediction": (
                    "<state>visible</state><plan>same plan</plan>"
                    "<action>OpenObject(Drawer)</action>"
                ),
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                "\n".join(json.dumps(value) for value in predictions) + "\n",
                encoding="utf-8",
            )
            structure = structure_audit(path, cot_contract=True, expected_count=4)
            pairs = counterfactual_pair_audit(rows, manifest, path)
        self.assertEqual(structure["status"], "partial")
        self.assertEqual(structure["coverage"], 0.5)
        self.assertEqual(structure["conditions"]["all"]["strict_structure_valid"], 1.0)
        self.assertEqual(pairs["evaluated_pairs"], 1)
        self.assertEqual(pairs["pair_exact_accuracy"], 1.0)
        self.assertEqual(pairs["sample_exact_accuracy"], 1.0)
        self.assertEqual(pairs["one_member_correct_pairs"], 0)
        self.assertEqual(pairs["zero_members_correct_pairs"], 0)
        self.assertEqual(pairs["same_plan_rate"], 1.0)
        self.assertEqual(pairs["same_action_sequence_rate"], 0.0)

    def test_cpu_generator_selects_both_members_of_complete_pairs(self) -> None:
        raw = [
            row("a", "s1", "cf-1", "put", ["PickupObject(Cup)"]),
            row("b", "s2", "cf-1", "put", ["OpenObject(Drawer)"]),
            row("c", "s3", "cf-incomplete", "open", ["OpenObject(Box)"]),
        ]
        prepared = [{"row": value} for value in ("a", "b", "c")]
        manifest = {"validation_sample_ids": ["a", "b", "c"]}
        selected = select_counterfactual_rows(prepared, raw, manifest, max_pairs=1)
        self.assertEqual([value[1] for value in selected], ["a", "b"])
        self.assertEqual({value[2] for value in selected}, {"cf-1"})

    def test_generator_all_selection_preserves_validation_order_and_limit(self) -> None:
        raw = [
            row("a", "s1", "cf-1", "put", ["PickupObject(Cup)"]),
            row("b", "s2", None, "open", ["OpenObject(Box)"]),
        ]
        prepared = [{"row": "a"}, {"row": "b"}]
        manifest = {"validation_sample_ids": ["a", "b"]}
        selected = select_all_rows(prepared, raw, manifest, max_samples=1)
        self.assertEqual(selected, [(0, "a", "cf-1", prepared[0])])


if __name__ == "__main__":
    unittest.main()
