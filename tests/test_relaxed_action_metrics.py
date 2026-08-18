from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_relaxed_action_metrics import parse_relaxed_action_plan  # noqa: E402


OBJECTS = {"Book", "Drawer", "Fridge", "Plate"}


class RelaxedActionParserTests(unittest.TestCase):
    def test_canonical_action_contract_is_preserved(self) -> None:
        result = parse_relaxed_action_plan(
            "<action>\nGotoLocation(Fridge)\nOpenObject(Fridge)\n</action>",
            OBJECTS,
        )
        self.assertTrue(result.deterministic_plan)
        self.assertEqual(
            result.actions,
            ("GotoLocation(Fridge)", "OpenObject(Fridge)"),
        )

    def test_minor_contract_format_errors_are_recovered(self) -> None:
        result = parse_relaxed_action_plan(
            "<action>\n1. GotoLocation: Fridge\n- OpenObject Fridge\n</action>",
            OBJECTS,
        )
        self.assertEqual(
            result.actions,
            ("GotoLocation(Fridge)", "OpenObject(Fridge)"),
        )

    def test_explicit_action_names_in_prose_are_recovered(self) -> None:
        text = (
            "<plan>\n"
            "1. 使用 GotoLocation 将玩家移动到 Fridge 位置。\n"
            "2. 使用 OpenObject 打开 Fridge。\n"
            "</plan>\n<action>GotoLocation</action>"
        )
        result = parse_relaxed_action_plan(text, OBJECTS)
        self.assertEqual(
            result.actions,
            ("GotoLocation(Fridge)", "OpenObject(Fridge)"),
        )
        self.assertEqual(result.source, "natural_plan")

    def test_explicit_chinese_actions_are_recovered(self) -> None:
        result = parse_relaxed_action_plan(
            "<plan>\n1. 前往 Fridge。\n2. 打开 Fridge。\n</plan>",
            OBJECTS,
        )
        self.assertEqual(
            result.actions,
            ("GotoLocation(Fridge)", "OpenObject(Fridge)"),
        )

    def test_two_argument_action_uses_only_explicit_objects(self) -> None:
        result = parse_relaxed_action_plan(
            "<plan>\n1. 将 Book 放入 Drawer。\n</plan>", OBJECTS
        )
        self.assertEqual(result.actions, ("PutObject(Book,Drawer)",))

    def test_conditional_branches_do_not_become_a_plan(self) -> None:
        result = parse_relaxed_action_plan(
            "<plan>\n"
            "1. 如果 Drawer 未打开，执行 OpenObject Drawer。\n"
            "2. 如果 Drawer 已打开，执行 CloseObject Drawer。\n"
            "</plan>",
            OBJECTS,
        )
        self.assertTrue(result.relaxed_parseable)
        self.assertFalse(result.deterministic_plan)
        self.assertEqual(result.actions, ())

    def test_missing_object_is_not_filled_from_external_context(self) -> None:
        result = parse_relaxed_action_plan(
            "<plan>\n1. 前往目标。\n2. 打开它。\n</plan>", OBJECTS
        )
        self.assertTrue(result.relaxed_parseable)
        self.assertFalse(result.deterministic_plan)
        self.assertEqual(result.actions, ())

    def test_move_object_is_not_misread_as_navigation(self) -> None:
        result = parse_relaxed_action_plan(
            "<plan>\n1. 将 Book 移动到 Drawer。\n</plan>", OBJECTS
        )
        self.assertFalse(result.deterministic_plan)
        self.assertEqual(result.actions, ())


if __name__ == "__main__":
    unittest.main()
