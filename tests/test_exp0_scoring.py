from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp0.evaluation import diagnose  # noqa: E402
from exp0.prompts import (  # noqa: E402
    FORMAT_EXAMPLE_PLAN,
    build_messages,
    build_prompt_cases,
    system_prompt,
)
from exp0.schema import (  # noqa: E402
    extract_summary,
    parse_plan,
    parse_plan_lenient,
    score_summary,
)


ACTIONS = [
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
GOLD = ["GotoLocation(Fridge)", "OpenObject(Fridge)"]
GOLD_NL = "前往 Fridge，然后打开 Fridge。"
THRESHOLDS = {"d_min_score": 0.6, "near_gap": 0.03, "large_gap": 0.1, "floor_score": 0.15}


class PromptContractTests(unittest.TestCase):
    def test_no_literal_placeholder_anywhere_in_the_prompt(self) -> None:
        """The 0.8B copied `ActionName(Object)` back in 61% of D outputs."""
        for inline in (True, False):
            self.assertNotIn("ActionName", system_prompt(ACTIONS, inline_example=inline))

    def test_user_turn_carries_no_formatting_instructions(self) -> None:
        """Rules live in the system turn so a verbatim echo cannot look like an answer."""
        sample = {
            "instruction": "清洁 Plate。",
            "image": "a.png",
            "wrong_image": "b.png",
            "subgoals": ["前往 Plate。", "清洁 Plate。"],
            "spatial_facts": ["Plate 位于左侧 Agent。"],
            "scene_graph": {"objects": [], "relations": []},
        }
        cases = build_prompt_cases(sample, Path("/tmp"), ACTIONS)
        self.assertEqual(len(cases), 7)
        for case in cases:
            self.assertNotIn("随后用一句自然语言", case.prompt)
            self.assertNotIn("ActionName", case.prompt)
            self.assertNotIn("可用动作仅限", case.prompt)

    def test_format_example_uses_objects_absent_from_the_dataset(self) -> None:
        """Copying the example must stay detectable rather than inflate scores."""
        samples_path = Path(__file__).resolve().parents[1] / "exp0" / "data" / "samples.jsonl"
        if not samples_path.is_file():
            self.skipTest("diagnostic dataset not present")
        import json

        used: set[str] = set()
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for obj in row["scene_graph"]["objects"]:
                used.add(obj["type"])
        for action in FORMAT_EXAMPLE_PLAN:
            argument = action.split("(", 1)[1].rstrip(")")
            self.assertNotIn(argument, used)

    def test_messages_have_a_system_turn_and_optional_demo(self) -> None:
        plain = build_messages("目标指令：清洁 Plate。", None, ACTIONS)
        self.assertEqual([m["role"] for m in plain], ["system", "user"])
        demo = build_messages("目标指令：清洁 Plate。", None, ACTIONS, format_demo_as_turns=True)
        self.assertEqual([m["role"] for m in demo], ["system", "user", "assistant", "user"])


class LenientParsingTests(unittest.TestCase):
    def test_action_block_wins_over_high_level_cot_plan(self) -> None:
        text = (
            "<state>Fridge is closed.</state>\n"
            "<plan>\n1. Acquire Apple.\n</plan>\n"
            "<action>\nGotoLocation(Fridge)\nOpenObject(Fridge)\n</action>"
        )
        self.assertEqual(parse_plan_lenient(text, ACTIONS), GOLD)
        self.assertEqual(parse_plan(text).actions, GOLD)

    def test_missing_parentheses_are_recovered(self) -> None:
        """99.3% of the first run's plan lines were written without parentheses."""
        text = "<plan>\nGotoLocation Fridge\nOpenObject Fridge\n</plan>"
        self.assertEqual(parse_plan_lenient(text, ACTIONS), GOLD)
        self.assertFalse(parse_plan(text).structure_valid)

    def test_markers_casing_and_junk_lines(self) -> None:
        text = "<plan>\n好的，我来规划：\n1. gotolocation(Fridge)\n- OPENOBJECT Fridge\n</plan>"
        self.assertEqual(parse_plan_lenient(text, ACTIONS), GOLD)

    def test_out_of_vocabulary_heads_are_dropped(self) -> None:
        text = "<plan>\nActionName(Fridge)\nGotoLocation(Fridge)\nOpenObject(Fridge)\n</plan>"
        self.assertEqual(parse_plan_lenient(text, ACTIONS), GOLD)

    def test_strict_parser_is_unchanged(self) -> None:
        strict = parse_plan("<plan>\nGotoLocation(Fridge)\nOpenObject(Fridge)\n</plan>")
        self.assertTrue(strict.structure_valid)
        self.assertEqual(strict.actions, GOLD)


class SummaryScoringTests(unittest.TestCase):
    def test_summary_block_is_preferred(self) -> None:
        text = "<plan>\nGotoLocation(Fridge)\n</plan>\n<summary>\n走到 Fridge 前。\n</summary>"
        self.assertEqual(extract_summary(text), "走到 Fridge 前。")

    def test_unclosed_summary_still_extracts(self) -> None:
        self.assertEqual(extract_summary("<plan>\nx\n</plan>\n<summary>走到 Fridge。"), "走到 Fridge。")

    def test_echoed_contract_is_stripped_from_the_fallback(self) -> None:
        text = "<plan>\nGotoLocation Fridge\n</plan>\n随后用一句自然语言描述相同步骤。\n走到 Fridge 前。"
        self.assertEqual(extract_summary(text), "走到 Fridge 前。")

    def test_paraphrase_counts_as_correct(self) -> None:
        scores = score_summary("先走到 Fridge 那里，再把 Fridge 的门拉开。", GOLD, GOLD_NL)
        self.assertEqual(scores["nl_plan_match"], 1.0)

    def test_wrong_order_fails(self) -> None:
        scores = score_summary("先打开 Fridge，然后走到 Fridge。", GOLD, GOLD_NL)
        self.assertEqual(scores["nl_order_ok"], 0.0)
        self.assertEqual(scores["nl_plan_match"], 0.0)

    def test_missing_object_fails_even_with_all_verbs(self) -> None:
        scores = score_summary("走过去，把门打开。", GOLD, GOLD_NL)
        self.assertEqual(scores["nl_action_recall"], 1.0)
        self.assertEqual(scores["nl_object_recall"], 0.0)
        self.assertEqual(scores["nl_plan_match"], 0.0)

    def test_empty_summary_scores_zero(self) -> None:
        self.assertEqual(score_summary("", GOLD, GOLD_NL)["nl_present"], 0.0)


class DiagnosisGateTests(unittest.TestCase):
    @staticmethod
    def _summary(values: dict[str, float]) -> dict[str, dict[str, float | None]]:
        return {name: {"nl_plan_match": score} for name, score in values.items()}

    def test_low_d_suppresses_every_other_conclusion(self) -> None:
        """§6.1: if D fails, A/A'/B/C comparisons are measuring plumbing."""
        result = diagnose(
            self._summary(
                {"A": 0.0, "A_prime": 0.0, "B_natural": 0.0, "B_json": 0.0, "B_triples": 0.0, "C": 0.0, "D": 0.0}
            ),
            THRESHOLDS,
            "nl_plan_match",
        )
        self.assertFalse(result["d_gate_passed"])
        self.assertEqual(len(result["conclusions"]), 1)
        self.assertEqual(result["deltas"], {})
        self.assertNotIn(
            "模型没有有效使用图像", " ".join(result["conclusions"])
        )

    def test_passing_d_unlocks_the_ladder(self) -> None:
        result = diagnose(
            self._summary(
                {"A": 0.10, "A_prime": 0.09, "B_natural": 0.40, "B_json": 0.41, "B_triples": 0.39, "C": 0.38, "D": 0.85}
            ),
            THRESHOLDS,
            "nl_plan_match",
        )
        self.assertTrue(result["d_gate_passed"])
        joined = " ".join(result["conclusions"])
        self.assertIn("A 与 A′ 接近", joined)
        self.assertIn("B 显著高于 A", joined)

    def test_diverging_serializations_are_flagged_as_confounded(self) -> None:
        result = diagnose(
            self._summary(
                {"A": 0.10, "A_prime": 0.09, "B_natural": 0.60, "B_json": 0.20, "B_triples": 0.35, "C": 0.30, "D": 0.85}
            ),
            THRESHOLDS,
            "nl_plan_match",
        )
        self.assertIn("受序列化格式混淆", " ".join(result["conclusions"]))


if __name__ == "__main__":
    unittest.main()
