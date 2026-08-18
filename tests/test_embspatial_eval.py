from __future__ import annotations

import unittest

from training.evaluate_embspatial import (
    answer_label,
    compare_predictions,
    format_question,
    parse_choice,
    summarize,
)


class EmbSpatialEvaluationTests(unittest.TestCase):
    def test_answer_label_accepts_official_zero_based_indices(self) -> None:
        self.assertEqual(answer_label(0), "A")
        self.assertEqual(answer_label(3), "D")
        self.assertEqual(answer_label("b"), "B")

    def test_prompt_has_stable_choice_labels(self) -> None:
        prompt = format_question("Where is the cup?", ["left", "right", "above", "below"])
        self.assertIn("A. left", prompt)
        self.assertIn("D. below", prompt)
        self.assertTrue(prompt.endswith("Answer with only A, B, C, or D."))

    def test_cot_prompt_requires_reasoning_then_final_answer(self) -> None:
        prompt = format_question(
            "Where is the cup?", ["left", "right", "above", "below"], prompt_style="cot"
        )
        self.assertIn("Reasoning:", prompt)
        self.assertIn("Final answer: X", prompt)
        self.assertTrue(prompt.index("Reasoning:") < prompt.index("Final answer:"))
        self.assertIn("Do not use XML tags", prompt)

    def test_choice_parser_is_strict_but_handles_common_wrappers(self) -> None:
        self.assertEqual(parse_choice("C"), "C")
        self.assertEqual(parse_choice("(b)."), "B")
        self.assertEqual(parse_choice("The answer is D."), "D")
        self.assertEqual(
            parse_choice("<reasoning>A is tempting, but B is correct.</reasoning><answer>B</answer>"),
            "B",
        )
        self.assertEqual(
            parse_choice("<answer>A</answer> revised <answer>D</answer>"),
            "D",
        )
        self.assertEqual(
            parse_choice("<reasoning>Concise analysis.\n<answer>C"),
            "C",
        )
        self.assertEqual(
            parse_choice("Reasoning: A looks plausible.\nFinal answer: C"),
            "C",
        )
        self.assertEqual(
            parse_choice("Final answer: A\nCorrection.\nFinal answer: B"),
            "B",
        )
        self.assertEqual(parse_choice("right", ["left", "right", "above", "below"]), "B")
        self.assertIsNone(parse_choice("I cannot tell."))

    def test_summary_reports_slices_and_invalid_predictions(self) -> None:
        rows = [
            {
                "question_id": "1",
                "relation": "left",
                "data_source": "mp3d",
                "predicted_label": "A",
                "correct": True,
            },
            {
                "question_id": "2",
                "relation": "left",
                "data_source": "ai2thor",
                "predicted_label": None,
                "correct": False,
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["valid_prediction_rate"], 0.5)
        self.assertEqual(result["by_relation"]["left"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
