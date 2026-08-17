from __future__ import annotations

import unittest

from training.compare_counterfactual_predictions import (
    compare,
    mcnemar_exact_p,
    paired_difference,
    wilson_interval,
)


def raw_row(sample_id: str, group: str, state: str, actions: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "gold": {"plan_actions": actions},
        "meta": {
            "counterfactual_group": group,
            "receptacle_state": state,
        },
    }


def prediction(sample_id: str, actions: list[str]) -> dict:
    body = "\n".join(actions)
    return {
        "sample_id": sample_id,
        "prediction": f"<state>x</state><plan>x</plan><action>{body}</action>",
        "generated_tokens": 5,
        "seconds": 0.5,
    }


class CompareCounterfactualPredictionsTest(unittest.TestCase):
    def test_statistics_helpers(self) -> None:
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertAlmostEqual(mcnemar_exact_p(0, 2), 0.5)
        result = paired_difference([False, False, True], [True, True, True])
        self.assertEqual(result["candidate_only_correct"], 2)
        self.assertAlmostEqual(result["accuracy_delta"], 2 / 3)

    def test_compare_reports_paired_and_state_metrics(self) -> None:
        raw = [
            raw_row("a", "cf1", "open", ["GotoLocation(Drawer)"]),
            raw_row(
                "b",
                "cf1",
                "closed",
                ["GotoLocation(Drawer)", "OpenObject(Drawer)"],
            ),
            raw_row("c", "cf2", "open", ["GotoLocation(Fridge)"]),
            raw_row(
                "d",
                "cf2",
                "closed",
                ["GotoLocation(Fridge)", "OpenObject(Fridge)"],
            ),
        ]
        manifest = {"validation_sample_ids": ["a", "b", "c", "d"]}
        baseline = {
            "a": prediction("a", ["GotoLocation(Drawer)", "OpenObject(Drawer)"]),
            "b": prediction("b", ["GotoLocation(Drawer)", "OpenObject(Drawer)"]),
            "c": prediction("c", ["GotoLocation(Fridge)"]),
            "d": prediction("d", ["GotoLocation(Fridge)"]),
        }
        candidate = {
            "a": prediction("a", ["GotoLocation(Drawer)"]),
            "b": prediction("b", ["GotoLocation(Drawer)", "OpenObject(Drawer)"]),
            "c": prediction("c", ["GotoLocation(Fridge)"]),
            "d": prediction("d", ["GotoLocation(Fridge)", "OpenObject(Fridge)"]),
        }
        summary, rows = compare(
            raw,
            manifest,
            baseline,
            candidate,
            "step0",
            "step11193",
        )
        self.assertEqual(summary["scope"]["complete_counterfactual_pairs"], 2)
        self.assertEqual(summary["models"]["step0"]["sample_exact_accuracy"], 0.5)
        self.assertEqual(summary["models"]["step0"]["pair_exact_accuracy"], 0.0)
        self.assertEqual(summary["models"]["step11193"]["pair_exact_accuracy"], 1.0)
        self.assertEqual(
            summary["models"]["step0"]["pair_member_outcomes"][
                "one_member_correct"
            ],
            2,
        )
        self.assertEqual(
            summary["models"]["step11193"]["pair_member_outcomes"]["both_correct"],
            2,
        )
        self.assertEqual(
            summary["paired_comparison"]["pair_exact"]["candidate_only_correct"], 2
        )
        self.assertEqual({row["comparison"] for row in rows}, {"candidate_fixed"})


if __name__ == "__main__":
    unittest.main()
