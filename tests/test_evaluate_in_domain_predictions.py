from __future__ import annotations

import unittest

from training.evaluate_in_domain_predictions import evaluate


def raw(sample_id: str, scene: str, group: str | None, state: str, actions: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "instruction": "ensure drawer open",
        "gold": {"plan_actions": actions},
        "meta": {
            "scene_id": scene,
            "counterfactual_group": group,
            "task_group": "counterfactual_put" if group else "open_close",
            "receptacle_state": state,
            "plan_length": len(actions),
            "sim_verified": True,
            "target_visible": True,
        },
    }


def prepared(state_fact: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "<image> task"},
            {"role": "assistant", "content": f"<state>\n{state_fact}\n</state>"},
        ]
    }


def prediction(sample_id: str, state_fact: str, actions: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "prediction": (
            f"<state>{state_fact}</state><plan>do it</plan>"
            f"<action>{'\n'.join(actions)}</action>"
        ),
    }


class EvaluateInDomainPredictionsTests(unittest.TestCase):
    def test_full_metrics_and_a_prime_delta(self) -> None:
        rows = [
            raw("t1", "train1", None, "open", ["GotoLocation(Drawer)"]),
            raw("t2", "train2", None, "closed", ["GotoLocation(Drawer)"]),
            raw("a", "val1", "cf1", "open", ["GotoLocation(Drawer)"]),
            raw(
                "b",
                "val1",
                "cf1",
                "closed",
                ["GotoLocation(Drawer)", "OpenObject(Drawer)"],
            ),
        ]
        manifest = {"validation_sample_ids": ["a", "b"]}
        prepared_rows = [prepared("Drawer is open."), prepared("Drawer is closed.")]
        correct = {
            "a": prediction("a", "Drawer is open.", ["GotoLocation(Drawer)"]),
            "b": prediction(
                "b",
                "Drawer is closed.",
                ["GotoLocation(Drawer)", "OpenObject(Drawer)"],
            ),
        }
        a_prime = {
            "a": prediction(
                "a",
                "Drawer is closed.",
                ["GotoLocation(Drawer)", "OpenObject(Drawer)"],
            ),
            "b": prediction("b", "Drawer is open.", ["GotoLocation(Drawer)"]),
        }
        result = evaluate(rows, prepared_rows, manifest, correct, a_prime)
        overall = result["correct_image"]["slices"]["all"]
        self.assertEqual(overall["action_sequence_exact"], 1.0)
        self.assertEqual(overall["state_fact_f1"], 1.0)
        self.assertEqual(result["correct_image"]["counterfactual_pairs"]["pair_exact"], 1.0)
        self.assertEqual(result["a_prime"]["counterfactual_pairs"]["pair_exact"], 0.0)
        self.assertEqual(
            result["visual_contribution"]["action_exact_a_minus_a_prime"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
