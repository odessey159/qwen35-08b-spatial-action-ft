from __future__ import annotations

import unittest

from training.compare_pre_post_metrics import _metric_values


class MetricValueTests(unittest.TestCase):
    def test_pair_exact_comes_from_pair_summary(self) -> None:
        overall = {
            "action_sequence_exact": 0.1,
            "step_position_match_recall": 0.2,
            "step_position_match_precision": 0.3,
            "state_fact_precision": 0.4,
            "state_fact_recall": 0.5,
            "state_fact_f1": 0.6,
            "state_exact": 0.7,
            "p_action_exact_given_state_exact": 0.8,
            "p_action_exact_given_state_wrong": 0.9,
            "strict_structure_valid": 1.0,
            "invalid_action_vocab_rate": 0.0,
            "invalid_object_vocab_rate": 0.0,
            "placeholder_copy_rate": 0.0,
            "text_oracle_action_exact": 0.55,
            "text_oracle_coverage": 0.95,
        }
        result = {
            "correct_image": {
                "slices": {"all": overall},
                "counterfactual_pairs": {"pair_exact": 0.75},
            }
        }

        values = _metric_values(result)

        self.assertEqual(values["action_sequence_exact"], 0.1)
        self.assertEqual(values["cf_pair_exact"], 0.75)
        self.assertEqual(values["text_oracle_coverage"], 0.95)


if __name__ == "__main__":
    unittest.main()
