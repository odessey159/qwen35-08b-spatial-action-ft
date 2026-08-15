from __future__ import annotations

import copy
import unittest

from exp0.generate_cot_data import (
    build_cot_sample,
    extract_task_relevant_state,
    parse_response_sections,
    validate_cot_sample,
)
from exp0.subgoal_abstraction import (
    abstract_subgoals,
    validate_subgoal_abstraction,
)


class CotGenerationTests(unittest.TestCase):
    def _sample(self) -> dict:
        return {
            "sample_id": "sample_1",
            "image": "image.png",
            "instruction": "把 Apple 放到 Table 上。",
            "gold": {
                "plan_actions": [
                    "GotoLocation(Fridge)",
                    "OpenObject(Fridge)",
                    "PickupObject(Apple)",
                    "GotoLocation(Table)",
                    "PutObject(Apple,Table)",
                ]
            },
            "scene_graph": {
                "objects": [
                    {"id": "Agent", "type": "Agent", "attributes": {"visible": True}},
                    {"id": "Apple", "type": "Apple", "attributes": {"visible": True}},
                    {
                        "id": "Fridge",
                        "type": "Fridge",
                        "attributes": {"visible": True, "is_open": False},
                    },
                    {"id": "Table", "type": "Table", "attributes": {"visible": True}},
                    {"id": "Vase", "type": "Vase", "attributes": {"visible": True}},
                ],
                "relations": [
                    {"subject": "Apple", "relation": "inside", "object": "Fridge"},
                    {"subject": "Table", "relation": "near", "object": "Fridge"},
                    {"subject": "Vase", "relation": "left_of", "object": "Agent"},
                ],
            },
            "meta": {
                "scene_id": "procthor_train_1",
                "sim_verified": True,
                "target_visible": True,
            },
        }

    def test_transfer_actions_become_high_level_subgoals(self) -> None:
        self.assertEqual(
            abstract_subgoals(self._sample()["gold"]["plan_actions"]),
            [
                "Acquire Apple.",
                "Transport Apple to Table.",
                "Place Apple at Table.",
            ],
        )

    def test_pickup_absorbs_navigation_and_container_access(self) -> None:
        self.assertEqual(
            abstract_subgoals(
                [
                    "GotoLocation(Fridge)",
                    "OpenObject(Fridge)",
                    "PickupObject(Apple)",
                ]
            ),
            ["Acquire Apple."],
        )

    def test_state_is_relevant_and_excludes_distractors(self) -> None:
        state = extract_task_relevant_state(self._sample())
        self.assertIn("Apple is inside Fridge.", state)
        self.assertIn("Fridge is closed.", state)
        self.assertNotIn("Vase is left of Agent.", state)

    def test_required_object_id_disambiguates_duplicate_types(self) -> None:
        sample = self._sample()
        sample["instruction"] = "打开 Drawer。"
        sample["gold"]["plan_actions"] = [
            "GotoLocation(Drawer)",
            "OpenObject(Drawer)",
        ]
        sample["scene_graph"] = {
            "objects": [
                {"id": "Agent", "type": "Agent", "attributes": {"visible": True}},
                {
                    "id": "Drawer|one",
                    "type": "Drawer",
                    "attributes": {"visible": True, "is_open": True},
                },
                {
                    "id": "Drawer|two",
                    "type": "Drawer",
                    "attributes": {"visible": True, "is_open": False},
                },
            ],
            "relations": [
                {"subject": "Drawer|two", "relation": "near", "object": "Agent"}
            ],
        }
        sample["meta"]["required_object_ids"] = ["Drawer|two"]
        state = extract_task_relevant_state(sample)
        self.assertIn("Drawer is closed.", state)
        self.assertNotIn("Drawer is open.", state)

    def test_built_sections_match_oracle_and_validator_rejects_stale_plan(self) -> None:
        row = build_cot_sample(self._sample())
        validate_cot_sample(row)
        sections = parse_response_sections(row["conversations"][1]["content"])
        self.assertEqual(sections["state"], row["oracle"]["state"])
        self.assertEqual(sections["plan"], row["oracle"]["plan"])
        self.assertEqual(sections["action"], row["oracle"]["actions"])

        stale = copy.deepcopy(row)
        stale["oracle"]["plan"][0] = "Acquire Cup."
        with self.assertRaisesRegex(ValueError, "plan differs from oracle"):
            validate_cot_sample(stale)

    def test_plan_action_object_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_subgoal_abstraction(
                ["GotoLocation(Apple)", "PickupObject(Apple)"],
                ["Acquire Cup."],
            )


if __name__ == "__main__":
    unittest.main()
