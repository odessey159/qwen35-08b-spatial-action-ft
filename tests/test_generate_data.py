from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp0.generate_data import (  # noqa: E402
    SceneContext,
    TaskCandidate,
    ThorDataGenerator,
    ViewOption,
    apply_shard,
    iter_jsonl,
    parse_sample_index,
    signature_from_row,
    split_count,
)
from exp0.data_quality import (  # noqa: E402
    generation_quality_summary,
    instruction_plan_collisions,
)


def _config(output_dir: str = "out") -> dict:
    return {
        "seed": 42,
        "source": "ithor",
        "scene_names": ["FloorPlan1", "FloorPlan2"],
        "num_scenes": 2,
        "output_dir": output_dir,
        "render": {
            "width": 16,
            "height": 16,
            "field_of_view": 90,
            "visibility_distance": 5.0,
            "grid_size": 0.25,
            "rotate_step_degrees": 45,
        },
        "generation": {
            "max_views_per_scene": 1,
            "max_samples_per_scene": 12,
            "camera_horizons": [0],
            "camera_rotations": [0],
            "counterfactual_pairs": 0,
            "group_quotas": {
                "pickup": 2,
                "clean": 0,
                "heat": 0,
                "toggle": 0,
                "slice": 0,
                "open_close": 0,
            },
        },
        "spatial": {
            "axis_threshold_m": 0.15,
            "close_threshold_m": 1.5,
            "far_threshold_m": 3.0,
        },
    }


def _row(index: int, scene_id: str = "scene_a", group: str = "pickup") -> dict:
    sample_id = f"exp0_{index:04d}"
    return {
        "sample_id": sample_id,
        "image": f"images/{sample_id}.png",
        "wrong_image": "",
        "instruction": "拿起 Apple。",
        "gold": {
            "plan_actions": ["GotoLocation(Apple)", "PickupObject(Apple)"],
            "plan_nl": "前往 Apple，然后拿起 Apple。",
        },
        "scene_graph": {"objects": [], "relations": []},
        "spatial_facts": [],
        "subgoals": [],
        "meta": {
            "scene_id": scene_id,
            "target_id": "Apple|1",
            "counterfactual_group": None,
            "task_group": group,
            "target_visible": True,
            "receptacle_state": None,
            "spatial_relations": [],
            "plan_length": 2,
            "sim_verified": True,
            "camera_pose": {
                "position": {"x": 1.23, "y": 0.9, "z": 4.56},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                "horizon": 30.0,
            },
            "required_object_ids": ["Apple|1"],
        },
    }


def _generator(root: Path, overwrite: bool = False, config: dict | None = None) -> ThorDataGenerator:
    payload = config if config is not None else _config()
    config_path = root / "generator_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return ThorDataGenerator(payload, config_path, overwrite)


def _commit(generator: ThorDataGenerator, row: dict) -> None:
    generator._restore_row_state(row)
    (generator.output_dir / row["image"]).parent.mkdir(parents=True, exist_ok=True)
    (generator.output_dir / row["image"]).write_bytes(b"png")
    generator.commit_rows([row])


class GenerateDataHelpersTests(unittest.TestCase):
    def test_split_count_distributes_remainder(self) -> None:
        sizes = [split_count(10, index, 3) for index in range(3)]
        self.assertEqual(sizes, [4, 3, 3])

    def test_apply_shard_does_not_require_even_quotas(self) -> None:
        shards = []
        for index in range(3):
            config = _config()
            config["num_scenes"] = 10
            config["generation"]["counterfactual_pairs"] = 10
            config["generation"]["group_quotas"]["pickup"] = 7
            apply_shard(config, index, 3)
            shards.append(config)
        self.assertEqual([item["num_scenes"] for item in shards], [4, 3, 3])
        self.assertEqual(
            [item["generation"]["counterfactual_pairs"] for item in shards], [4, 3, 3]
        )
        self.assertEqual(
            [item["generation"]["group_quotas"]["pickup"] for item in shards], [3, 2, 2]
        )
        self.assertEqual(shards[0]["output_dir"], "shard_data/shard_0")

    def test_signature_roundtrip_uses_saved_target_id(self) -> None:
        row = _row(1)
        self.assertEqual(parse_sample_index(row["sample_id"]), 1)
        self.assertEqual(
            signature_from_row(row),
            (
                "scene_a",
                1.23,
                4.56,
                90,
                30,
                "pickup",
                "Apple|1",
                None,
            ),
        )


class GenerateDataPersistenceTests(unittest.TestCase):
    def test_commit_rows_appends_instead_of_replacing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            generator = _generator(root)
            _commit(generator, _row(1, "scene_a"))
            _commit(generator, _row(2, "scene_b"))
            rows = list(iter_jsonl(generator.samples_path))
            self.assertEqual([row["sample_id"] for row in rows], ["exp0_0001", "exp0_0002"])
            self.assertEqual(generator.committed_count, 2)

    def test_incomplete_finalize_keeps_partial_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            generator = _generator(root)
            _commit(generator, _row(1, "scene_a"))
            with self.assertRaisesRegex(RuntimeError, "Partial samples were saved"):
                generator.finalize(complete=False)
            rows = list(iter_jsonl(generator.samples_path))
            self.assertEqual(len(rows), 1)
            report = json.loads(generator.report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["complete"])
            self.assertEqual(report["sample_count"], 1)

    def test_complete_finalize_rejects_same_image_instruction_with_multiple_plans(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            generator = _generator(root)
            first = _row(1, "scene_a")
            second = _row(2, "scene_b")
            second["gold"]["plan_actions"] = [
                "GotoLocation(Apple)",
                "SliceObject(Apple)",
            ]
            _commit(generator, first)
            _commit(generator, second)

            with self.assertRaisesRegex(RuntimeError, "multiple gold plans"):
                generator.finalize(complete=True)

    def test_resume_restores_counts_signatures_and_skips_overwrite_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            first = _generator(root)
            _commit(first, _row(1, "scene_a"))
            first.write_checkpoint(complete=False)

            second = _generator(root, overwrite=False)
            second.load_existing()
            self.assertTrue(second.resumed)
            self.assertEqual(second.committed_count, 1)
            self.assertEqual(second.next_sample_index, 2)
            self.assertEqual(second.group_counts["pickup"], 1)
            self.assertIn(signature_from_row(_row(1, "scene_a")), second.used_signatures)

    def test_truncated_jsonl_tail_is_dropped_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            generator = _generator(root)
            _commit(generator, _row(1, "scene_a"))
            with generator.samples_path.open("a", encoding="utf-8") as handle:
                handle.write('{"sample_id": "exp0_0002", "partial"')
            resumed = _generator(root)
            resumed.load_existing()
            self.assertEqual(resumed.committed_count, 1)
            rows = list(iter_jsonl(resumed.samples_path))
            self.assertEqual([row["sample_id"] for row in rows], ["exp0_0001"])

    def test_unpaired_counterfactual_tail_is_dropped_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            generator = _generator(root)
            row = _row(1, "scene_a", group="counterfactual_put")
            row["meta"]["counterfactual_group"] = "cf_0001"
            row["meta"]["receptacle_state"] = "open"
            _commit(generator, row)
            resumed = _generator(root)
            resumed.load_existing()
            self.assertEqual(resumed.committed_count, 0)
            self.assertEqual(list(iter_jsonl(resumed.samples_path)), [])
            self.assertFalse((generator.images_dir / "exp0_0001.png").exists())

    def test_resume_deletes_orphan_images(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            generator = _generator(root)
            _commit(generator, _row(1, "scene_a"))
            orphan = generator.images_dir / "exp0_9999.png"
            orphan.write_bytes(b"png")
            resumed = _generator(root)
            resumed.load_existing()
            self.assertFalse(orphan.exists())
            self.assertTrue((generator.images_dir / "exp0_0001.png").exists())

    def test_overwrite_clears_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            first = _generator(root)
            _commit(first, _row(1, "scene_a"))
            self.assertTrue(first.samples_path.exists())
            second = _generator(root, overwrite=True)
            self.assertFalse(second.samples_path.exists())
            self.assertEqual(second.committed_count, 0)

    def test_wrong_images_assigned_without_keeping_scene_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            config = _config()
            config["generation"]["group_quotas"]["pickup"] = 2
            generator = _generator(root, config=config)
            _commit(generator, _row(1, "scene_a"))
            _commit(generator, _row(2, "scene_b"))
            generator.assign_wrong_images()
            rows = list(iter_jsonl(generator.samples_path))
            self.assertEqual(rows[0]["wrong_image"], "images/exp0_0002.png")
            self.assertEqual(rows[1]["wrong_image"], "images/exp0_0001.png")
            self.assertNotEqual(rows[0]["wrong_image"], rows[0]["image"])

    def test_apply_shard_offsets_are_disjoint(self) -> None:
        offsets = []
        counts = []
        for index in range(4):
            config = copy.deepcopy(_config())
            config["num_scenes"] = 9
            apply_shard(config, index, 4)
            offsets.append(config["scene_offset"])
            counts.append(config["num_scenes"])
        ranges = [
            set(range(offset, offset + count))
            for offset, count in zip(offsets, counts)
            if count
        ]
        for first_index, first in enumerate(ranges):
            for second in ranges[first_index + 1 :]:
                self.assertFalse(first & second)
        self.assertEqual(sum(counts), 9)


class _FakeEvent:
    def __init__(self, objects: list[dict]) -> None:
        self.metadata = {
            "objects": objects,
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
        }


def _visible_object(object_id: str, object_type: str, **kwargs: object) -> dict:
    row = {
        "objectId": object_id,
        "objectType": object_type,
        "visible": True,
        "position": {"x": 1.0, "y": 0.8, "z": 1.0},
        "distance": 1.0,
        "parentReceptacles": [],
    }
    row.update(kwargs)
    return row


def _dummy_candidate(group: str, target_id: str) -> TaskCandidate:
    return TaskCandidate(
        group=group,
        target_id=target_id,
        relevant_ids=(target_id,),
        plan_actions=("GotoLocation(Apple)",),
        instruction="拿起 Apple。",
        subgoals=("前往 Apple。",),
        verify=lambda: True,
    )


def _context() -> SceneContext:
    return SceneContext(
        house="FloorPlan1",
        scene_id="FloorPlan1",
        camera_pose={
            "position": {"x": 1.0, "y": 0.9, "z": 2.0},
            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            "horizon": 30.0,
        },
    )


class ViewSelectionTests(unittest.TestCase):
    def test_listing_candidates_is_pure_metadata_and_clean_defers_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            generator = _generator(Path(raw_root))
            event = _FakeEvent(
                [
                    _visible_object("Apple|1", "Apple", pickupable=True, isPickedUp=False),
                    _visible_object("Plate|1", "Plate", dirtyable=True, isDirty=False),
                    _visible_object(
                        "Potato|1", "Potato", cookable=True, isCooked=False, pickupable=True
                    ),
                ]
            )
            pickups = generator.list_pickup_candidates(event)
            cleans = generator.list_atomic_candidates(event, "clean")
            heats = generator.list_atomic_candidates(event, "heat")
            self.assertEqual({item.target_id for item in pickups}, {"Apple|1", "Potato|1"})
            self.assertEqual([item.target_id for item in cleans], ["Plate|1"])
            self.assertEqual([item.target_id for item in heats], ["Potato|1"])
            self.assertIsNone(pickups[0].setup)
            self.assertIsNotNone(cleans[0].setup)

    def test_atomic_candidate_navigates_to_visible_parent_and_annotates_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            generator = _generator(Path(raw_root))
            parent = _visible_object("DiningTable|1", "DiningTable", receptacle=True)
            target = _visible_object(
                "Plate|1",
                "Plate",
                dirtyable=True,
                isDirty=True,
                parentReceptacles=[parent["objectId"]],
            )
            candidate = generator.list_atomic_candidates(
                _FakeEvent([target, parent]), "clean"
            )[0]
            self.assertEqual(
                candidate.plan_actions,
                ("GotoLocation(DiningTable)", "CleanObject(Plate)"),
            )
            self.assertEqual(candidate.relevant_ids, ("Plate|1", "DiningTable|1"))

    def test_atomic_candidate_falls_back_to_target_without_visible_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            generator = _generator(Path(raw_root))
            target = _visible_object(
                "Fridge|1", "Fridge", openable=True, isOpen=False
            )
            candidate = generator.list_atomic_candidates(
                _FakeEvent([target]), "open_close"
            )[0]
            self.assertEqual(
                candidate.plan_actions,
                ("GotoLocation(Fridge)", "OpenObject(Fridge)"),
            )
            self.assertEqual(candidate.relevant_ids, ("Fridge|1",))

    def test_open_and_toggle_candidates_are_disjoint_and_use_distinct_verbs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            generator = _generator(Path(raw_root))
            event = _FakeEvent(
                [
                    _visible_object(
                        "Laptop|1",
                        "Laptop",
                        toggleable=True,
                        openable=True,
                        isToggled=False,
                        isOpen=False,
                    ),
                    _visible_object(
                        "CellPhone|1", "CellPhone", toggleable=True, isToggled=False
                    ),
                    _visible_object(
                        "Cabinet|1", "Cabinet", openable=True, isOpen=False
                    ),
                ]
            )
            toggles = generator.list_atomic_candidates(event, "toggle")
            open_close = generator.list_atomic_candidates(event, "open_close")
            self.assertEqual([item.target_id for item in toggles], ["CellPhone|1"])
            self.assertEqual(toggles[0].instruction, "开启 CellPhone。")
            self.assertEqual([item.target_id for item in open_close], ["Cabinet|1"])
            self.assertEqual(open_close[0].instruction, "打开 Cabinet。")

    def test_enumerate_skips_filled_groups_and_impossible_heat_views(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            config = _config()
            config["generation"]["group_quotas"] = {
                "pickup": 1,
                "clean": 0,
                "heat": 8,
                "toggle": 0,
                "slice": 0,
                "open_close": 0,
            }
            generator = _generator(Path(raw_root), config=config)
            generator.group_counts["pickup"] = 1
            living_room = _FakeEvent(
                [
                    _visible_object("Apple|1", "Apple", pickupable=True, isPickedUp=False),
                    _visible_object("Book|1", "Book", pickupable=True, isPickedUp=False),
                ]
            )
            options = generator.enumerate_view_options(
                _context(), living_room, allow_pairs=False
            )
            self.assertEqual(options, [])

            kitchen = _FakeEvent(
                [
                    _visible_object("Apple|1", "Apple", pickupable=True, isPickedUp=False),
                    _visible_object("Potato|1", "Potato", cookable=True, isCooked=False),
                ]
            )
            kitchen_options = generator.enumerate_view_options(
                _context(), kitchen, allow_pairs=False
            )
            self.assertEqual({option.group for option in kitchen_options}, {"heat"})
            self.assertEqual(kitchen_options[0].candidate.target_id, "Potato|1")

    def test_pick_view_option_weights_by_quota_deficit_not_candidate_count(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            config = _config()
            config["generation"]["group_quotas"]["pickup"] = 100
            config["generation"]["group_quotas"]["heat"] = 100
            generator = _generator(Path(raw_root), config=config)
            generator.group_counts["pickup"] = 90
            generator.group_counts["heat"] = 10
            options = [
                *[
                    ViewOption(
                        group="pickup",
                        kind="atomic",
                        candidate=_dummy_candidate("pickup", f"Apple|{index}"),
                    )
                    for index in range(10)
                ],
                ViewOption(
                    group="heat",
                    kind="atomic",
                    candidate=_dummy_candidate("heat", "Potato|1"),
                ),
            ]
            picks: dict[str, int] = {"pickup": 0, "heat": 0}
            for _ in range(2000):
                chosen = generator.pick_view_option(options)
                assert chosen is not None
                picks[chosen.group] += 1
            self.assertGreater(picks["heat"], picks["pickup"])
            self.assertGreater(picks["heat"], 1400)

    def test_enumerate_includes_put_pairs_only_when_deficit_and_space_remain(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            config = _config()
            config["generation"]["counterfactual_pairs"] = 1
            config["generation"]["group_quotas"]["pickup"] = 0
            generator = _generator(Path(raw_root), config=config)
            event = _FakeEvent(
                [
                    _visible_object("Apple|1", "Apple", pickupable=True, isPickedUp=False),
                    _visible_object(
                        "Fridge|1",
                        "Fridge",
                        receptacle=True,
                        openable=True,
                        pickupable=False,
                    ),
                ]
            )
            options = generator.enumerate_view_options(_context(), event, allow_pairs=True)
            self.assertTrue(any(option.kind == "put_pair" for option in options))
            self.assertTrue(any(option.kind == "open_pair" for option in options))
            blocked = generator.enumerate_view_options(_context(), event, allow_pairs=False)
            self.assertFalse(any(option.kind in {"put_pair", "open_pair"} for option in blocked))


class DataQualityTests(unittest.TestCase):
    def test_collision_key_includes_image_and_instruction(self) -> None:
        first = _row(1)
        second = copy.deepcopy(first)
        second["gold"]["plan_actions"] = ["GotoLocation(Apple)", "SliceObject(Apple)"]
        self.assertEqual(len(instruction_plan_collisions([first, second])), 1)

        second["image"] = "images/exp0_0002.png"
        self.assertEqual(instruction_plan_collisions([first, second]), {})

        first["meta"]["image_sha256"] = "same-pixels"
        second["meta"]["image_sha256"] = "same-pixels"
        self.assertEqual(len(instruction_plan_collisions([first, second])), 1)

    def test_generation_quality_summary_counts_actions_and_goto_schema(self) -> None:
        parent_row = _row(1)
        parent_row["gold"]["plan_actions"] = [
            "GotoLocation(DiningTable)",
            "PickupObject(Apple)",
        ]
        fallback_row = _row(2)
        summary = generation_quality_summary([parent_row, fallback_row])
        self.assertEqual(summary["instruction_collision"], 0)
        self.assertEqual(summary["action_distribution"], {"PickupObject": 2})
        self.assertEqual(
            summary["goto_schema"],
            {"parent_location": 1, "object_location": 1},
        )


if __name__ == "__main__":
    unittest.main()
