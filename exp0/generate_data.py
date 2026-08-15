from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# Restricted to ALFRED-compatible classes used by the project. Objects that are
# not in these sets are retained as harmless scene distractors but never chosen
# as task arguments.
ALFRED_OBJECT_CLASSES = {
    "AlarmClock",
    "Apple",
    "BaseballBat",
    "BasketBall",
    "Book",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "Candle",
    "CD",
    "CellPhone",
    "Cloth",
    "CreditCard",
    "Cup",
    "DishSponge",
    "Egg",
    "Fork",
    "Glassbottle",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "Lettuce",
    "Mug",
    "Newspaper",
    "Pan",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "RemoteControl",
    "SaltShaker",
    "SoapBar",
    "SoapBottle",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "TeddyBear",
    "TennisRacket",
    "TissueBox",
    "ToiletPaper",
    "Tomato",
    "Towel",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
}

ALFRED_RECEPTACLE_CLASSES = {
    "ArmChair",
    "BathtubBasin",
    "Bed",
    "Bowl",
    "Box",
    "Cabinet",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "Cup",
    "Desk",
    "DiningTable",
    "Drawer",
    "Dresser",
    "Fridge",
    "GarbageCan",
    "Microwave",
    "Mug",
    "Ottoman",
    "Pan",
    "Plate",
    "Pot",
    "Safe",
    "Shelf",
    "SideTable",
    "SinkBasin",
}

ALLOWED_CLASSES = ALFRED_OBJECT_CLASSES | ALFRED_RECEPTACLE_CLASSES


@dataclass(frozen=True)
class SceneContext:
    house: dict[str, Any] | str
    scene_id: str
    camera_pose: dict[str, Any]


@dataclass(frozen=True)
class TaskCandidate:
    group: str
    target_id: str
    relevant_ids: tuple[str, ...]
    plan_actions: tuple[str, ...]
    instruction: str
    subgoals: tuple[str, ...]
    verify: Callable[[], bool]
    state_label: str | None = None
    counterfactual_group: str | None = None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def object_map(event: Any) -> dict[str, dict[str, Any]]:
    return {obj["objectId"]: obj for obj in event.metadata.get("objects", [])}


def visible_objects(event: Any) -> list[dict[str, Any]]:
    return [
        obj
        for obj in event.metadata.get("objects", [])
        if obj.get("visible") and obj.get("objectType") in ALLOWED_CLASSES
    ]


def object_type(obj: dict[str, Any]) -> str:
    return str(obj["objectType"])


def canonical_action(name: str, *arguments: str) -> str:
    return f"{name}({','.join(arguments)})"


def action_to_nl(action: str) -> str:
    name, arguments = action.split("(", 1)
    args = arguments[:-1].split(",") if arguments[:-1] else []
    templates = {
        "GotoLocation": lambda x: f"前往 {x[0]}",
        "PickupObject": lambda x: f"拿起 {x[0]}",
        "PutObject": lambda x: f"把 {x[0]} 放入或放到 {x[1]}",
        "SliceObject": lambda x: f"切开 {x[0]}",
        "CleanObject": lambda x: f"清洁 {x[0]}",
        "HeatObject": lambda x: f"加热 {x[0]}",
        "ToggleObject": lambda x: f"切换 {x[0]} 的开关状态",
        "OpenObject": lambda x: f"打开 {x[0]}",
        "CloseObject": lambda x: f"关闭 {x[0]}",
    }
    return templates[name](args)


def plan_to_nl(actions: tuple[str, ...]) -> str:
    return "，然后".join(action_to_nl(action) for action in actions) + "。"


def agent_local_position(
    position: dict[str, float], agent: dict[str, Any]
) -> tuple[float, float, float]:
    dx = float(position["x"]) - float(agent["position"]["x"])
    dy = float(position["y"]) - float(agent["position"]["y"])
    dz = float(position["z"]) - float(agent["position"]["z"])
    yaw = math.radians(float(agent["rotation"]["y"]))
    right = dx * math.cos(yaw) - dz * math.sin(yaw)
    forward = dx * math.sin(yaw) + dz * math.cos(yaw)
    return right, dy, forward


def qualitative_relations(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    axis_threshold: float,
    close_threshold: float,
    far_threshold: float,
) -> list[str]:
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    dz = first[2] - second[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    result: list[str] = []
    if dx > axis_threshold:
        result.append("right_of")
    elif dx < -axis_threshold:
        result.append("left_of")
    if dy > axis_threshold:
        result.append("above")
    elif dy < -axis_threshold:
        result.append("below")
    if distance <= close_threshold:
        result.append("close")
    elif distance >= far_threshold:
        result.append("far")
    return result


def relation_to_chinese(subject: str, relation: str, target: str) -> str:
    labels = {
        "right_of": "位于右侧",
        "left_of": "位于左侧",
        "above": "位于上方",
        "below": "位于下方",
        "close": "靠近",
        "far": "远离",
        "in": "位于里面",
        "on": "位于上面",
    }
    return f"{subject} {labels.get(relation, relation)} {target}。"


class ThorDataGenerator:
    def __init__(self, config: dict[str, Any], config_path: Path, overwrite: bool):
        self.config = config
        self.config_path = config_path
        self.rng = random.Random(int(config["seed"]))
        self.output_dir = (config_path.parent / config["output_dir"]).resolve()
        self.images_dir = self.output_dir / "images"
        self.samples_path = self.output_dir / "samples.jsonl"
        self.report_path = self.output_dir / "generation_report.json"
        self.controller: Any = None
        self.rows: list[dict[str, Any]] = []
        self.rejections: Counter[str] = Counter()
        self.group_counts: Counter[str] = Counter()
        self.scene_counts: Counter[str] = Counter()
        self.used_signatures: set[tuple[Any, ...]] = set()
        self.next_sample_index = 1

        if overwrite:
            if self.images_dir.exists():
                shutil.rmtree(self.images_dir)
            for path in (self.samples_path, self.report_path):
                if path.exists():
                    path.unlink()
        elif self.samples_path.exists() or self.images_dir.exists():
            raise FileExistsError(
                f"Generated data already exists under {self.output_dir}; use --overwrite explicitly"
            )
        self.images_dir.mkdir(parents=True, exist_ok=True)

    @property
    def render_config(self) -> dict[str, Any]:
        return self.config["render"]

    @property
    def generation_config(self) -> dict[str, Any]:
        return self.config["generation"]

    @property
    def spatial_config(self) -> dict[str, Any]:
        return self.config["spatial"]

    def load_scenes(self) -> list[tuple[dict[str, Any] | str, str]]:
        source = self.config.get("source", "procthor")
        count = int(self.config["num_scenes"])
        offset = int(self.config.get("scene_offset", 0))
        if offset < 0:
            raise ValueError("scene_offset must be non-negative")
        if source == "ithor":
            configured_scenes = self.config.get("scene_names")
            if configured_scenes:
                selected = list(configured_scenes)[offset : offset + count]
                return [(str(scene), str(scene)) for scene in selected]
            all_scenes = [
                *(f"FloorPlan{i}" for i in range(1, 31)),
                *(f"FloorPlan{i}" for i in range(201, 231)),
                *(f"FloorPlan{i}" for i in range(301, 331)),
                *(f"FloorPlan{i}" for i in range(401, 431)),
            ]
            self.rng.shuffle(all_scenes)
            selected = all_scenes[offset : offset + count]
            return [(scene, scene) for scene in selected]
        if source != "procthor":
            raise ValueError(f"Unknown scene source: {source}")

        import prior

        dataset = prior.load_dataset(
            "procthor-10k",
            revision=str(self.config.get("revision", "main")),
        )
        split_name = str(self.config.get("split", "train"))
        split = dataset[split_name]
        configured_indices = self.config.get("scene_indices")
        if configured_indices is not None:
            indices = [int(index) for index in configured_indices]
            invalid = [index for index in indices if not 0 <= index < len(split)]
            if invalid:
                raise ValueError(f"scene_indices out of range for {split_name}: {invalid}")
            selected_indices = indices[offset : offset + count]
            return [
                (split[index], f"procthor_{split_name}_{index:05d}")
                for index in selected_indices
            ]
        indices = list(range(len(split)))
        self.rng.shuffle(indices)
        scenes: list[tuple[dict[str, Any], str]] = []
        for index in indices[offset : offset + count]:
            house = split[index]
            scene_id = f"procthor_{split_name}_{index:05d}"
            scenes.append((house, scene_id))
        return scenes

    def start_controller(self, initial_scene: dict[str, Any] | str) -> None:
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering, Linux64

        render = self.render_config
        platform_name = os.environ.get(
            "AI2THOR_PLATFORM", str(render.get("platform", "CloudRendering"))
        )
        platforms = {"CloudRendering": CloudRendering, "Linux64": Linux64}
        if platform_name not in platforms:
            raise ValueError(
                f"Unknown AI2THOR platform {platform_name!r}; expected one of {sorted(platforms)}"
            )
        self.controller = Controller(
            scene=initial_scene,
            platform=platforms[platform_name],
            agentMode="default",
            width=int(render["width"]),
            height=int(render["height"]),
            fieldOfView=float(render["field_of_view"]),
            visibilityDistance=float(render["visibility_distance"]),
            gridSize=float(render["grid_size"]),
            rotateStepDegrees=int(render["rotate_step_degrees"]),
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            snapToGrid=False,
        )

    def reset_scene(self, scene: dict[str, Any] | str) -> bool:
        try:
            event = self.controller.reset(scene=scene)
        except Exception:
            self.rejections["scene_reset_exception"] += 1
            return False
        if not event.metadata.get("lastActionSuccess", True):
            self.rejections["scene_reset_failed"] += 1
            return False
        return True

    def teleport_camera(self, pose: dict[str, Any]) -> Any | None:
        event = self.controller.step(
            action="TeleportFull",
            position=pose["position"],
            rotation=pose["rotation"],
            horizon=pose["horizon"],
            standing=True,
        )
        if not event.metadata.get("lastActionSuccess"):
            self.rejections["camera_teleport_failed"] += 1
            return None
        return event

    def get_reachable_positions(self) -> list[dict[str, float]]:
        event = self.controller.step(action="GetReachablePositions")
        if not event.metadata.get("lastActionSuccess"):
            return []
        return list(event.metadata.get("actionReturn") or [])

    def goto_object(self, object_id: str) -> bool:
        event = self.controller.step(
            action="GetInteractablePoses",
            objectId=object_id,
            rotations=self.generation_config["camera_rotations"],
            horizons=self.generation_config["camera_horizons"],
            standings=[True],
        )
        poses = list(event.metadata.get("actionReturn") or [])
        if not event.metadata.get("lastActionSuccess") or not poses:
            return False
        self.rng.shuffle(poses)
        for pose in poses[:10]:
            teleported = self.controller.step(action="TeleportFull", **pose)
            if teleported.metadata.get("lastActionSuccess"):
                return True
        return False

    def execute_object_action(self, action: str, object_id: str, **kwargs: Any) -> bool:
        if not self.goto_object(object_id):
            return False
        event = self.controller.step(
            action=action,
            objectId=object_id,
            forceAction=False,
            **kwargs,
        )
        return bool(event.metadata.get("lastActionSuccess"))

    def set_object_state(self, action: str, object_id: str) -> bool:
        desired_states = {
            "OpenObject": ("isOpen", True),
            "CloseObject": ("isOpen", False),
            "DirtyObject": ("isDirty", True),
            "CleanObject": ("isDirty", False),
            "ToggleObjectOn": ("isToggled", True),
            "ToggleObjectOff": ("isToggled", False),
        }
        current = object_map(self.controller.last_event).get(object_id)
        desired = desired_states.get(action)
        if current is not None and desired is not None:
            state_key, state_value = desired
            if current.get(state_key) is state_value:
                return True
        event = self.controller.step(
            action=action,
            objectId=object_id,
            forceAction=True,
        )
        return bool(event.metadata.get("lastActionSuccess"))

    def rollback_saved_rows(
        self,
        rows: list[dict[str, Any]],
        next_sample_index: int,
        group_counts: Counter[str],
        scene_counts: Counter[str],
        used_signatures: set[tuple[Any, ...]],
    ) -> None:
        for row in rows:
            image_path = self.output_dir / row["image"]
            if image_path.exists():
                image_path.unlink()
        self.next_sample_index = next_sample_index
        self.group_counts = group_counts
        self.scene_counts = scene_counts
        self.used_signatures = used_signatures

    def visible_parent(
        self, target: dict[str, Any], visible_by_id: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        for parent_id in target.get("parentReceptacles") or []:
            parent = visible_by_id.get(parent_id)
            if parent is not None and parent.get("objectType") in ALFRED_RECEPTACLE_CLASSES:
                return parent
        return None

    def build_scene_annotations(
        self,
        event: Any,
        relevant_ids: tuple[str, ...],
    ) -> tuple[dict[str, Any], list[str], list[list[str]]]:
        visible = visible_objects(event)
        counts = Counter(object_type(obj) for obj in visible)
        type_indices: Counter[str] = Counter()
        aliases: dict[str, str] = {}
        for obj in sorted(visible, key=lambda item: (object_type(item), float(item.get("distance", 0)))):
            obj_type = object_type(obj)
            type_indices[obj_type] += 1
            aliases[obj["objectId"]] = (
                obj_type if counts[obj_type] == 1 else f"{obj_type}_{type_indices[obj_type]}"
            )

        objects: list[dict[str, Any]] = [
            {"id": "Agent", "type": "Agent", "attributes": {"visible": True}}
        ]
        for obj in visible:
            attributes: dict[str, Any] = {"visible": True}
            property_state_pairs = (
                ("openable", "isOpen", "is_open"),
                ("toggleable", "isToggled", "is_toggled"),
                ("dirtyable", "isDirty", "is_dirty"),
                ("cookable", "isCooked", "is_cooked"),
                ("sliceable", "isSliced", "is_sliced"),
                ("pickupable", "isPickedUp", "is_picked_up"),
            )
            for capability, state_key, output_key in property_state_pairs:
                if obj.get(capability):
                    attributes[output_key] = bool(obj.get(state_key))
            objects.append(
                {
                    "id": aliases[obj["objectId"]],
                    "type": object_type(obj),
                    "attributes": attributes,
                }
            )

        agent = event.metadata["agent"]
        relevant = [obj for obj in visible if obj["objectId"] in relevant_ids]
        local_positions = {
            obj["objectId"]: agent_local_position(obj["position"], agent) for obj in relevant
        }
        spatial = self.spatial_config
        relations: list[dict[str, str]] = []

        agent_origin = (0.0, 0.0, 0.0)
        for obj in relevant:
            alias = aliases[obj["objectId"]]
            for relation in qualitative_relations(
                local_positions[obj["objectId"]],
                agent_origin,
                float(spatial["axis_threshold_m"]),
                float(spatial["close_threshold_m"]),
                float(spatial["far_threshold_m"]),
            ):
                relations.append({"subject": alias, "relation": relation, "object": "Agent"})

        for first_index, first in enumerate(relevant):
            for second in relevant[first_index + 1 :]:
                for relation in qualitative_relations(
                    local_positions[first["objectId"]],
                    local_positions[second["objectId"]],
                    float(spatial["axis_threshold_m"]),
                    float(spatial["close_threshold_m"]),
                    float(spatial["far_threshold_m"]),
                ):
                    relations.append(
                        {
                            "subject": aliases[first["objectId"]],
                            "relation": relation,
                            "object": aliases[second["objectId"]],
                        }
                    )

        for obj in relevant:
            for parent_id in obj.get("parentReceptacles") or []:
                if parent_id not in aliases:
                    continue
                parent = next(item for item in visible if item["objectId"] == parent_id)
                relation = "in" if parent.get("openable") else "on"
                relations.append(
                    {
                        "subject": aliases[obj["objectId"]],
                        "relation": relation,
                        "object": aliases[parent_id],
                    }
                )

        deduplicated: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation in relations:
            key = (relation["subject"], relation["relation"], relation["object"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(relation)

        facts = [
            relation_to_chinese(row["subject"], row["relation"], row["object"])
            for row in deduplicated
        ]
        for obj in relevant:
            alias = aliases[obj["objectId"]]
            if obj.get("openable"):
                facts.append(f"{alias} 当前{'打开' if obj.get('isOpen') else '关闭'}。")
            if obj.get("toggleable"):
                facts.append(f"{alias} 当前{'开启' if obj.get('isToggled') else '关闭'}。")
            if obj.get("dirtyable"):
                facts.append(f"{alias} 当前{'脏' if obj.get('isDirty') else '干净'}。")
            if obj.get("cookable"):
                facts.append(f"{alias} 当前{'已加热' if obj.get('isCooked') else '未加热'}。")
            if obj.get("sliceable"):
                facts.append(f"{alias} 当前{'已切开' if obj.get('isSliced') else '未切开'}。")

        relation_rows = [
            [row["subject"], row["relation"], row["object"]] for row in deduplicated
        ]
        return {"objects": objects, "relations": deduplicated}, facts, relation_rows

    def save_sample(
        self,
        context: SceneContext,
        event: Any,
        candidate: TaskCandidate,
        image_suffix: str = "",
    ) -> dict[str, Any] | None:
        visible_ids = {obj["objectId"] for obj in visible_objects(event)}
        if not set(candidate.relevant_ids).issubset(visible_ids):
            self.rejections["required_object_not_visible"] += 1
            return None

        signature = (
            context.scene_id,
            round(float(context.camera_pose["position"]["x"]), 2),
            round(float(context.camera_pose["position"]["z"]), 2),
            int(context.camera_pose["rotation"]["y"]),
            int(context.camera_pose["horizon"]),
            candidate.group,
            candidate.target_id,
            candidate.state_label,
        )
        if signature in self.used_signatures:
            self.rejections["duplicate_signature"] += 1
            return None

        if not candidate.verify():
            self.rejections[f"simulation_failed:{candidate.group}"] += 1
            return None

        from PIL import Image

        sample_id = f"exp0_{self.next_sample_index:04d}"
        self.next_sample_index += 1
        image_name = f"{sample_id}{image_suffix}.png"
        Image.fromarray(event.frame).save(self.images_dir / image_name)
        scene_graph, facts, relation_rows = self.build_scene_annotations(
            event, candidate.relevant_ids
        )
        target_obj = object_map(event)[candidate.target_id]
        row = {
            "sample_id": sample_id,
            "image": f"images/{image_name}",
            "wrong_image": "",
            "instruction": candidate.instruction,
            "gold": {
                "plan_actions": list(candidate.plan_actions),
                "plan_nl": plan_to_nl(candidate.plan_actions),
            },
            "scene_graph": scene_graph,
            "spatial_facts": facts,
            "subgoals": list(candidate.subgoals),
            "meta": {
                "scene_id": context.scene_id,
                "counterfactual_group": candidate.counterfactual_group,
                "task_group": candidate.group,
                "target_visible": bool(target_obj.get("visible")),
                "receptacle_state": candidate.state_label,
                "spatial_relations": relation_rows,
                "plan_length": len(candidate.plan_actions),
                "sim_verified": True,
                "camera_pose": context.camera_pose,
                "required_object_ids": list(candidate.relevant_ids),
            },
        }
        self.used_signatures.add(signature)
        self.group_counts[candidate.group] += 1
        self.scene_counts[context.scene_id] += 1
        return row

    def _atomic_candidate(
        self,
        group: str,
        target: dict[str, Any],
        simulator_action: str,
        instruction: str,
        high_level_action: str,
    ) -> TaskCandidate:
        target_type = object_type(target)
        actions = (
            canonical_action("GotoLocation", target_type),
            canonical_action(high_level_action, target_type),
        )

        def verify() -> bool:
            return self.execute_object_action(simulator_action, target["objectId"])

        return TaskCandidate(
            group=group,
            target_id=target["objectId"],
            relevant_ids=(target["objectId"],),
            plan_actions=actions,
            instruction=instruction,
            subgoals=tuple(action_to_nl(action) + "。" for action in actions),
            verify=verify,
        )

    def make_pickup_candidate(self, event: Any) -> TaskCandidate | None:
        visible = visible_objects(event)
        by_id = {obj["objectId"]: obj for obj in visible}
        targets = [obj for obj in visible if obj.get("pickupable") and not obj.get("isPickedUp")]
        self.rng.shuffle(targets)
        if not targets:
            return None
        target = targets[0]
        parent = self.visible_parent(target, by_id)
        location = object_type(parent) if parent else object_type(target)
        target_type = object_type(target)
        actions = (
            canonical_action("GotoLocation", location),
            canonical_action("PickupObject", target_type),
        )

        def verify() -> bool:
            return self.execute_object_action("PickupObject", target["objectId"])

        relevant = (target["objectId"],) + ((parent["objectId"],) if parent else ())
        location_text = f"{location} 上或里面的 " if parent else ""
        return TaskCandidate(
            group="pickup",
            target_id=target["objectId"],
            relevant_ids=relevant,
            plan_actions=actions,
            instruction=f"拿起{location_text}{target_type}。",
            subgoals=tuple(action_to_nl(action) + "。" for action in actions),
            verify=verify,
        )

    def make_atomic_candidate(self, event: Any, group: str) -> TaskCandidate | None:
        visible = visible_objects(event)
        if group == "clean":
            targets = [obj for obj in visible if obj.get("dirtyable")]
            self.rng.shuffle(targets)
            for target in targets:
                if not self.set_object_state("DirtyObject", target["objectId"]):
                    continue
                return self._atomic_candidate(
                    group,
                    target,
                    "CleanObject",
                    f"清洁 {object_type(target)}。",
                    "CleanObject",
                )
        elif group == "heat":
            targets = [obj for obj in visible if obj.get("cookable") and not obj.get("isCooked")]
            self.rng.shuffle(targets)
            if targets:
                target = targets[0]
                return self._atomic_candidate(
                    group,
                    target,
                    "CookObject",
                    f"加热 {object_type(target)}。",
                    "HeatObject",
                )
        elif group == "slice":
            targets = [obj for obj in visible if obj.get("sliceable") and not obj.get("isSliced")]
            self.rng.shuffle(targets)
            if targets:
                target = targets[0]
                return self._atomic_candidate(
                    group,
                    target,
                    "SliceObject",
                    f"切开 {object_type(target)}。",
                    "SliceObject",
                )
        elif group == "toggle":
            targets = [obj for obj in visible if obj.get("toggleable")]
            self.rng.shuffle(targets)
            if targets:
                target = targets[0]
                simulator_action = "ToggleObjectOff" if target.get("isToggled") else "ToggleObjectOn"
                desired = "关闭" if target.get("isToggled") else "打开"
                return self._atomic_candidate(
                    group,
                    target,
                    simulator_action,
                    f"{desired} {object_type(target)}。",
                    "ToggleObject",
                )
        elif group == "open_close":
            targets = [obj for obj in visible if obj.get("openable")]
            self.rng.shuffle(targets)
            if targets:
                target = targets[0]
                desired_open = not bool(target.get("isOpen"))
                simulator_action = "OpenObject" if desired_open else "CloseObject"
                high_level = simulator_action
                desired = "打开" if desired_open else "关闭"
                return self._atomic_candidate(
                    group,
                    target,
                    simulator_action,
                    f"{desired} {object_type(target)}。",
                    high_level,
                )
        return None

    def choose_put_pair(
        self, event: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None] | None:
        visible = visible_objects(event)
        by_id = {obj["objectId"]: obj for obj in visible}
        targets = [obj for obj in visible if obj.get("pickupable") and not obj.get("isPickedUp")]
        destinations = [
            obj
            for obj in visible
            if obj.get("receptacle") and obj.get("openable") and not obj.get("pickupable")
        ]
        candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = []
        for target in targets:
            parents = set(target.get("parentReceptacles") or [])
            source = self.visible_parent(target, by_id)
            for destination in destinations:
                if destination["objectId"] not in parents and destination["objectId"] != target["objectId"]:
                    candidates.append((target, destination, source))
        self.rng.shuffle(candidates)
        return candidates[0] if candidates else None

    def configure_put_variant(
        self,
        context: SceneContext,
        target_id: str,
        destination_id: str,
        should_be_open: bool,
    ) -> Any | None:
        if not self.reset_scene(context.house):
            return None
        state_action = "OpenObject" if should_be_open else "CloseObject"
        if not self.set_object_state(state_action, destination_id):
            return None
        event = self.teleport_camera(context.camera_pose)
        if event is None:
            return None
        visible_ids = {obj["objectId"] for obj in visible_objects(event)}
        if target_id not in visible_ids or destination_id not in visible_ids:
            return None
        return event

    def make_put_candidate(
        self,
        event: Any,
        context: SceneContext,
        target: dict[str, Any],
        destination: dict[str, Any],
        source: dict[str, Any] | None,
        should_be_open: bool,
        counterfactual_group: str,
    ) -> TaskCandidate:
        target_type = object_type(target)
        destination_type = object_type(destination)
        source_type = object_type(source) if source else target_type
        actions = [
            canonical_action("GotoLocation", source_type),
            canonical_action("PickupObject", target_type),
            canonical_action("GotoLocation", destination_type),
        ]
        if not should_be_open:
            actions.append(canonical_action("OpenObject", destination_type))
        actions.append(canonical_action("PutObject", target_type, destination_type))
        if not should_be_open:
            actions.append(canonical_action("CloseObject", destination_type))
        action_tuple = tuple(actions)

        def verify() -> bool:
            if not self.goto_object(target["objectId"]):
                return False
            pickup = self.controller.step(
                action="PickupObject", objectId=target["objectId"], forceAction=False
            )
            if not pickup.metadata.get("lastActionSuccess"):
                return False
            if not self.goto_object(destination["objectId"]):
                return False
            if not should_be_open:
                opened = self.controller.step(
                    action="OpenObject", objectId=destination["objectId"], forceAction=False
                )
                if not opened.metadata.get("lastActionSuccess"):
                    return False
            placed = self.controller.step(
                action="PutObject",
                objectId=destination["objectId"],
                forceAction=False,
                placeStationary=True,
            )
            if not placed.metadata.get("lastActionSuccess"):
                return False
            if not should_be_open:
                closed = self.controller.step(
                    action="CloseObject", objectId=destination["objectId"], forceAction=False
                )
                if not closed.metadata.get("lastActionSuccess"):
                    return False
            final_objects = object_map(self.controller.last_event)
            final_destination = final_objects.get(destination["objectId"], {})
            contained = target["objectId"] in (final_destination.get("receptacleObjectIds") or [])
            return bool(contained)

        relevant = [target["objectId"], destination["objectId"]]
        if source is not None:
            relevant.append(source["objectId"])
        return TaskCandidate(
            group="counterfactual_put",
            target_id=target["objectId"],
            relevant_ids=tuple(dict.fromkeys(relevant)),
            plan_actions=action_tuple,
            instruction=f"把 {target_type} 放进 {destination_type}。",
            subgoals=tuple(action_to_nl(action) + "。" for action in action_tuple),
            verify=verify,
            state_label="open" if should_be_open else "closed",
            counterfactual_group=counterfactual_group,
        )

    def configure_open_state_variant(
        self,
        context: SceneContext,
        target_id: str,
        should_be_open: bool,
    ) -> Any | None:
        if not self.reset_scene(context.house):
            return None
        state_action = "OpenObject" if should_be_open else "CloseObject"
        if not self.set_object_state(state_action, target_id):
            return None
        event = self.teleport_camera(context.camera_pose)
        if event is None:
            return None
        if target_id not in {obj["objectId"] for obj in visible_objects(event)}:
            return None
        return event

    def make_open_state_candidate(
        self,
        target: dict[str, Any],
        should_be_open: bool,
        counterfactual_group: str,
    ) -> TaskCandidate:
        target_type = object_type(target)
        actions = [canonical_action("GotoLocation", target_type)]
        if not should_be_open:
            actions.append(canonical_action("OpenObject", target_type))
        action_tuple = tuple(actions)

        def verify() -> bool:
            if not self.goto_object(target["objectId"]):
                return False
            if not should_be_open:
                opened = self.controller.step(
                    action="OpenObject", objectId=target["objectId"], forceAction=False
                )
                if not opened.metadata.get("lastActionSuccess"):
                    return False
            final_target = object_map(self.controller.last_event).get(target["objectId"], {})
            return bool(final_target.get("isOpen"))

        return TaskCandidate(
            group="counterfactual_put",
            target_id=target["objectId"],
            relevant_ids=(target["objectId"],),
            plan_actions=action_tuple,
            instruction=f"确保 {target_type} 是打开的。",
            subgoals=tuple(action_to_nl(action) + "。" for action in action_tuple),
            verify=verify,
            state_label="open" if should_be_open else "closed",
            counterfactual_group=counterfactual_group,
        )

    def try_open_state_pair(
        self, context: SceneContext, initial_event: Any
    ) -> list[dict[str, Any]] | None:
        targets = [
            obj
            for obj in visible_objects(initial_event)
            if obj.get("openable") and not obj.get("pickupable")
        ]
        self.rng.shuffle(targets)
        if not targets:
            return None
        target = targets[0]
        group_id = f"cf_{1 + self.group_counts['counterfactual_put'] // 2:04d}"
        pair_rows: list[dict[str, Any]] = []
        transaction = (
            self.next_sample_index,
            self.group_counts.copy(),
            self.scene_counts.copy(),
            set(self.used_signatures),
        )
        for should_be_open in (True, False):
            event = self.configure_open_state_variant(
                context, target["objectId"], should_be_open
            )
            if event is None:
                self.rollback_saved_rows(pair_rows, *transaction)
                return None
            current_target = object_map(event).get(target["objectId"])
            if current_target is None:
                self.rollback_saved_rows(pair_rows, *transaction)
                return None
            candidate = self.make_open_state_candidate(
                current_target, should_be_open, group_id
            )
            row = self.save_sample(
                context,
                event,
                candidate,
                image_suffix="_open" if should_be_open else "_closed",
            )
            if row is None:
                self.rollback_saved_rows(pair_rows, *transaction)
                return None
            pair_rows.append(row)
        return pair_rows

    def try_counterfactual_pair(
        self, context: SceneContext, initial_event: Any
    ) -> list[dict[str, Any]] | None:
        selected = self.choose_put_pair(initial_event)
        if selected is None:
            return None
        target, destination, source = selected
        group_id = f"cf_{1 + self.group_counts['counterfactual_put'] // 2:04d}"
        pair_rows: list[dict[str, Any]] = []
        transaction = (
            self.next_sample_index,
            self.group_counts.copy(),
            self.scene_counts.copy(),
            set(self.used_signatures),
        )
        for should_be_open in (True, False):
            event = self.configure_put_variant(
                context,
                target["objectId"],
                destination["objectId"],
                should_be_open,
            )
            if event is None:
                self.rollback_saved_rows(pair_rows, *transaction)
                return None
            current_objects = object_map(event)
            current_target = current_objects.get(target["objectId"])
            current_destination = current_objects.get(destination["objectId"])
            current_source = current_objects.get(source["objectId"]) if source else None
            if current_target is None or current_destination is None:
                self.rollback_saved_rows(pair_rows, *transaction)
                return None
            candidate = self.make_put_candidate(
                event,
                context,
                current_target,
                current_destination,
                current_source,
                should_be_open,
                group_id,
            )
            row = self.save_sample(
                context,
                event,
                candidate,
                image_suffix="_open" if should_be_open else "_closed",
            )
            if row is None:
                self.rollback_saved_rows(pair_rows, *transaction)
                return None
            pair_rows.append(row)
        return pair_rows

    def remaining_groups(self) -> list[str]:
        quotas = self.generation_config["group_quotas"]
        groups = [
            group for group, quota in quotas.items() if self.group_counts[group] < int(quota)
        ]
        self.rng.shuffle(groups)
        return groups

    def non_pair_complete(self) -> bool:
        return not self.remaining_groups()

    def pair_complete(self) -> bool:
        required_rows = 2 * int(self.generation_config["counterfactual_pairs"])
        return self.group_counts["counterfactual_put"] >= required_rows

    def target_sample_count(self) -> int:
        non_pair = sum(int(value) for value in self.generation_config["group_quotas"].values())
        return non_pair + 2 * int(self.generation_config["counterfactual_pairs"])

    def assign_wrong_images(self) -> None:
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            by_scene[row["meta"]["scene_id"]].append(row)
        scene_ids = sorted(by_scene)
        if len(scene_ids) < 2:
            raise RuntimeError("A′ requires generated samples from at least two distinct scenes")
        for row in self.rows:
            own_scene = row["meta"]["scene_id"]
            other_scenes = [scene_id for scene_id in scene_ids if scene_id != own_scene]
            wrong_scene = self.rng.choice(other_scenes)
            wrong_row = self.rng.choice(by_scene[wrong_scene])
            row["wrong_image"] = wrong_row["image"]

    def run(self) -> None:
        scenes = self.load_scenes()
        if not scenes:
            raise RuntimeError("No scenes were loaded")
        initial_scene = scenes[0][0]
        bootstrap_index = self.config.get("bootstrap_procthor_index")
        if bootstrap_index is not None:
            import prior

            bootstrap_dataset = prior.load_dataset(
                "procthor-10k",
                revision=str(self.config.get("revision", "main")),
            )
            bootstrap_split = str(self.config.get("bootstrap_split", "train"))
            initial_scene = bootstrap_dataset[bootstrap_split][int(bootstrap_index)]
            print(
                f"Bootstrapping controller with ProcTHOR {bootstrap_split}[{bootstrap_index}]",
                flush=True,
            )
        self.start_controller(initial_scene)
        print("Controller ready", flush=True)
        generation = self.generation_config
        try:
            for house, scene_id in scenes:
                if self.pair_complete() and self.non_pair_complete():
                    break
                print(f"Loading scene={scene_id}", flush=True)
                if not self.reset_scene(house):
                    continue
                reachable = self.get_reachable_positions()
                if not reachable:
                    self.rejections["no_reachable_positions"] += 1
                    continue

                view_attempts = int(generation["max_views_per_scene"])
                for _ in range(view_attempts):
                    if self.pair_complete() and self.non_pair_complete():
                        break
                    if self.scene_counts[scene_id] >= int(generation["max_samples_per_scene"]):
                        break
                    if not self.reset_scene(house):
                        break
                    pose = {
                        "position": self.rng.choice(reachable),
                        "rotation": {
                            "x": 0.0,
                            "y": float(self.rng.choice(generation["camera_rotations"])),
                            "z": 0.0,
                        },
                        "horizon": float(self.rng.choice(generation["camera_horizons"])),
                    }
                    event = self.teleport_camera(pose)
                    if event is None or len(visible_objects(event)) < 2:
                        self.rejections["insufficient_visible_objects"] += 1
                        continue
                    context = SceneContext(house=house, scene_id=scene_id, camera_pose=pose)

                    if not self.pair_complete() and self.scene_counts[scene_id] <= int(
                        generation["max_samples_per_scene"]
                    ) - 2:
                        pair_rows = self.try_counterfactual_pair(context, event)
                        if pair_rows is None:
                            pair_rows = self.try_open_state_pair(context, event)
                        if pair_rows is not None:
                            self.rows.extend(pair_rows)
                            print(
                                f"scene={scene_id} samples={len(self.rows)}/{self.target_sample_count()} "
                                f"groups={dict(self.group_counts)}",
                                flush=True,
                            )
                            continue

                    for group in self.remaining_groups():
                        if not self.reset_scene(house):
                            break
                        task_event = self.teleport_camera(pose)
                        if task_event is None:
                            continue
                        candidate = (
                            self.make_pickup_candidate(task_event)
                            if group == "pickup"
                            else self.make_atomic_candidate(task_event, group)
                        )
                        if candidate is None:
                            self.rejections[f"no_candidate:{group}"] += 1
                            continue
                        # Some state setup actions change the current view. Return to the
                        # fixed camera before checking visibility and saving the RGB frame.
                        task_event = self.teleport_camera(pose)
                        if task_event is None:
                            continue
                        row = self.save_sample(context, task_event, candidate)
                        if row is not None:
                            self.rows.append(row)
                            print(
                                f"scene={scene_id} samples={len(self.rows)}/{self.target_sample_count()} "
                                f"groups={dict(self.group_counts)}",
                                flush=True,
                            )
                            break

            expected = self.target_sample_count()
            if len(self.rows) != expected:
                raise RuntimeError(
                    f"Generation incomplete: expected {expected}, produced {len(self.rows)}. "
                    f"Counts={dict(self.group_counts)}, rejections={dict(self.rejections)}"
                )
            self.assign_wrong_images()
            write_jsonl(self.samples_path, self.rows)
            write_json(
                self.report_path,
                {
                    "seed": self.config["seed"],
                    "source": self.config["source"],
                    "sample_count": len(self.rows),
                    "scene_count": len(self.scene_counts),
                    "group_counts": dict(self.group_counts),
                    "scene_counts": dict(self.scene_counts),
                    "counterfactual_ratio": self.group_counts["counterfactual_put"]
                    / len(self.rows),
                    "rejections": dict(self.rejections),
                },
            )
        finally:
            if self.controller is not None:
                self.controller.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate simulator-verified Exp 0 data from AI2-THOR/ProcTHOR"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("generator_config.json"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete only previously generated samples.jsonl, report, and images before generation",
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    return parser


def apply_shard(config: dict[str, Any], shard_index: int, shard_count: int) -> None:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count")
    scene_total = int(config["num_scenes"])
    scene_base, scene_remainder = divmod(scene_total, shard_count)
    scene_sizes = [
        scene_base + (1 if index < scene_remainder else 0)
        for index in range(shard_count)
    ]
    config["scene_offset"] = sum(scene_sizes[:shard_index])
    config["num_scenes"] = scene_sizes[shard_index]
    shard_output_root = str(config.get("shard_output_root", "shard_data")).rstrip("/\\")
    config["output_dir"] = f"{shard_output_root}/shard_{shard_index}"

    generation = config["generation"]
    pair_total = int(generation["counterfactual_pairs"])
    if pair_total % shard_count:
        raise ValueError("counterfactual_pairs must divide evenly across shards")
    generation["counterfactual_pairs"] = pair_total // shard_count
    for group, quota in list(generation["group_quotas"].items()):
        quota = int(quota)
        if quota % shard_count:
            raise ValueError(f"Quota for {group} must divide evenly across shards")
        generation["group_quotas"][group] = quota // shard_count


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if (args.shard_index is None) != (args.shard_count is None):
        raise ValueError("--shard-index and --shard-count must be provided together")
    if args.shard_index is not None and args.shard_count is not None:
        apply_shard(config, args.shard_index, args.shard_count)
    generator = ThorDataGenerator(config, config_path, args.overwrite)
    generator.run()
    print(f"Generated {len(generator.rows)} samples at {generator.samples_path}")


if __name__ == "__main__":
    main()
