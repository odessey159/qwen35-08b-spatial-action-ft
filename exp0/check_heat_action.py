from __future__ import annotations

import argparse

import prior
from ai2thor.controller import Controller
from ai2thor.platform import Linux64


FOOD_TYPES = {"Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato"}


def walk_objects(objects: list[dict]) -> list[dict]:
    result: list[dict] = []
    for obj in objects:
        result.append(obj)
        result.extend(walk_objects(obj.get("children", [])))
    return result


def promote_food_children(objects: list[dict]) -> list[dict]:
    promoted: list[dict] = []
    for obj in objects:
        children = obj.get("children", [])
        retained: list[dict] = []
        for child in children:
            if child.get("id", "").split("|", 1)[0] in FOOD_TYPES:
                promoted.append(child)
            else:
                promoted.extend(promote_food_children([child]))
                retained.append(child)
        obj["children"] = retained
    objects.extend(promoted)
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scene-index", type=int, required=True)
    parser.add_argument("--promote-food", action="store_true")
    args = parser.parse_args()

    house = prior.load_dataset("procthor-10k", revision=args.revision)["train"][args.scene_index]
    raw_foods = [
        obj
        for obj in walk_objects(house["objects"])
        if obj.get("id", "").split("|", 1)[0] in FOOD_TYPES
    ]
    print(
        "raw_foods="
        + repr(
            [
                {
                    key: obj.get(key)
                    for key in ("id", "assetId", "position", "rotation", "kinematic")
                }
                for obj in raw_foods
            ]
        ),
        flush=True,
    )
    if args.promote_food:
        promoted = promote_food_children(house["objects"])
        print(
            "promoted="
            + repr([(obj.get("id"), obj.get("assetId"), obj.get("position")) for obj in promoted])
        )
    controller = Controller(
        scene=house,
        platform=Linux64,
        width=300,
        height=300,
        fieldOfView=90,
        visibilityDistance=5.0,
        gridSize=0.25,
        rotateStepDegrees=45,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        snapToGrid=False,
    )
    try:
        metadata_objects = controller.last_event.metadata["objects"]
        foods = [obj for obj in metadata_objects if obj.get("objectType") in FOOD_TYPES]
        print(
            "foods="
            + repr(
                [
                    (
                        obj["objectType"],
                        obj["objectId"],
                        obj.get("visible"),
                        obj.get("pickupable"),
                        obj.get("cookable"),
                        obj.get("isCooked"),
                    )
                    for obj in foods
                ]
            )
        )
        cookables = [obj for obj in metadata_objects if obj.get("cookable")]
        print(
            "cookables="
            + repr(
                [
                    (obj["objectType"], obj["objectId"], obj.get("visible"), obj.get("isCooked"))
                    for obj in cookables
                ]
            )
        )
        for target in foods:
            poses_event = controller.step(
                action="GetInteractablePoses",
                objectId=target["objectId"],
                rotations=[0, 45, 90, 135, 180, 225, 270, 315],
                horizons=[0, 30],
                standings=[True],
            )
            poses = poses_event.metadata.get("actionReturn") or []
            print(
                f"target={target['objectType']} poses={len(poses)} "
                f"pose_error={poses_event.metadata.get('errorMessage')!r}"
            )
            if poses:
                controller.step(action="TeleportFull", **poses[0])
            result = controller.step(
                action="CookObject", objectId=target["objectId"], forceAction=True
            )
            print(
                f"force_true success={result.metadata.get('lastActionSuccess')} "
                f"error={result.metadata.get('errorMessage')!r} "
                f"isCooked={next((obj.get('isCooked') for obj in result.metadata['objects'] if obj['objectId'] == target['objectId']), None)}"
            )
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
