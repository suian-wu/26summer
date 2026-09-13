"""Diagnostic: run the full FoundationModelPerception.detect_scene() pipeline
for seed=31 (t1_public_001, "put red cube in square tray"), draw every
returned box on the overhead frame, and print each detected world position
next to the true (ground-truth) position for every relevant object.

This does NOT bypass the real pipeline -- it calls the same
FoundationModelPerception.detect_scene() that student_policy.py uses, hitting
the live sam_server.py over HTTP exactly as evaluate.py would.
"""
from __future__ import annotations

import mujoco
from PIL import Image, ImageDraw

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.perception import FoundationModelPerception

COLORS = {
    "square_tray": (255, 0, 0),
    "round_tray": (0, 0, 255),
    "red_cube": (0, 255, 0),
    "green_cylinder": (255, 255, 0),
    "blue_box": (255, 0, 255),
}


def main() -> None:
    with GraspEnv() as env:
        task = TaskSpec(
            "t1_public_001",
            "把红色方块放进方盘。",
            "red_cube",
            31,
            destination="square_tray",
            scene_objects=("red_cube", "green_cylinder", "blue_box"),
        )
        obs = env.reset(task)  # real HOME_Q pose, no clearing/shifting move
        camera = camera_by_name(obs, "overhead")
        perception = FoundationModelPerception()
        detections, evidence = perception.detect_scene(camera)

        img = Image.fromarray(camera.rgb).convert("RGB")
        draw = ImageDraw.Draw(img)
        for name, ev in evidence.items():
            x1, y1, x2, y2 = ev.box_xyxy
            color = COLORS.get(name, (255, 255, 255))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), f"{name}:{ev.score:.2f}", fill=color)
        img.save("runs/debug_all_detections_overlay.png")
        print("Saved overlay to runs/debug_all_detections_overlay.png")

        print("\n=== detected vs true world position ===")
        for name in ["red_cube", "square_tray", "round_tray", "green_cylinder", "blue_box"]:
            detected_pos = detections[name].position if name in detections else None
            if name in ("square_tray", "round_tray"):
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            else:
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            true_pos = env.data.xpos[body_id]
            print(f"{name:16s} detected={detected_pos} true={true_pos}")


if __name__ == "__main__":
    main()
