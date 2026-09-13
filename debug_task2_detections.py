"""Diagnostic: run the full FoundationModelPerception.detect_scene() pipeline
for every Task 2 episode (configs/task2_ycb_public.json), draw every
returned box on the overhead frame, and print each detected world position
next to the true (ground-truth) position for banana/apple/square_tray/round_tray.

This does NOT bypass the real pipeline -- it calls the same
FoundationModelPerception.detect_scene() that student_policy.py uses, hitting
the live sam_server.py over HTTP exactly as evaluate.py would.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

import numpy as np

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import DampedLeastSquaresIK, move_toward
from graspbench.perception import FoundationModelPerception
from graspbench.types import JointPositionCommand

COLORS = {
    "square_tray": (255, 0, 0),
    "round_tray": (0, 0, 255),
    "banana": (255, 255, 0),
    "apple": (0, 255, 0),
}
RELEVANT_NAMES = ("banana", "apple", "square_tray", "round_tray")

# Mirrors StudentPolicy._shift_start / STARTING_SHIFT_Y and TRANSIT_HEIGHT
# in policies/student_policy.py, so this script sees the same overhead frame
# the real policy actually detects from (not the untouched HOME_Q frame).
STARTING_SHIFT_Y = 0.33
TRANSIT_HEIGHT = 0.52
CARTESIAN_STEP = 0.02
MAX_SHIFT_STEPS = 400


def _shift_like_policy(env: GraspEnv, obs) -> None:
    ik = DampedLeastSquaresIK(env.model)
    home_quaternion = obs.ee_quaternion.copy()
    shift_target = obs.ee_position.copy()
    shift_target[1] += STARTING_SHIFT_Y
    shift_target[2] = TRANSIT_HEIGHT
    q = obs.joint_position.copy()
    for _ in range(MAX_SHIFT_STEPS):
        offset = shift_target - obs.ee_position
        distance = float(np.linalg.norm(offset))
        if distance < 0.015:
            break
        waypoint = (
            obs.ee_position + offset * (CARTESIAN_STEP / distance)
            if distance > CARTESIAN_STEP
            else shift_target
        )
        result = ik.solve(q, waypoint, home_quaternion, rest_qpos=q)
        q = move_toward(q, result.joint_position)
        obs, _info = env.step(JointPositionCommand(q, 1.0))
    return obs


def main() -> None:
    task_file = Path("configs/task2_ycb_public.json")
    payload = json.loads(task_file.read_text(encoding="utf-8"))
    tasks = [TaskSpec.from_dict(item) for item in payload["episodes"]]

    with GraspEnv() as env:
        perception = FoundationModelPerception()
        for task in tasks:
            obs = env.reset(task)
            obs = _shift_like_policy(env, obs)  # same one-way shift the real policy does
            camera = camera_by_name(obs, "overhead")
            detections, evidence = perception.detect_scene(camera)

            img = Image.fromarray(camera.rgb).convert("RGB")
            draw = ImageDraw.Draw(img)
            for name, ev in evidence.items():
                x1, y1, x2, y2 = ev.box_xyxy
                color = COLORS.get(name, (255, 255, 255))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 2, max(0, y1 - 12)), f"{name}:{ev.score:.2f}", fill=color)
            out_path = f"runs/debug_task2_{task.episode_id}_overlay.png"
            img.save(out_path)

            print(f"\n=== {task.episode_id} (seed={task.seed}) instruction={task.instruction!r} ===")
            print(f"saved overlay to {out_path}")
            for name in RELEVANT_NAMES:
                detected_pos = detections[name].position if name in detections else None
                body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                true_pos = env.data.xpos[body_id]
                print(f"  {name:14s} detected={detected_pos} true={true_pos}")


if __name__ == "__main__":
    main()
