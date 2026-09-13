"""Diagnostic: bypass FoundationModelPerception.detect_scene()'s silent
`except (TypeError, ValueError): continue` and surface the *actual* reason
square_tray's candidate gets rejected during decoding, for t2_public_002
(seed=62), where the live sam_server.py log confirms Grounding DINO found a
candidate box but detect_scene() still returned no square_tray detection.
"""
from __future__ import annotations

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.perception import FoundationModelPerception, ObjectSpec

# Mirrors StudentPolicy._shift_start, same as debug_task2_detections.py, so
# this sees the same frame the real policy detects from.
import numpy as np
from graspbench.ik import DampedLeastSquaresIK, move_toward
from graspbench.types import JointPositionCommand

STARTING_SHIFT_Y = 0.33
TRANSIT_HEIGHT = 0.52
CARTESIAN_STEP = 0.02
MAX_SHIFT_STEPS = 400


def _shift_like_policy(env: GraspEnv, obs):
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
    task = TaskSpec(
        "t2_public_002",
        "Place the apple into the square tray.",
        "apple",
        62,
        destination="square_tray",
        scene_objects=("banana", "apple"),
    )
    with GraspEnv() as env:
        obs = env.reset(task)
        obs = _shift_like_policy(env, obs)
        camera = camera_by_name(obs, "overhead")

        perception = FoundationModelPerception()
        prompt = perception.SAM_PROMPTS["square_tray"]
        response, latency = perception._call_sam3(camera.rgb, [prompt])
        result = response["results"][0]
        print("raw result: scores=", result["scores"], "boxes=", result["boxes"])

        spec = perception.spec_by_name["square_tray"]
        try:
            detected, evidence = perception._decode_detection(
                camera,
                spec,
                prompt,
                result,
                latency_s=latency,
                prior_xy=None,
                surface_percentile=(spec.surface_percentile if isinstance(spec, ObjectSpec) else 70.0),
            )
            print("SUCCESS:", detected.position)
        except (TypeError, ValueError) as exc:
            print("REJECTED with:", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
