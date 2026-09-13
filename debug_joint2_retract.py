"""Diagnostic: swing only joint2 (shoulder pitch, the "up-down" joint that
folds the arm back over its own base) away from HOME_Q, leaving every other
joint (including joint1, the base yaw) untouched. This retracts the arm off
the table -- its xy position changes because folding the shoulder back pulls
the whole arm backward -- without swinging sideways via joint1.

Sent directly as a joint-space command (not through Cartesian/IK _move_to),
per instruction: only one joint moves.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from graspbench.camera import camera_by_name
from graspbench.config import HOME_Q, TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import move_toward
from graspbench.types import JointPositionCommand


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
        # joint2 range is -1.7628 to 1.7628; HOME_Q[1] is -0.45. Pulling it
        # toward the negative limit folds the shoulder back over the base.
        for joint2_target in [-1.6, -1.7628]:
            obs = env.reset(task)
            target_q = HOME_Q.copy()
            target_q[1] = joint2_target
            q = obs.joint_position.copy()
            for _ in range(500):
                q = move_toward(q, target_q, max_step=0.02)
                obs, _info = env.step(JointPositionCommand(q, 1.0))
            camera = camera_by_name(obs, "overhead")
            Image.fromarray(camera.rgb).save(f"runs/debug_joint2_{joint2_target:.2f}.png")
            print(
                f"joint2_target={joint2_target:.2f} final_joint2={obs.joint_position[1]:.3f} "
                f"final_ee_xyz={obs.ee_position.round(3)}"
            )


if __name__ == "__main__":
    main()
