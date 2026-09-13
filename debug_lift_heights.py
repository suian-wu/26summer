"""Diagnostic: instead of swinging joint1 to clear the overhead view, test
lifting the end-effector straight up (fixed xy, only z increases) via IK, at
several heights, and check whether the arm body still overlaps the
square_tray region of the frame.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import DampedLeastSquaresIK
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
        obs = env.reset(task)
        ik = DampedLeastSquaresIK(env.model)
        home_xy = obs.ee_position[:2].copy()
        home_quat = obs.ee_quaternion.copy()

        for target_z in [0.55, 0.65, 0.75, 0.85, 0.95, 1.05]:
            obs = env.reset(task)
            target = np.array([home_xy[0], home_xy[1], target_z])
            q = obs.joint_position.copy()
            for _ in range(300):
                result = ik.solve(q, target, home_quat, max_iterations=50)
                q = result.joint_position
                obs, _info = env.step(JointPositionCommand(q, 1.0))
            camera = camera_by_name(obs, "overhead")
            Image.fromarray(camera.rgb).save(f"runs/debug_lift_z_{target_z:.2f}.png")
            print(
                f"target_z={target_z:.2f} converged={result.converged} "
                f"final_ee_z={obs.ee_position[2]:.3f} pos_err={result.position_error:.4f}"
            )


if __name__ == "__main__":
    main()
