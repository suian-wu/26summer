"""A beginner-friendly MuJoCo arm motion demo.

This script is deliberately independent of the assignment policy.  It sends
joint-position commands to the simulated Panda and records the result.

Run from the project root:

    python scripts/manual_arm_demo.py --output runs/manual_demo

On a headless WSL machine the script selects EGL before importing MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# MuJoCo must see this before its renderer is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from graspbench.config import HOME_Q, TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import move_toward
from graspbench.types import JointPositionCommand


def move_to(
    env: GraspEnv,
    observation,
    target_q: np.ndarray,
    *,
    gripper_opening: float,
    frames: list[np.ndarray],
    max_steps: int = 120,
) :
    """Move smoothly to one joint pose and collect diagonal-camera frames."""
    target_q = np.asarray(target_q, dtype=np.float64)
    for step in range(max_steps):
        q_command = move_toward(
            observation.joint_position,
            target_q,
            max_step=0.035,
        )
        command = JointPositionCommand(q_command, gripper_opening)
        observation, info = env.step(command)

        # Rendering every other control step keeps the demo reasonably fast.
        if step % 2 == 0:
            frames.append(env.render(camera="diagonal", width=640, height=480))

        if np.linalg.norm(observation.joint_position - target_q) < 0.015:
            break

    print(
        f"  reached target in {step + 1} control steps; "
        f"ee={np.round(observation.ee_position, 3).tolist()} "
        f"gripper={observation.gripper_opening:.2f} "
        f"unsafe_contacts={info.unsafe_contacts}"
    )
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description="Move the simulated Panda through safe poses")
    parser.add_argument("--output", default="runs/manual_demo")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # This task is only used to initialize a valid scene; no answer is read.
    task = TaskSpec(
        episode_id="manual_demo",
        instruction="manual arm motion demo",
        target="red_cube",
        seed=0,
    )

    # A few modest variations around HOME_Q.  They are joint targets, not
    # hidden task coordinates, and are only for learning how control works.
    poses = [
        ("home", HOME_Q.copy(), 1.0),
        ("move_left", HOME_Q + np.array([0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.0),
        ("move_right", HOME_Q + np.array([-0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.0),
        ("bend", HOME_Q + np.array([0.0, 0.99, 0.0, 0.20, 0.0, -0.12, 0.0]), 1.0),
        ("close_gripper", HOME_Q.copy(), 0.0),
        ("open_gripper", HOME_Q.copy(), 1.0),
    ]

    frames: list[np.ndarray] = []
    trajectory: list[dict] = []

    with GraspEnv() as env:
        observation = env.reset(task)
        frames.append(env.render(camera="diagonal", width=640, height=480))

        print("Starting manual Panda motion demo")
        for name, target_q, gripper in poses:
            print(f"[{name}] target_q={np.round(target_q, 3).tolist()}")
            observation = move_to(
                env,
                observation,
                target_q,
                gripper_opening=gripper,
                frames=frames,
            )
            trajectory.append(
                {
                    "stage": name,
                    "sim_time_s": observation.time,
                    "joint_position": observation.joint_position.tolist(),
                    "ee_position": observation.ee_position.tolist(),
                    "gripper_opening": observation.gripper_opening,
                }
            )

    imageio.mimsave(output / "manual_arm_demo.mp4", frames, fps=15, macro_block_size=16)
    imageio.imwrite(output / "first_frame.png", frames[0])
    imageio.imwrite(output / "last_frame.png", frames[-1])
    (output / "trajectory.json").write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Wrote demo files to {output.resolve()}")


if __name__ == "__main__":
    main()
