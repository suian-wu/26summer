"""Diagnostic: find the true reachable ceiling for horizontal transit height,
between the previously-verified-safe 0.52m and the newly-tried-but-unreachable
0.60m. TRANSIT_HEIGHT needs to be as high as possible (to clear tray rims
during transit_place) while still being IK-reachable across the whole
tabletop x range the policy actually visits, including the shifted starting
position (STARTING_SHIFT_Y) used before first detection.
"""
from __future__ import annotations

import numpy as np

from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import DampedLeastSquaresIK

# Mirror the real x/y range this policy actually needs to transit across:
# object pickup range (0.54-0.70), the shifted starting corridor (round tray
# side, y~0.33 offset from home), and container placement positions
# (~0.30-0.35 x range for both trays).
XS = [0.30, 0.35, 0.40, 0.55, 0.60, 0.65, 0.70]
Y = -0.10  # representative y; container/tray y ranges are the ones that matter for edge clearance


def main() -> None:
    with GraspEnv() as env:
        obs = env.reset(TaskSpec("t", "pick red cube", "red_cube", 0))
        ik = DampedLeastSquaresIK(env.model)
        quat = obs.ee_quaternion.copy()
        q = obs.joint_position.copy()

        for z in [0.52, 0.54, 0.56, 0.58, 0.60]:
            row = []
            for x in XS:
                target = np.array([x, Y, z])
                result = ik.solve(q, target, quat, max_iterations=300)
                row.append("OK" if result.converged else f"{result.position_error:.3f}")
            print(f"z={z:.2f}: " + " | ".join(f"x={x:.2f}:{v:>7s}" for x, v in zip(XS, row)))


if __name__ == "__main__":
    main()
