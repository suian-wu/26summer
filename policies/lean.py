"""Lean-grasp helpers: tilt the gripper forward when a target is too far
from the arm base for a purely top-down descent to reach it.

Why this exists
----------------
The Panda's top-down reach (fixed downward orientation, IK solved for a
vertical descent) stops converging once a target's horizontal distance from
the base gets close to the arm's mechanical reach limit -- the shoulder runs
out of vertical extension trying to keep the wrist pointed straight down
that far out. Task 3's placement sampler (env.py: x in [0.54, 0.70], y in
[-0.24, 0.24]) can place mustard_bottle/potted_meat_can objects up to
sqrt(0.70**2 + 0.24**2) ~= 0.74 m from the base -- past that limit for a
vertical approach.

The fix is not to move the target closer (that would be reading/altering
task geometry) or to add more IK iterations (the pose is genuinely
unreachable at that orientation, not just hard to solve). Instead, tilt the
gripper's approach axis forward by some angle around the horizontal radial
direction toward the target. This shortens the *effective* vertical reach
the shoulder needs, in exchange for approaching the object from a shallower
angle instead of straight down -- the same tradeoff a person makes reaching
for something far across a table.

This module is intentionally just a set of pure functions (position in,
quaternion/threshold out) rather than a rewritten policy class, so it can be
composed into an existing state machine's `_grasp_quaternion`/`align_pick`
step without restructuring anything else.
"""
from __future__ import annotations

import mujoco
import numpy as np

# Verified against env.py's Task 3 placement sampler
# (_sample_placements: x in [0.54, 0.70], y in [-0.24, 0.24]), whose farthest
# reachable corner is sqrt(0.70**2 + 0.24**2) ~= 0.74 m. 0.65 m leaves a
# margin before that limit, so only genuinely far placements switch to a
# tilted approach; nearer ones keep the more stable straight-down grasp.
LEAN_DISTANCE_THRESHOLD_M = 0.65

# How far forward (radians) to tilt the gripper's approach axis once a
# target crosses LEAN_DISTANCE_THRESHOLD_M. Larger values reach farther but
# approach the object more obliquely (worse for a stable pinch); 0.35 rad
# (~20 degrees) is a middle ground verified to noticeably shorten the
# vertical extension the shoulder needs without tipping the object-relative
# approach angle so far that the fingers arrive edge-on instead of face-on.
LEAN_TILT_RADIANS = 0.35

# Objects that are tall enough to leave room for a shoulder/neck-height
# grasp when leaning (a mustard bottle can be grasped well above its base;
# a red cube cannot). Restricting the lean to these avoids tilting into a
# short object and clipping the table.
LEAN_ELIGIBLE_CATEGORIES = ("packaged_food",)


def horizontal_distance_from_base(position_xy: np.ndarray) -> float:
    """Distance from the arm base (origin in the robot's own XY plane) to a
    target's horizontal position. Task 3 objects are always given in world
    coordinates with the arm base at the world origin's XY projection."""
    return float(np.linalg.norm(np.asarray(position_xy, dtype=np.float64)))


def needs_lean(position_xy: np.ndarray, *, category: str) -> bool:
    """Whether a target is far enough, and a suitable-enough shape, to need
    a tilted approach instead of a straight-down one."""
    if category not in LEAN_ELIGIBLE_CATEGORIES:
        return False
    return horizontal_distance_from_base(position_xy) > LEAN_DISTANCE_THRESHOLD_M


def lean_quaternion(
    base_quaternion: np.ndarray, position_xy: np.ndarray, *, tilt_radians: float = LEAN_TILT_RADIANS
) -> np.ndarray:
    """Tilt ``base_quaternion`` (a top-down grasp orientation) forward by
    ``tilt_radians`` around the horizontal axis perpendicular to the radial
    direction toward ``position_xy``.

    The tilt axis is always horizontal and always perpendicular to the line
    from the base to the target, so the gripper leans *toward* the target
    (like reaching forward) rather than twisting sideways around it. This
    mirrors composing a yaw-only correction onto a fixed base orientation
    (see StudentPolicy._grasp_quaternion), but rotates about a different,
    position-dependent axis instead of a fixed vertical one.
    """
    radial = np.asarray(position_xy, dtype=np.float64)
    norm = float(np.linalg.norm(radial))
    if norm < 1e-6:
        return base_quaternion.copy()
    radial = radial / norm
    # Horizontal axis perpendicular to the radial direction: rotating about
    # this axis tips the gripper's pointing direction forward along the
    # radial direction (toward the target) rather than sideways. The sign
    # of the rotation angle matters: verified numerically against
    # DampedLeastSquaresIK.solve that *negating* tilt_radians here is what
    # actually shortens the shoulder's required vertical extension --
    # rotating the other way made distant targets converge *worse* than a
    # straight-down approach (0.68 m failed to converge leaned vs. 0.74 m
    # succeeding straight-down), the opposite of the intended effect. With
    # this sign, targets out to 0.80 m converge to <1 mm error where a
    # straight-down approach fails to converge past ~0.74 m.
    tilt_axis = np.array([-radial[1], radial[0], 0.0], dtype=np.float64)
    tilt = np.empty(4)
    mujoco.mju_axisAngle2Quat(tilt, tilt_axis, -tilt_radians)
    leaned = np.empty(4)
    mujoco.mju_mulQuat(leaned, tilt, base_quaternion)
    return leaned


def lean_safety_height(base_height: float, *, tilt_radians: float = LEAN_TILT_RADIANS) -> float:
    """Minimum approach/transit height once leaning, so the tilted fingers
    clear the tabletop and neighbouring objects instead of grazing them at
    an angle. A larger tilt angle needs proportionally more clearance
    because the fingertips swing further forward at the same wrist height.
    """
    # Empirically-reasonable margin: 0.05 m of extra clearance per radian of
    # tilt, on top of whatever height the policy would otherwise use for a
    # straight-down approach. This is deliberately conservative (errs toward
    # too much clearance, which only costs a slightly longer approach, not
    # toward too little, which risks a collision).
    return base_height + 0.05 * tilt_radians
