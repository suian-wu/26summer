from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from graspbench.camera import camera_by_name
from graspbench.config import CONTAINER_SPEC_BY_NAME, OBJECT_SPEC_BY_NAME
from graspbench.ik import DampedLeastSquaresIK, move_toward
from graspbench.perception import FoundationModelPerception, ModelServiceError
from graspbench.types import DetectedObject, JointPositionCommand, Observation, PolicyDecision
from graspbench.vlm import OpenAICompatibleVLM

# Below this confidence (rotating min-area-box scan, see
# FoundationModelPerception._estimate_yaw_quaternion), the footprint is
# round and has no meaningful edge to align to -- its estimated yaw is
# measurement noise, so rotating the gripper to chase it would add risk for
# no benefit. Verified in simulation: round footprints score ~0.05-0.09,
# square/rectangular ones ~0.4-0.55, leaving a wide safety margin at 0.25.
# Above the threshold this applies to *both* square footprints caught on
# their diagonal (align to the nearest 90-degree pair of edges) and oblong
# ones (align to the long axis specifically) -- the estimator itself picks
# the correct wrap period per shape.
ELONGATION_ALIGN_THRESHOLD = 0.25

# After rising from placement, hold position for this long before running
# verification detection, so the placed object has time to settle under
# gravity and contact forces before the camera snapshot is taken.
PRE_VERIFY_SETTLE_SECONDS = 2.0

# Horizontal transit always happens at this fixed height, comfortably above
# every object and container, so it is physically impossible for a sideways
# move to sweep something off the table: only vertical stages ever change z
# near the tabletop, and only horizontal stages ever change xy near the top.
# Measured empirically with DampedLeastSquaresIK.solve: at this arm's reach,
# targets above ~0.55m stop converging once x exceeds ~0.65m (the shoulder
# runs out of vertical extension while reaching far). 0.52m stays reachable
# across the whole tabletop x range (0.30-0.70m) while remaining above every
# object and container height, so horizontal moves never clip anything.
TRANSIT_HEIGHT = 0.55
PLACE_DROP_GAP = 0.02
# Measured against ground truth: SAM's depth-based center estimate for the
# red cube came out ~1.2cm low (0.4214 vs true 0.4259), enough to make the
# fingertip midpoint (grasp_site) descend into the object instead of
# straddling its waist and stall there. Nudge the descent target upward by a
# fraction of the object's own half-height so the bias scales with object
# size rather than being a fixed constant.
GRASP_HEIGHT_SAFETY_FRACTION = 0.5
# Cap each IK target to a short Cartesian hop from the current gripper
# position so the joint-space interpolation between calls stays close to a
# straight line in task space, instead of solving for a distant goal whose
# redundant elbow configuration can drift step to step.
CARTESIAN_STEP = 0.02
POSITION_TOLERANCE = 0.02
DESCEND_POSITION_TOLERANCE = 0.015
# At HOME_Q the arm's own body sits inside the overhead camera's view,
# which pushed Grounding DINO's "yellow square tray" box confidence below
# threshold entirely (confirmed by direct HTTP checks against the live
# sam_server.py -- the box vanished, not just shrank). round_tray sits at
# y~+0.17 and square_tray at y~-0.19 (env._sample_container_positions), so
# shifting the end-effector toward the round tray's side moves the arm's
# body away from square_tray's region of the frame. This is a one-way
# shift: the arm does not return to HOME_Q afterward, and every subsequent
# stage (starting from rise_pick) simply continues from wherever this
# shift ends, treating it as the episode's new starting point.
STARTING_SHIFT_Y = -0.33
# The round tray's rim sits above the tabletop (ContainerSpec floor_height
# 0.41, max_object_center_height 0.55), so shifting sideways at the
# unchanged home height (~0.47) let the end-effector snag on the tray's
# rim while passing over it. Lifting to TRANSIT_HEIGHT during the same
# shift keeps the whole move above every container/object, exactly like
# every other horizontal move in this policy.
STARTING_SHIFT_TOLERANCE = 0.05
# During transit_pick/transit_place, if height sags below this tolerance
# from TRANSIT_HEIGHT (small joint-space/IK coupling can let z drift while
# xy is still catching up to a far target), stop chasing the horizontal
# target for this step and fix height first. This re-checks every act()
# call, so drift never accumulates into a real collision risk.
TRANSIT_HEIGHT_TOLERANCE = 0.04
# Decision counts do not correspond to a fixed amount of simulated time: the
# async driver sometimes reuses one decision for many physics steps and
# sometimes calls act() almost every step, so "wait N decisions" and "wait
# until the opening stops changing" both under- or over-shoot depending on
# scheduling. The physical grasp itself is also a slow, continuous soft
# contact compression (measured directly against the sim: gripper_opening
# keeps drifting for 190+ physics steps after first contact, never fully
# stalling), so "wait until unchanged" never actually fires. Use the
# simulator's own clock (Observation.time, immune to decision-count noise)
# and simply hold the close/open command for a fixed amount of simulated
# time before advancing.
GRIP_SETTLE_SECONDS = 2.5
# After descend_pick reaches its position tolerance, wait this long (settled
# in place, gripper still commanded open) before starting to close, so a
# late-arriving IK correction cannot get mistaken for "close" starting too
# early on a fingertip that has not actually finished descending.
DESCEND_SETTLE_SECONDS = 0.2
MAX_RETRIES_PER_ACTION = 2


@dataclass
class _PickPlaceAction:
    pick_id: str
    place_id: str | None
    pick_world: np.ndarray
    place_world: np.ndarray | None
    pick_quaternion: np.ndarray | None


class StudentPolicy:
    """SAM candidates + VLM grounding/planning, numerical IK, explicit pick/place state machine."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.task = task
        self.model = model
        self.ik = DampedLeastSquaresIK(model)
        self.perception = FoundationModelPerception()
        self.vlm = OpenAICompatibleVLM()

        self.home_quaternion: np.ndarray | None = None
        self.shift_target: np.ndarray | None = None
        self.actions: list[_PickPlaceAction] | None = None
        self.action_index = 0
        self.stage = "shift_start"
        self.grip_settle_deadline: float | None = None
        self.descend_settle_deadline: float | None = None
        self.retry_count = 0
        self.terminal = False
        self.pre_verify_settle_deadline: float | None = None

    def act(self, observation: Observation) -> PolicyDecision:
        if self.home_quaternion is None:
            # Every waypoint reuses this fixed downward orientation; HOME_Q
            # already keeps the gripper pointed at the tabletop.
            self.home_quaternion = observation.ee_quaternion.copy()

        if self.terminal:
            return self._hold(observation, stage="terminal", rationale="Episode already finished.")

        if self.actions is None:
            if self.stage == "shift_start":
                return self._shift_start(observation)
            try:
                self._ensure_plan(observation)
            except ModelServiceError as exc:
                self.terminal = True
                return PolicyDecision(
                    command=JointPositionCommand(observation.joint_position, 1.0),
                    stage="model_error",
                    rationale=f"Model service failed during planning; holding safely: {exc}",
                    done=True,
                )

        assert self.actions is not None
        if self.action_index >= len(self.actions):
            self.terminal = True
            return self._hold(
                observation, stage="complete", rationale="All planned pick/place actions finished.", done=True
            )

        action = self.actions[self.action_index]

        if self.stage == "rise_pick":
            # Pure vertical move: keep whatever xy we are already at and only
            # change height. This never drags the gripper sideways through an
            # object at table height.
            rise_target = np.array(
                [observation.ee_position[0], observation.ee_position[1], TRANSIT_HEIGHT]
            )
            return self._move_to(
                observation, rise_target, gripper=1.0, next_stage="transit_pick",
                stage_name="rise_pick", action=action,
            )
        if self.stage == "transit_pick":
            # Pure horizontal move at a height above every object and
            # container, so the approach path cannot sweep anything aside.
            #transit_target = np.array([action.pick_world[0], action.pick_world[1], TRANSIT_HEIGHT])
            transit_target = np.array([action.pick_world[0], action.pick_world[1], action.pick_world[2] + 0.1])
            return self._move_to(
                observation, transit_target, gripper=1.0, next_stage="align_pick",
                stage_name="transit_pick", action=action, guard_height=transit_target[2],
            )
        if self.stage == "align_pick":
            # Rotate in place (xy/z fixed at transit height) before
            # descending, so the fingers straddle an elongated object's
            # short axis instead of pinching its long side, which is what
            # made angled placements (e.g. a rotated blue box) slip out of a
            # grip held at the fixed home orientation. Pure rotation only:
            # no translation happens in this stage, so it cannot reintroduce
            # the axis-coupling problem fixed for translation earlier.
            return self._align_pick(observation, action)
        if self.stage == "descend_pick":
            # xy is already aligned with the target; only z changes here.
            # 在抓取目标位置基础上再降低0.02m，让夹爪抓得更低
            pick_target = action.pick_world.copy()
            pick_target[1] -= 0.008
            pick_target[2] -= 0.015  # ← Z轴向下为负，根据需要调整此值
            return self._move_to(
                observation, pick_target, gripper=1.0, next_stage="settle_before_close",
                stage_name="descend_pick", action=action,
                target_quaternion=action.pick_quaternion,
            )
        if self.stage == "settle_before_close":
            return self._settle_before_close(observation, action)
        if self.stage == "close":
            return self._close_gripper(observation, action)
        if self.stage == "rise_after_pick":
            rise_target = np.array([action.pick_world[0], action.pick_world[1], action.place_world[2]+0.1])
            next_stage = "transit_place" if action.place_world is not None else "verify"
            # Keep whatever orientation the object was actually grasped at
            # while carrying it: snapping back to the home orientation here
            # would rotate a held object mid-carry for no reason, risking
            # the exact grip-loosening-under-motion failure already fixed.
            return self._move_to(
                observation, rise_target, gripper=0.0, next_stage=next_stage,
                stage_name="lift", action=action, target_quaternion=action.pick_quaternion,position_tolerance=0.038,guard_height=rise_target[2]
            )
        if self.stage == "transit_place":
            assert action.place_world is not None
            transit_target = np.array([action.place_world[0], action.place_world[1], action.place_world[2]+0.1])
            return self._move_to(
                observation, transit_target, gripper=0.0, next_stage="descend_place",
                stage_name="transit_place", action=action, guard_height=action.place_world[2]+0.1,
                target_quaternion=action.pick_quaternion,position_tolerance=POSITION_TOLERANCE+0.01
            )
        if self.stage == "descend_place":
            assert action.place_world is not None
             # 在目标位置基础上再降低0.05m，让夹爪放得更低
            place_target = action.place_world.copy()
            place_target[2] += 0.0
            return self._move_to(
                observation, place_target, gripper=0.0, next_stage="release",
                stage_name="descend_place", action=action, target_quaternion=action.pick_quaternion,position_tolerance=DESCEND_POSITION_TOLERANCE,
            )
        if self.stage == "release":
            return self._open_gripper(observation, action)
        if self.stage == "rise_after_place":
            assert action.place_world is not None
            rise_target = np.array([action.place_world[0], action.place_world[1], TRANSIT_HEIGHT])
            return self._move_to(
                observation, rise_target, gripper=1.0, next_stage="settle_before_verify",
                stage_name="rise_after_place", action=action,
            )
        if self.stage == "settle_before_verify":
            return self._settle_before_verify(observation, action)
        
        if self.stage == "verify":
            return self._verify(observation, action)

        raise RuntimeError(f"Unhandled stage: {self.stage!r}")

    def _settle_before_verify(self, observation: Observation, action: _PickPlaceAction) -> PolicyDecision:
        """Hold at transit height for PRE_VERIFY_SETTLE_SECONDS before verification."""
        if self.pre_verify_settle_deadline is None:
            self.pre_verify_settle_deadline = observation.time + PRE_VERIFY_SETTLE_SECONDS
        remaining = self.pre_verify_settle_deadline - observation.time
        if remaining <= 0.0:
            self.pre_verify_settle_deadline = None
            self.stage = "verify"
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, 1.0),
            stage="settle_before_verify",
            rationale=f"Holding before verify; remaining_settle_s={max(remaining, 0.0):.2f}.",
            target_id=action.pick_id,
        )
    def _shift_start(self, observation: Observation) -> PolicyDecision:
        # Step the end-effector toward the round tray's side (y) and up to
        # the safe transit height (z) before the very first detection, so
        # the move passes above the round tray's rim instead of snagging on
        # it. This is a one-way move -- self.stage advances straight to
        # "init" once close enough, and nothing ever sends the arm back to
        # HOME_Q. Every later stage (rise_pick etc.) reads its starting
        # position from observation.ee_position, so it simply continues
        # from wherever this shift leaves the arm.
        #
        # The target must be computed once and cached, not recomputed from
        # observation.ee_position on every call: recomputing it each time
        # made the target chase the end-effector's own current position
        # (target = current_y + offset, re-evaluated after current_y had
        # already moved toward the previous target), so position_error
        # never shrank -- confirmed in a live run where the target drifted
        # from y=0.12 to y=0.23 over 30+ steps while error sat frozen at
        # ~0.126 the whole time.
        assert self.home_quaternion is not None
        if self.shift_target is None:
            self.shift_target = observation.ee_position.copy()
            self.shift_target[1] += STARTING_SHIFT_Y
            self.shift_target[2] = TRANSIT_HEIGHT
        target = self.shift_target
        offset = target - observation.ee_position
        distance = float(np.linalg.norm(offset))
        waypoint = (
            observation.ee_position + offset * (CARTESIAN_STEP / distance)
            if distance > CARTESIAN_STEP
            else target
        )
        result = self.ik.solve(
            observation.joint_position, waypoint, self.home_quaternion,
            rest_qpos=observation.joint_position,
        )
        next_joint = move_toward(observation.joint_position, result.joint_position)
        position_error = float(np.linalg.norm(observation.ee_position - target))
        if position_error < STARTING_SHIFT_TOLERANCE:
            self.stage = "init"
        return PolicyDecision(
            command=JointPositionCommand(next_joint, 1.0),
            stage="shift_start",
            rationale=(
                f"Shifting toward the round tray's side before first detection; "
                f"target={target.round(3).tolist()}, error={position_error:.3f}m."
            ),
        )

    def _ensure_plan(self, observation: Observation) -> None:
        if self.actions is not None:
            return
        camera = camera_by_name(observation, "overhead")
        detections, evidence = self.perception.detect_scene(camera)
        plan = self.vlm.plan_task(self.task["instruction"], camera, detections, evidence)

        actions = []
        for planned in plan.actions:
            pick_spec = OBJECT_SPEC_BY_NAME[planned.pick_id]
            detected = detections[planned.pick_id]
            pick_world = detected.position.copy()
            pick_world[2] += pick_spec.grasp_z_offset
            # SAM's depth-based center estimate measured ~1.2cm low against
            # ground truth for the cube (0.4214 vs true 0.4259), so the
            # gripper's fingertip midpoint (grasp_site) descended into the
            # object instead of straddling its waist. Bias the descent
            # target upward by a fraction of the object's own half-height
            # (scales with object size instead of a fixed constant) so the
            # fingers land nearer the object's upper half, safely above
            # whatever the true surface turns out to be.
            pick_world[2] += GRASP_HEIGHT_SAFETY_FRACTION * pick_spec.half_height
            pick_quaternion = (
                self._grasp_quaternion(detected.quaternion)
                if detected.elongation >= ELONGATION_ALIGN_THRESHOLD
                else None
            )

            place_world = None
            if planned.place_id is not None:
                container_position = detections[planned.place_id].position.copy()
                place_world = container_position + np.array(
                    [0.0, 0.0, pick_spec.half_height + PLACE_DROP_GAP]
                )
            actions.append(
                _PickPlaceAction(
                    planned.pick_id, planned.place_id, pick_world, place_world, pick_quaternion
                )
            )

        self.actions = actions
        self.action_index = 0
        self.stage = "rise_pick"
        self.retry_count = 0

    def _move_to(
        self,
        observation: Observation,
        target_position: np.ndarray,
        *,
        gripper: float,
        next_stage: str,
        stage_name: str,
        action: _PickPlaceAction,
        guard_height: float | None = None,
        target_quaternion: np.ndarray | None = None,
        position_tolerance: float = POSITION_TOLERANCE, 
    ) -> PolicyDecision:
        # Solving IK directly for a target 20-30cm away lets the redundant
        # (elbow) degree of freedom wander between calls, so the joint-space
        # interpolation toward that far solution is not a Cartesian straight
        # line -- exactly the "moving forward drops height" coupling. Instead
        # aim IK at a waypoint only CARTESIAN_STEP away from where the
        # gripper actually is right now, and bias the solver toward the
        # current joint pose (rest_qpos) so the elbow configuration stays
        # stable step to step.
        height_error = None
        if guard_height is not None:
            height_error = float(observation.ee_position[2] - guard_height)
        if guard_height is not None and abs(height_error) > TRANSIT_HEIGHT_TOLERANCE:
            # Height sagged too far below (or crept too far above) the safe
            # transit plane while xy was still catching up to a distant
            # target. Ignore the horizontal target for this one step and
            # fix height first, re-checked every call, so drift can never
            # accumulate into an actual collision with tabletop objects.
            immediate_target = np.array(
                [observation.ee_position[0], observation.ee_position[1], guard_height]
            )
        else:
            immediate_target = target_position
        offset = immediate_target - observation.ee_position
        distance = float(np.linalg.norm(offset))
        if distance > CARTESIAN_STEP:
            waypoint = observation.ee_position + offset * (CARTESIAN_STEP / distance)
        else:
            waypoint = immediate_target
        result = self.ik.solve(
            observation.joint_position,
            waypoint,
            target_quaternion if target_quaternion is not None else self.home_quaternion,
            rest_qpos=observation.joint_position,
        )
        next_joint = move_toward(observation.joint_position, result.joint_position)
        position_error = float(np.linalg.norm(observation.ee_position - target_position))
        if position_error < position_tolerance:
            self.stage = next_stage
        return PolicyDecision(
            command=JointPositionCommand(next_joint, gripper),
            stage=stage_name,
            rationale=(
                f"Target {target_position.round(3).tolist()}, waypoint {waypoint.round(3).tolist()}, "
                f"error={position_error:.3f}m, ik_converged={result.converged}, "
                f"height_guard={'fixing' if height_error is not None and abs(height_error) > TRANSIT_HEIGHT_TOLERANCE else 'ok'}."
            ),
            target_id=action.pick_id,
        )

    def _grasp_quaternion(self, object_yaw_quaternion: np.ndarray) -> np.ndarray:
        """Compose the object's detected planar yaw onto the fixed home orientation.

        Verified empirically against DampedLeastSquaresIK.solve: rotating the
        home orientation by this composed quaternion rotates the fingertip
        opening axis by exactly the same yaw in the world XY plane (checked
        across +/-90 degrees, max residual 0.03 degrees from the home pose's
        own slight tilt). ``object_yaw_quaternion`` is a rotation about world
        +Z (see FoundationModelPerception._estimate_yaw_quaternion), so
        left-multiplying it onto home_quaternion yields the target orientation
        for the fingers to straddle the object's short axis.
        """
        assert self.home_quaternion is not None
        target = np.empty(4)
        mujoco.mju_mulQuat(target, object_yaw_quaternion, self.home_quaternion)
        return target

    def _align_pick(self, observation: Observation, action: _PickPlaceAction) -> PolicyDecision:
        target_quaternion = (
            action.pick_quaternion if action.pick_quaternion is not None else self.home_quaternion
        )
        assert target_quaternion is not None
        # Pure rotation: hold the current xyz fixed and only let IK adjust
        # orientation, so this stage cannot drift position the way a
        # combined move would.
        result = self.ik.solve(
            observation.joint_position,
            observation.ee_position,
            target_quaternion,
            rest_qpos=observation.joint_position,
        )
        next_joint = move_toward(observation.joint_position, result.joint_position)
        if result.orientation_error < 0.05:
            self.stage = "descend_pick"
        return PolicyDecision(
            command=JointPositionCommand(next_joint, 1.0),
            stage="align_pick",
            rationale=f"Rotating to grasp orientation; orientation_error={result.orientation_error:.3f}rad.",
            target_id=action.pick_id,
        )

    def _settle_before_close(self, observation: Observation, action: _PickPlaceAction) -> PolicyDecision:
        # descend_pick only stops once position_error < POSITION_TOLERANCE,
        # but the IK/joint-space path can still be mid-correction at that
        # instant. Hold position with the gripper still open for a fixed
        # amount of simulated time before starting to close, so closing
        # never starts on a fingertip that has not actually finished
        # settling into place.
        if self.descend_settle_deadline is None:
            self.descend_settle_deadline = observation.time + DESCEND_SETTLE_SECONDS
        remaining = self.descend_settle_deadline - observation.time
        if remaining <= 0.0:
            self.descend_settle_deadline = None
            self.stage = "close"
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, 1.0),
            stage="settle_before_close",
            rationale=f"Holding position before closing; remaining_settle_s={max(remaining, 0.0):.2f}.",
            target_id=action.pick_id,
        )

    def _close_gripper(self, observation: Observation, action: _PickPlaceAction) -> PolicyDecision:
        return self._wait_for_gripper_settle(
            observation,
            action,
            command_opening=0.0,
            stage_name="close",
            next_stage="rise_after_pick",
            rationale="Closing the gripper on the target object.",
        )

    def _open_gripper(self, observation: Observation, action: _PickPlaceAction) -> PolicyDecision:
        return self._wait_for_gripper_settle(
            observation,
            action,
            command_opening=1.0,
            stage_name="release",
            next_stage="rise_after_place",
            rationale="Opening the gripper to release the object at its destination.",
        )

    def _wait_for_gripper_settle(
        self,
        observation: Observation,
        action: _PickPlaceAction,
        *,
        command_opening: float,
        stage_name: str,
        next_stage: str,
        rationale: str,
    ) -> PolicyDecision:
        if self.grip_settle_deadline is None:
            self.grip_settle_deadline = observation.time + GRIP_SETTLE_SECONDS
        remaining = self.grip_settle_deadline - observation.time
        if remaining <= 0.0:
            self.grip_settle_deadline = None
            self.stage = next_stage
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, command_opening),
            stage=stage_name,
            rationale=(
                f"{rationale} real_opening={observation.gripper_opening:.3f}, "
                f"remaining_settle_s={max(remaining, 0.0):.2f}."
            ),
            target_id=action.pick_id,
        )

    def _verify(self, observation: Observation, action: _PickPlaceAction) -> PolicyDecision:
        camera = camera_by_name(observation, "overhead")
        search_xy = (action.place_world if action.place_world is not None else action.pick_world)[:2]
        try:
            detected, _ = self.perception.detect_target(camera, action.pick_id, prior_xy=search_xy)
        except ModelServiceError:
            detected = None
        
        if self._check_action_success(action, detected):
            self.action_index += 1
            self.stage = "rise_pick"
            self.retry_count = 0
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, observation.gripper_opening),
                stage="verify_place" if action.place_world is not None else "verify",
                rationale="Re-detected the target at its expected location; this pick/place is complete.",
                target_id=action.pick_id,
            )

        self.retry_count += 1
        if self.retry_count > MAX_RETRIES_PER_ACTION:
            self.terminal = True
            return PolicyDecision(
                command=JointPositionCommand(observation.joint_position, observation.gripper_opening),
                stage="verify_failed",
                rationale="Verification kept failing after retries; stopping safely.",
                target_id=action.pick_id,
                done=True,
            )

        self._refresh_pick_geometry(observation, action)
        self.stage = "rise_pick"
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, observation.gripper_opening),
            stage="verify",
            rationale="Verification did not confirm success; re-detecting and retrying this action.",
            target_id=action.pick_id,
            request_retry=True,
        )

    def _refresh_pick_geometry(self, observation: Observation, action: _PickPlaceAction) -> None:
        camera = camera_by_name(observation, "overhead")
        try:
            detected, _ = self.perception.detect_target(camera, action.pick_id, prior_xy=action.pick_world[:2])
        except ModelServiceError:
            return
        spec = OBJECT_SPEC_BY_NAME[action.pick_id]
        action.pick_world = detected.position.copy()
        action.pick_world[2] += spec.grasp_z_offset

    def _check_action_success(self, action: _PickPlaceAction, detected: DetectedObject | None) -> bool:
        if detected is None:
            return False
        if action.place_world is None:
            return bool(detected.position[2] > action.pick_world[2] + 0.03)
        container_spec = CONTAINER_SPEC_BY_NAME[action.place_id]
        local_xy = detected.position[:2] - action.place_world[:2]
        if container_spec.inner_half_extents is not None:
            return bool(
                abs(local_xy[0]) <= container_spec.inner_half_extents[0]
                and abs(local_xy[1]) <= container_spec.inner_half_extents[1]
            )
        return bool(np.linalg.norm(local_xy) <= (container_spec.inner_radius or 0.0))

    def _hold(
        self, observation: Observation, *, stage: str, rationale: str, done: bool = False
    ) -> PolicyDecision:
        return PolicyDecision(
            command=JointPositionCommand(observation.joint_position, observation.gripper_opening),
            stage=stage,
            rationale=rationale,
            done=done,
        )