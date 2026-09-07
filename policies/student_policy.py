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

# Horizontal transit always happens at this fixed height, comfortably above
# every object and container, so it is physically impossible for a sideways
# move to sweep something off the table: only vertical stages ever change z
# near the tabletop, and only horizontal stages ever change xy near the top.
# Measured empirically with DampedLeastSquaresIK.solve: at this arm's reach,
# targets above ~0.55m stop converging once x exceeds ~0.65m (the shoulder
# runs out of vertical extension while reaching far). 0.52m stays reachable
# across the whole tabletop x range (0.30-0.70m) while remaining above every
# object and container height, so horizontal moves never clip anything.
TRANSIT_HEIGHT = 0.52
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
POSITION_TOLERANCE = 0.015
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
GRIP_SETTLE_SECONDS = 2.0
# After descend_pick reaches its position tolerance, wait this long (settled
# in place, gripper still commanded open) before starting to close, so a
# late-arriving IK correction cannot get mistaken for "close" starting too
# early on a fingertip that has not actually finished descending.
DESCEND_SETTLE_SECONDS = 0.5
MAX_RETRIES_PER_ACTION = 2


@dataclass
class _PickPlaceAction:
    pick_id: str
    place_id: str | None
    pick_world: np.ndarray
    place_world: np.ndarray | None


class StudentPolicy:
    """SAM candidates + VLM grounding/planning, numerical IK, explicit pick/place state machine."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.task = task
        self.model = model
        self.ik = DampedLeastSquaresIK(model)
        self.perception = FoundationModelPerception()
        self.vlm = OpenAICompatibleVLM()

        self.home_quaternion: np.ndarray | None = None
        self.actions: list[_PickPlaceAction] | None = None
        self.action_index = 0
        self.stage = "init"
        self.grip_settle_deadline: float | None = None
        self.descend_settle_deadline: float | None = None
        self.retry_count = 0
        self.terminal = False

    def act(self, observation: Observation) -> PolicyDecision:
        if self.home_quaternion is None:
            # Every waypoint reuses this fixed downward orientation; HOME_Q
            # already keeps the gripper pointed at the tabletop.
            self.home_quaternion = observation.ee_quaternion.copy()

        if self.terminal:
            return self._hold(observation, stage="terminal", rationale="Episode already finished.")

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
            transit_target = np.array([action.pick_world[0], action.pick_world[1], TRANSIT_HEIGHT])
            return self._move_to(
                observation, transit_target, gripper=1.0, next_stage="descend_pick",
                stage_name="transit_pick", action=action,
            )
        if self.stage == "descend_pick":
            # xy is already aligned with the target; only z changes here.
            return self._move_to(
                observation, action.pick_world, gripper=1.0, next_stage="settle_before_close",
                stage_name="descend_pick", action=action,
            )
        if self.stage == "settle_before_close":
            return self._settle_before_close(observation, action)
        if self.stage == "close":
            return self._close_gripper(observation, action)
        if self.stage == "rise_after_pick":
            rise_target = np.array([action.pick_world[0], action.pick_world[1], TRANSIT_HEIGHT])
            next_stage = "transit_place" if action.place_world is not None else "verify"
            return self._move_to(
                observation, rise_target, gripper=0.0, next_stage=next_stage,
                stage_name="lift", action=action,
            )
        if self.stage == "transit_place":
            assert action.place_world is not None
            transit_target = np.array([action.place_world[0], action.place_world[1], TRANSIT_HEIGHT])
            return self._move_to(
                observation, transit_target, gripper=0.0, next_stage="descend_place",
                stage_name="transit_place", action=action,
            )
        if self.stage == "descend_place":
            assert action.place_world is not None
            return self._move_to(
                observation, action.place_world, gripper=0.0, next_stage="release",
                stage_name="descend_place", action=action,
            )
        if self.stage == "release":
            return self._open_gripper(observation, action)
        if self.stage == "rise_after_place":
            assert action.place_world is not None
            rise_target = np.array([action.place_world[0], action.place_world[1], TRANSIT_HEIGHT])
            return self._move_to(
                observation, rise_target, gripper=1.0, next_stage="verify",
                stage_name="rise_after_place", action=action,
            )
        if self.stage == "verify":
            return self._verify(observation, action)

        raise RuntimeError(f"Unhandled stage: {self.stage!r}")

    def _ensure_plan(self, observation: Observation) -> None:
        if self.actions is not None:
            return
        camera = camera_by_name(observation, "overhead")
        detections, evidence = self.perception.detect_scene(camera)
        plan = self.vlm.plan_task(self.task["instruction"], camera, detections, evidence)

        actions = []
        for planned in plan.actions:
            pick_spec = OBJECT_SPEC_BY_NAME[planned.pick_id]
            pick_world = detections[planned.pick_id].position.copy()
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

            place_world = None
            if planned.place_id is not None:
                container_position = detections[planned.place_id].position.copy()
                place_world = container_position + np.array(
                    [0.0, 0.0, pick_spec.half_height + PLACE_DROP_GAP]
                )
            actions.append(_PickPlaceAction(planned.pick_id, planned.place_id, pick_world, place_world))

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
    ) -> PolicyDecision:
        # Solving IK directly for a target 20-30cm away lets the redundant
        # (elbow) degree of freedom wander between calls, so the joint-space
        # interpolation toward that far solution is not a Cartesian straight
        # line -- exactly the "moving forward drops height" coupling. Instead
        # aim IK at a waypoint only CARTESIAN_STEP away from where the
        # gripper actually is right now, and bias the solver toward the
        # current joint pose (rest_qpos) so the elbow configuration stays
        # stable step to step.
        offset = target_position - observation.ee_position
        distance = float(np.linalg.norm(offset))
        if distance > CARTESIAN_STEP:
            waypoint = observation.ee_position + offset * (CARTESIAN_STEP / distance)
        else:
            waypoint = target_position
        result = self.ik.solve(
            observation.joint_position,
            waypoint,
            self.home_quaternion,
            rest_qpos=observation.joint_position,
        )
        next_joint = move_toward(observation.joint_position, result.joint_position)
        position_error = float(np.linalg.norm(observation.ee_position - target_position))
        if position_error < POSITION_TOLERANCE:
            self.stage = next_stage
        return PolicyDecision(
            command=JointPositionCommand(next_joint, gripper),
            stage=stage_name,
            rationale=(
                f"Target {target_position.round(3).tolist()}, waypoint {waypoint.round(3).tolist()}, "
                f"error={position_error:.3f}m, ik_converged={result.converged}."
            ),
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