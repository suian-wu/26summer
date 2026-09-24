from __future__ import annotations

import mujoco
import numpy as np

from graspbench.adapters import CartesianDeltaAdapter
from graspbench.types import CartesianDeltaCommand, Observation, PolicyDecision


class VLATemplate:
    """VLA action-head contract before wrapping with CartesianDeltaAdapter."""

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.instruction = task["instruction"]
        # TODO: load a small/local policy or initialize a remote inference client.
        # All additional imports stay inside the TODO implementation so the
        # original module/class structure remains unchanged.
        import base64
        import json
        import os
        import time
        import urllib.error
        import urllib.request
        from collections import deque

        import imageio.v3 as iio

        del model  # CartesianDeltaAdapter owns the numerical IK model.

        class VLAServiceError(RuntimeError):
            pass

        self._service_error_type = VLAServiceError
        self.endpoint = os.getenv("GRASPBENCH_VLA_URL", "").strip()
        self.api_key = os.getenv("GRASPBENCH_VLA_API_KEY", "").strip()
        self.model_name = os.getenv("GRASPBENCH_VLA_MODEL", "").strip()
        self.timeout_s = float(os.getenv("GRASPBENCH_VLA_TIMEOUT_S", "45"))
        self.max_chunk = max(1, int(os.getenv("GRASPBENCH_VLA_MAX_CHUNK", "16")))
        self.min_request_period_s = max(
            0.10, float(os.getenv("GRASPBENCH_VLA_REQUEST_PERIOD_S", "0.25"))
        )
        self.translation_scale = float(
            os.getenv("GRASPBENCH_VLA_TRANSLATION_SCALE", "1.0")
        )
        self.rotation_scale = float(
            os.getenv("GRASPBENCH_VLA_ROTATION_SCALE", "1.0")
        )
        self.max_translation = min(
            0.035, max(1e-4, float(os.getenv("GRASPBENCH_VLA_MAX_TRANSLATION", "0.035")))
        )
        self.max_rotation = min(
            0.15, max(1e-4, float(os.getenv("GRASPBENCH_VLA_MAX_ROTATION", "0.15")))
        )
        self.gripper_mode = os.getenv(
            "GRASPBENCH_VLA_GRIPPER_MODE", "zero_one"
        ).strip().lower()
        if self.gripper_mode not in {"zero_one", "minus_one_one"}:
            self.configuration_error = (
                "GRASPBENCH_VLA_GRIPPER_MODE must be zero_one or minus_one_one"
            )
        elif not self.endpoint:
            self.configuration_error = "GRASPBENCH_VLA_URL is not configured"
        else:
            self.configuration_error = None

        self.actions = deque()
        self.chunk_metadata = {}
        self.chunk_id = 0
        self.next_request_time = 0.0
        self.failure_count = 0
        self.max_failures = max(1, int(os.getenv("GRASPBENCH_VLA_MAX_FAILURES", "3")))
        self.terminal = False
        self.last_latency_s = None
        self.last_gripper = 1.0

        def encode_camera(camera):
            rgb_jpeg = iio.imwrite(
                "<bytes>",
                np.asarray(camera.rgb, dtype=np.uint8),
                extension=".jpg",
                quality=90,
            )
            # Metric depth is quantized to millimetres for a compact, lossless
            # payload. The service divides uint16 values by 1000 to recover m.
            depth_mm = np.clip(
                np.asarray(camera.depth, dtype=np.float64) * 1000.0,
                0.0,
                65535.0,
            ).astype(np.uint16)
            depth_png = iio.imwrite("<bytes>", depth_mm, extension=".png")
            return {
                "name": camera.name,
                "rgb_jpeg_b64": base64.b64encode(rgb_jpeg).decode("ascii"),
                "depth_u16_mm_png_b64": base64.b64encode(depth_png).decode("ascii"),
                "fovy_degrees": float(camera.fovy_degrees),
                "position": camera.position.tolist(),
                "rotation": camera.rotation.tolist(),
            }

        def parse_json_object(text):
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end < start:
                raise VLAServiceError("VLA response text contains no JSON object")
            value = json.loads(text[start : end + 1])
            if not isinstance(value, dict):
                raise VLAServiceError("VLA response must decode to a JSON object")
            return value

        def validate_action(value, observation):
            action = np.asarray(value, dtype=np.float64)
            if action.shape != (7,) or not np.all(np.isfinite(action)):
                raise VLAServiceError("Every VLA action must contain seven finite numbers")

            action = action.copy()
            action[:3] *= self.translation_scale
            action[3:6] *= self.rotation_scale
            action[:3] = np.clip(
                action[:3], -self.max_translation, self.max_translation
            )
            action[3:6] = np.clip(
                action[3:6], -self.max_rotation, self.max_rotation
            )

            if self.gripper_mode == "minus_one_one":
                if not -1.0 <= float(action[6]) <= 1.0:
                    raise VLAServiceError("VLA gripper output must be in [-1, 1]")
                action[6] = 0.5 * (action[6] + 1.0)
            elif not 0.0 <= float(action[6]) <= 1.0:
                raise VLAServiceError("VLA gripper output must be in [0, 1]")

            # Project translation onto the assignment's safe Cartesian box.
            lower = np.array([0.25, -0.32, 0.425], dtype=np.float64)
            upper = np.array([0.76, 0.32, 0.78], dtype=np.float64)
            safe_target = np.clip(observation.ee_position + action[:3], lower, upper)
            action[:3] = safe_target - observation.ee_position
            action[6] = float(np.clip(action[6], 0.0, 1.0))
            return action

        def request_action_chunk(observation):
            payload = {
                "instruction": self.instruction,
                "model": self.model_name or None,
                "observation": {
                    "time_s": float(observation.time),
                    "joint_position": observation.joint_position.tolist(),
                    "joint_velocity": observation.joint_velocity.tolist(),
                    "ee_position": observation.ee_position.tolist(),
                    "ee_quaternion_wxyz": observation.ee_quaternion.tolist(),
                    "gripper_opening": float(observation.gripper_opening),
                    "cameras": [encode_camera(camera) for camera in observation.cameras],
                },
                "action_contract": {
                    "order": [
                        "dx_m",
                        "dy_m",
                        "dz_m",
                        "rx_rad",
                        "ry_rad",
                        "rz_rad",
                        "gripper",
                    ],
                    "frame": "world",
                    "gripper_mode": self.gripper_mode,
                    "max_actions": self.max_chunk,
                    "response": {
                        "actions": [["seven finite numbers"]],
                        "stage": "short string",
                        "rationale": "short auditable string",
                        "target_id": "optional string or null",
                        "done": "boolean",
                    },
                },
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read(2000).decode("utf-8", errors="replace")
                raise VLAServiceError(
                    f"VLA endpoint returned HTTP {exc.code}: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise VLAServiceError(f"VLA endpoint failed: {exc}") from exc
            self.last_latency_s = time.perf_counter() - started

            if not isinstance(result, dict):
                raise VLAServiceError("VLA endpoint returned non-object JSON")
            if "error" in result:
                raise VLAServiceError(f"VLA endpoint returned: {result['error']}")

            # Also accept an OpenAI-compatible choices/message/content wrapper.
            if "choices" in result:
                try:
                    result = parse_json_object(
                        str(result["choices"][0]["message"]["content"])
                    )
                except (KeyError, IndexError, TypeError) as exc:
                    raise VLAServiceError(
                        "Invalid OpenAI-compatible VLA response wrapper"
                    ) from exc

            raw_actions = result.get("actions")
            if raw_actions is None and "action" in result:
                raw_actions = [result["action"]]
            if not isinstance(raw_actions, list) or not raw_actions:
                raise VLAServiceError("VLA response needs action or non-empty actions")
            if len(raw_actions) > self.max_chunk:
                raise VLAServiceError(
                    f"VLA returned {len(raw_actions)} actions; maximum is {self.max_chunk}"
                )

            # Shape/range validation occurs now; workspace projection is
            # repeated when each chunk action is executed from its live state.
            validated = [validate_action(item, observation) for item in raw_actions]
            self.actions.extend(validated)
            self.chunk_metadata = {
                "stage": str(result.get("stage", "vla_rollout"))[:80],
                "rationale": str(
                    result.get(
                        "rationale",
                        "Executing a validated action from the current VLA chunk.",
                    )
                )[:300],
                "target_id": result.get("target_id"),
                "done": bool(result.get("done", False)),
            }
            self.chunk_id += 1
            self.next_request_time = float(observation.time) + self.min_request_period_s

        self._validate_action = validate_action
        self._request_action_chunk = request_action_chunk

    def act(self, observation: Observation) -> PolicyDecision:
        # TODO: preprocess observation.cameras and proprioception exactly as the
        # selected VLA expects; map its action head to meters/radians and [0,1].
        action = np.zeros(7, dtype=np.float64)
        action[6] = float(np.clip(observation.gripper_opening, 0.0, 1.0))
        stage = "vla_rollout"
        rationale = "Holding safely while waiting for the next VLA event."
        target_id = None
        done = False
        request_retry = False

        if self.terminal:
            stage = "vla_done"
            rationale = "The VLA previously declared completion; holding the current pose."
            done = True
        elif self.configuration_error is not None:
            stage = "model_error"
            rationale = f"VLA unavailable; holding safely: {self.configuration_error}"
            done = True
            self.terminal = True
        else:
            try:
                if not self.actions and observation.time >= self.next_request_time:
                    # This is the only remote-call site. Action chunks and the
                    # request-period gate prevent calls from the 25 Hz loop.
                    self._request_action_chunk(observation)
                    self.failure_count = 0

                if self.actions:
                    action = self._validate_action(self.actions.popleft(), observation)
                    self.last_gripper = float(action[6])
                    stage = self.chunk_metadata.get("stage", "vla_rollout")
                    rationale = self.chunk_metadata.get(
                        "rationale",
                        "Executing a validated action from the current VLA chunk.",
                    )
                    raw_target_id = self.chunk_metadata.get("target_id")
                    target_id = (
                        raw_target_id
                        if isinstance(raw_target_id, str) and raw_target_id
                        else None
                    )
                    done = bool(
                        self.chunk_metadata.get("done", False) and not self.actions
                    )
                    self.terminal = done
                else:
                    action[6] = self.last_gripper
                    stage = "vla_wait"
                    rationale = (
                        "VLA request is rate-limited; holding the current Cartesian pose."
                    )
            except (self._service_error_type, ValueError, TypeError) as exc:
                self.failure_count += 1
                # Exponential simulated-time cooldown prevents an unavailable
                # endpoint from being retried at the physics rate.
                self.next_request_time = float(observation.time) + min(
                    4.0, self.min_request_period_s * (2**self.failure_count)
                )
                request_retry = self.failure_count < self.max_failures
                done = not request_retry
                self.terminal = done
                stage = "vla_retry" if request_retry else "model_error"
                rationale = (
                    f"VLA inference failed ({self.failure_count}/{self.max_failures}); "
                    f"holding safely: {exc}"
                )

        return PolicyDecision(
            command=CartesianDeltaCommand(action[:3], action[3:6], float(action[6])),
            stage=stage,
            rationale=rationale,
            target_id=target_id,
            done=done,
            request_retry=request_retry,
            debug={
                "vla_chunk_id": self.chunk_id,
                "vla_actions_remaining": len(self.actions),
                "vla_failures": self.failure_count,
                "vla_last_latency_s": self.last_latency_s,
            },
        )


class AdaptedVLATemplate(CartesianDeltaAdapter):
    """Runnable evaluator entry point: `policies.vla_template:AdaptedVLATemplate`."""

    def __init__(self) -> None:
        super().__init__(VLATemplate())
