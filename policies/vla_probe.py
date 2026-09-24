from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import imageio.v3 as iio
import mujoco
import numpy as np

from graspbench.adapters import CartesianDeltaAdapter
from graspbench.types import CartesianDeltaCommand, Observation, PolicyDecision
from graspbench.vlm import VLMConfig


@dataclass
class _VLAProbeStep:
    step: int
    latency_s: float
    raw_response: str
    parsed_action: list[float] | None
    parse_error: str | None


class VLATemplate:
    """Minimal end-to-end VLA probe: ask the VLM to emit a raw 7D action

    directly from (instruction, RGB frame, proprioception) with NO
    intermediate SAM detection and NO symbolic target ID. This is a one-shot
    feasibility probe for the experiment report's VLA-route comparison, not a
    production policy: every step is a fresh, uncached, un-post-processed
    model call, so it intentionally exposes whatever numeric/instability
    issues an off-the-shelf multimodal chat model has when asked to act as a
    VLA action head.
    """

    def reset(self, task: dict, model: mujoco.MjModel) -> None:
        self.instruction = task["instruction"]
        self.config = VLMConfig.from_env()
        self.step_count = 0
        self.log: list[_VLAProbeStep] = []

    def act(self, observation: Observation) -> PolicyDecision:
        self.step_count += 1
        camera = observation.cameras[0]
        encoded = iio.imwrite(
            "<bytes>", np.asarray(camera.rgb, dtype=np.uint8), extension=".jpg", quality=92
        )
        image_b64 = base64.b64encode(encoded).decode("ascii")

        prompt = (
            "You are the action head of a vision-language-action (VLA) robot "
            "controller. You receive the current overhead RGB image, the robot's "
            "instruction, and its proprioception. Output the next end-effector "
            "action directly as a 7-element delta: "
            "[dx, dy, dz, rx, ry, rz, gripper_opening]. dx/dy/dz are meters in the "
            "world frame (small deltas, roughly -0.03..0.03 per step). rx/ry/rz are "
            "an axis-angle rotation delta in radians (roughly -0.1..0.1 per step). "
            "gripper_opening is in [0, 1] (1=open, 0=closed). "
            "Return a JSON object with exactly two fields: action (a list of 7 "
            "floats) and reason (a short string). Do not call any tool, do not "
            "detect objects, do not return object IDs -- output raw numbers only.\n"
            f"instruction={json.dumps(self.instruction, ensure_ascii=False)}\n"
            f"ee_position={observation.ee_position.tolist()}\n"
            f"ee_quaternion={observation.ee_quaternion.tolist()}\n"
            f"gripper_opening={observation.gripper_opening}"
        )
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the requested JSON object. Use the image as evidence.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
        }

        action = np.zeros(7, dtype=np.float64)
        raw_response = ""
        parse_error = None
        latency_s = 0.0
        try:
            request = urllib.request.Request(
                self.config.chat_endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.api_key}",
                },
                method="POST",
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                value = json.load(response)
            latency_s = time.perf_counter() - started
            raw_response = str(value["choices"][0]["message"]["content"])
            start = raw_response.find("{")
            end = raw_response.rfind("}")
            parsed = json.loads(raw_response[start : end + 1])
            candidate = parsed["action"]
            if not isinstance(candidate, list) or len(candidate) != 7:
                raise ValueError(f"expected a 7-element action list, got {candidate!r}")
            action = np.asarray([float(v) for v in candidate], dtype=np.float64)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            IndexError,
        ) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

        self.log.append(
            _VLAProbeStep(
                step=self.step_count,
                latency_s=latency_s,
                raw_response=raw_response,
                parsed_action=None if parse_error else action.tolist(),
                parse_error=parse_error,
            )
        )

        return PolicyDecision(
            command=CartesianDeltaCommand(action[:3], action[3:6], float(np.clip(action[6], 0.0, 1.0))),
            stage="vla_probe",
            rationale="Raw VLA-style action head probe (see policies/vla_probe.py).",
            debug={
                "vla_probe": {
                    "latency_s": latency_s,
                    "raw_response": raw_response[:500],
                    "parse_error": parse_error,
                }
            },
        )


class AdaptedVLATemplate(CartesianDeltaAdapter):
    """Runnable evaluator entry point: `policies.vla_probe:AdaptedVLATemplate`."""

    def __init__(self) -> None:
        super().__init__(VLATemplate())
