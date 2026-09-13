"""Diagnostic: query the live sam_server.py over HTTP and inspect square_tray boxes.

Run with the real sam_server.py process already running (same as evaluate.py
would use). This does NOT bypass the server -- it hits the actual /infer
endpoint, so it tells us whether the oversized-box filter in sam_server.py
is actually active on the running process.
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request

import mujoco
import numpy as np
from PIL import Image

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec, HOME_Q
from graspbench.env import GraspEnv
from graspbench.types import JointPositionCommand

SAM3_URL = "http://127.0.0.1:8765/infer"


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
        # Swing the arm out of the overhead camera's view, same as the
        # earlier isolation test, so occlusion cannot be a factor here.
        clear_q = HOME_Q.copy()
        clear_q[0] = -2.8
        clear_q[3] = -0.3
        for _ in range(400):
            obs, _info = env.step(JointPositionCommand(clear_q, 1.0))

        camera = camera_by_name(obs, "overhead")
        buffer = io.BytesIO()
        Image.fromarray(camera.rgb).save(buffer, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        payload = json.dumps(
            {"image_jpeg_b64": image_b64, "prompts": ["yellow square tray", "purple round tray"]}
        ).encode("utf-8")
        request = urllib.request.Request(
            SAM3_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())

        frame_area = camera.rgb.shape[0] * camera.rgb.shape[1]
        for prompt, result in zip(["yellow square tray", "purple round tray"], body["results"]):
            print(f"=== prompt: {prompt!r} ===")
            boxes = np.asarray(result["boxes"])
            scores = np.asarray(result["scores"])
            if len(boxes) == 0:
                print("  NO BOXES RETURNED (server-side filter may have dropped everything)")
                continue
            for score, box in zip(scores.tolist(), boxes.round(1).tolist()):
                area_fraction = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
                print(f"  score={score:.3f} box={box} area_fraction={area_fraction:.1%}")

        tray_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "square_tray")
        print("true square_tray world pos:", env.data.xpos[tray_body])


if __name__ == "__main__":
    main()
