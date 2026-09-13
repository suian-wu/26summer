"""Diagnostic: query the live sam_server.py over HTTP (real /infer endpoint,
same path evaluate.py uses) with the arm at its real HOME_Q pose -- no
clearing move, matching exactly what the current student_policy.py now
does. Prints every returned candidate box/score for both trays, and saves
the raw overhead frame so we can look at it directly.
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
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv

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
        obs = env.reset(task)  # arm stays at HOME_Q, no clearing move

        camera = camera_by_name(obs, "overhead")
        Image.fromarray(camera.rgb).save("runs/debug_home_pose_frame.png")

        buffer = io.BytesIO()
        Image.fromarray(camera.rgb).save(buffer, format="JPEG", quality=92)
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
                print("  NO BOXES RETURNED")
                continue
            for score, box in zip(scores.tolist(), boxes.round(1).tolist()):
                area_fraction = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
                print(f"  score={score:.3f} box={box} area_fraction={area_fraction:.1%}")

        tray_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "square_tray")
        print("\ntrue square_tray world pos:", env.data.xpos[tray_body])
        print("Saved overhead frame to runs/debug_home_pose_frame.png")


if __name__ == "__main__":
    main()
