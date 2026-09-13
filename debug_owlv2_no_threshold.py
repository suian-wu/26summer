"""Diagnostic: run OWLv2 with threshold=0.0 against the exact frame that
sam_server.py's live log showed it returning zero candidates for ("yellow
square tray", t2_public_002 seed=62, after the shift_start move), and print
every raw candidate score/box before any threshold cuts it. This checks
whether box_threshold=0.30 (borrowed from Grounding DINO's tuning) is simply
too strict for OWLv2's own score distribution, or whether OWLv2 genuinely
found nothing at any score.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.ik import DampedLeastSquaresIK, move_toward
from graspbench.types import JointPositionCommand
from PIL import Image

STARTING_SHIFT_Y = 0.33
TRANSIT_HEIGHT = 0.52
CARTESIAN_STEP = 0.02
MAX_SHIFT_STEPS = 400


def _shift_like_policy(env: GraspEnv, obs):
    ik = DampedLeastSquaresIK(env.model)
    home_quaternion = obs.ee_quaternion.copy()
    shift_target = obs.ee_position.copy()
    shift_target[1] += STARTING_SHIFT_Y
    shift_target[2] = TRANSIT_HEIGHT
    q = obs.joint_position.copy()
    for _ in range(MAX_SHIFT_STEPS):
        offset = shift_target - obs.ee_position
        distance = float(np.linalg.norm(offset))
        if distance < 0.015:
            break
        waypoint = (
            obs.ee_position + offset * (CARTESIAN_STEP / distance)
            if distance > CARTESIAN_STEP
            else shift_target
        )
        result = ik.solve(q, waypoint, home_quaternion, rest_qpos=q)
        q = move_toward(q, result.joint_position)
        obs, _info = env.step(JointPositionCommand(q, 1.0))
    return obs


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owl_id = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(owl_id)
    model = Owlv2ForObjectDetection.from_pretrained(owl_id).to(device).eval()

    task = TaskSpec(
        "t2_public_002",
        "Place the apple into the square tray.",
        "apple",
        62,
        destination="square_tray",
        scene_objects=("banana", "apple"),
    )
    with GraspEnv() as env:
        obs = env.reset(task)
        obs = _shift_like_policy(env, obs)
        camera = camera_by_name(obs, "overhead")
        image = Image.fromarray(camera.rgb).convert("RGB")

        for prompt in ["yellow square tray", "purple round tray"]:
            text = prompt.strip().lower()
            inputs = processor(images=image, text=[[text]], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs, threshold=0.0, target_sizes=[image.size[::-1]]
            )[0]
            scores = results["scores"].cpu().numpy()
            boxes = results["boxes"].cpu().numpy()
            order = np.argsort(-scores)
            print(f"\n=== prompt: {prompt!r} (total candidates: {len(scores)}) ===")
            for i in order[:8]:
                box = boxes[i]
                frame_area = image.size[0] * image.size[1]
                area_fraction = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
                print(f"  score={scores[i]:.4f} box={[round(v, 1) for v in box]} area_fraction={area_fraction:.1%}")


if __name__ == "__main__":
    main()
