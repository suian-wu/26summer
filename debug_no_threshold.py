"""Diagnostic: dump every Grounding DINO candidate (no score threshold) for
'yellow square tray' at the arm's real home pose, and save the frame so we
can see whether the tray is actually occluded or just low-confidence.

This bypasses sam_server.py's HTTP layer and box_threshold entirely to see
what the raw model produces before any filtering.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv

DINO_ID = "IDEA-Research/grounding-dino-tiny"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()

    with GraspEnv() as env:
        task = TaskSpec(
            "t1_public_001",
            "把红色方块放进方盘。",
            "red_cube",
            31,
            destination="square_tray",
            scene_objects=("red_cube", "green_cylinder", "blue_box"),
        )
        obs = env.reset(task)  # real home pose, no clearing move
        camera = camera_by_name(obs, "overhead")
        image = Image.fromarray(camera.rgb).convert("RGB")
        image.save("runs/debug_home_pose_overhead.png")

        text = "yellow square tray."
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # threshold=0.0 so nothing is filtered -- see every raw candidate.
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, text_threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        order = np.argsort(-scores)
        print(f"Total raw candidates (no threshold): {len(scores)}")
        for i in order[:10]:
            frame_area = image.size[0] * image.size[1]
            box = boxes[i]
            area_fraction = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
            print(f"  score={scores[i]:.3f} box={[round(v, 1) for v in box]} area_fraction={area_fraction:.1%}")

        print("\nSaved overhead frame to runs/debug_home_pose_overhead.png")


if __name__ == "__main__":
    main()
