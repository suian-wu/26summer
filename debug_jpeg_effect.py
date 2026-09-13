"""Diagnostic: does JPEG re-encoding (as the real HTTP pipeline does) push
the correct small square_tray box's score below box_threshold=0.30?

perception.py's _call_sam3 encodes the frame as JPEG quality=92 before
sending it to sam_server.py. debug_no_threshold.py fed the model an
uncompressed in-memory PIL image directly and found a correct small box at
score=0.312 -- just 0.012 above the 0.30 cutoff. This test re-encodes the
same frame as JPEG (matching the real pipeline exactly) and re-scores it,
to see whether compression artifacts are what pushes that borderline score
under the threshold in the real end-to-end path.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv

DINO_ID = "IDEA-Research/grounding-dino-tiny"


def score_image(model, processor, device, image: Image.Image, label: str) -> None:
    text = "yellow square tray."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=0.0, text_threshold=0.0, target_sizes=[image.size[::-1]]
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    frame_area = image.size[0] * image.size[1]
    order = np.argsort(-scores)
    print(f"=== {label} ===")
    for i in order[:5]:
        box = boxes[i]
        area_fraction = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
        print(f"  score={scores[i]:.3f} box={[round(v, 1) for v in box]} area_fraction={area_fraction:.1%}")


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
        obs = env.reset(task)
        camera = camera_by_name(obs, "overhead")
        raw_image = Image.fromarray(camera.rgb).convert("RGB")
        score_image(model, processor, device, raw_image, "uncompressed (in-memory PIL)")

        # Re-encode exactly like perception.py's _call_sam3 does before
        # sending over HTTP: JPEG quality=92, then decode back.
        buffer = io.BytesIO()
        raw_image.save(buffer, format="JPEG", quality=92)
        buffer.seek(0)
        jpeg_image = Image.open(buffer).convert("RGB")
        score_image(model, processor, device, jpeg_image, "JPEG quality=92 round-trip (real pipeline)")


if __name__ == "__main__":
    main()
