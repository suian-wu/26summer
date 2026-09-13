"""Diagnostic: same as debug_check_boxes.py, but with the arm at its real
home position (as evaluate.py would actually see it), not swung out of view.
Also runs the full FoundationModelPerception.detect_scene() pipeline (not
just the raw HTTP boxes) so we see the final decoded position, not just the
box-selection stage.
"""
from __future__ import annotations

import mujoco

from graspbench.camera import camera_by_name
from graspbench.config import TaskSpec
from graspbench.env import GraspEnv
from graspbench.perception import FoundationModelPerception


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
        obs = env.reset(task)  # arm stays at its real home pose, no clearing move

        camera = camera_by_name(obs, "overhead")
        perception = FoundationModelPerception()

        # Raw box-selection stage, same as debug_check_boxes.py.
        response, _latency = perception._call_sam3(
            camera.rgb, [perception.SAM_PROMPTS["square_tray"]]
        )
        result = response["results"][0]
        print("=== raw boxes for 'yellow square tray' (arm at home pose) ===")
        for score, box in zip(result["scores"], result["boxes"]):
            frame_area = camera.rgb.shape[0] * camera.rgb.shape[1]
            area_fraction = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / frame_area
            print(f"  score={score:.3f} box={[round(v, 1) for v in box]} area_fraction={area_fraction:.1%}")

        # Full pipeline, same call path evaluate.py's student_policy uses.
        print("\n=== full detect_scene() result ===")
        detections, evidence = perception.detect_scene(camera)
        for name, detected in detections.items():
            print(f"  {name}: pos={detected.position}")
        if "square_tray" not in detections:
            print("  square_tray: NOT DETECTED (rejected somewhere in the pipeline)")

        tray_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "square_tray")
        print("\ntrue square_tray world pos:", env.data.xpos[tray_body])


if __name__ == "__main__":
    main()
