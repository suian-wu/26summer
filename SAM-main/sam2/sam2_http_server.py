"""GroundingDINO + SAM2.1 HTTP adapter for GraspBench.

The GraspBench client sends text prompts, while native SAM2 accepts geometric
prompts.  This service uses GroundingDINO to convert each text prompt into
boxes, then uses SAM2.1 to segment every detected box.  Its response matches
``FoundationModelPerception`` in ``graspbench/perception.py``.
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import threading
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


HOST = os.getenv("SAM2_HTTP_HOST", "0.0.0.0")
PORT = int(os.getenv("SAM2_HTTP_PORT", "8765"))
DEVICE = os.getenv("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "").strip()
MODEL_CONFIG = os.getenv(
    "SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"
)
DINO_MODEL_ID = os.getenv(
    "GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny"
)
BOX_THRESHOLD = float(os.getenv("GROUNDING_DINO_BOX_THRESHOLD", "0.20"))
TEXT_THRESHOLD = float(os.getenv("GROUNDING_DINO_TEXT_THRESHOLD", "0.18"))
MAX_DETECTIONS = max(1, int(os.getenv("SAM2_MAX_DETECTIONS", "8")))
MAX_PROMPTS = max(1, int(os.getenv("SAM2_MAX_PROMPTS", "16")))


if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("SAM2_DEVICE requests CUDA, but torch.cuda.is_available() is false")
if not CHECKPOINT:
    raise RuntimeError("Set SAM2_CHECKPOINT to a downloaded SAM2.1 .pt checkpoint")
if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(f"SAM2_CHECKPOINT does not exist: {CHECKPOINT}")


print(f"Loading SAM2.1 config={MODEL_CONFIG!r} checkpoint={CHECKPOINT!r} on {DEVICE}")
SAM2_MODEL = build_sam2(MODEL_CONFIG, CHECKPOINT, device=DEVICE)
SAM2_PREDICTOR = SAM2ImagePredictor(SAM2_MODEL)

print(f"Loading GroundingDINO model={DINO_MODEL_ID!r} on {DEVICE}")
DINO_PROCESSOR = AutoProcessor.from_pretrained(DINO_MODEL_ID)
DINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
DINO_MODEL = DINO_MODEL.to(DEVICE).eval()


MODEL_LOCK = threading.Lock()
app = FastAPI(title="GraspBench GroundingDINO + SAM2 adapter", version="1.0")


class InferRequest(BaseModel):
    image_jpeg_b64: str = Field(min_length=1)
    prompts: list[str] = Field(min_length=1, max_length=MAX_PROMPTS)


def _empty_result() -> dict[str, Any]:
    return {
        "scores": [],
        "boxes": [],
        "masks": {"count": 0, "packed": ""},
    }


def _detect_boxes(image: Image.Image, prompt: str) -> tuple[np.ndarray, np.ndarray]:
    # Nested text labels are the current Transformers GroundingDINO API.
    inputs = DINO_PROCESSOR(
        images=image,
        text=[[prompt]],
        return_tensors="pt",
    ).to(DEVICE)
    with torch.inference_mode():
        outputs = DINO_MODEL(**inputs)
    processed = DINO_PROCESSOR.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = processed["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = processed["scores"].detach().cpu().numpy().astype(np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    order = np.argsort(scores)[::-1][:MAX_DETECTIONS]
    boxes = boxes[order]
    scores = scores[order]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image.width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image.height - 1)
    valid = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
    return boxes[valid], scores[valid]


def _segment_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    masks: list[np.ndarray] = []
    for box in boxes:
        predicted, _sam_scores, _logits = SAM2_PREDICTOR.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        mask = np.asarray(predicted)
        if mask.ndim == 3:
            mask = mask[0]
        if mask.shape != (height, width):
            resized = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            resized = resized.resize((width, height), resample=Image.Resampling.NEAREST)
            mask = np.asarray(resized) > 0
        masks.append(mask.astype(bool))
    return np.stack(masks, axis=0) if masks else np.empty((0, height, width), bool)


def _encode_result(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    count = int(masks.shape[0])
    if boxes.shape != (count, 4) or scores.shape != (count,):
        raise ValueError(
            f"Inconsistent output: masks={masks.shape}, "
            f"boxes={boxes.shape}, scores={scores.shape}"
        )
    packed = np.packbits(masks.reshape(-1).astype(np.uint8), bitorder="little")
    return {
        "scores": scores.astype(float).tolist(),
        "boxes": boxes.astype(float).tolist(),
        "masks": {
            "count": count,
            "packed": base64.b64encode(packed.tobytes()).decode("ascii"),
        },
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "grounding-dino-sam2",
        "device": DEVICE,
        "sam2_config": MODEL_CONFIG,
        "grounding_dino_model": DINO_MODEL_ID,
    }


@app.post("/infer")
def infer(request: InferRequest) -> dict[str, Any]:
    prompts = [prompt.strip() for prompt in request.prompts]
    if any(not prompt for prompt in prompts):
        raise HTTPException(status_code=400, detail="prompts cannot contain empty strings")
    try:
        jpeg = base64.b64decode(request.image_jpeg_b64, validate=True)
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        image_array = np.asarray(image, dtype=np.uint8)
    except (binascii.Error, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_jpeg_b64: {exc}") from exc

    try:
        results = []
        with MODEL_LOCK:
            # SAM2 caches image embeddings once and reuses them for all boxes.
            with torch.inference_mode():
                SAM2_PREDICTOR.set_image(image_array)
            for prompt in prompts:
                boxes, scores = _detect_boxes(image, prompt)
                if len(boxes) == 0:
                    results.append(_empty_result())
                    continue
                with torch.inference_mode():
                    masks = _segment_boxes(boxes, image.height, image.width)
                results.append(_encode_result(boxes, scores, masks))
        return {"results": results}
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="GroundingDINO/SAM2 GPU out of memory") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    # Do not use more than one worker: each worker loads both models again.
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
