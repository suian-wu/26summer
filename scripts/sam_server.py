"""Local Grounded-SAM adapter exposing the SAM3 HTTP protocol.

Not part of the graded student policy. Runs as an independent, GPU-resident
process; graspbench.perception.FoundationModelPerception talks to it over
plain HTTP exactly as it would talk to any other SAM3-compatible service.
Grounding DINO turns each text prompt into candidate boxes; SAM2 turns each
box into a pixel-accurate mask.

Protocol (must match graspbench.perception.FoundationModelPerception):
    GET  /healthz -> {"ok": true}
    POST /infer   body: {"image_jpeg_b64": str, "prompts": [str, ...]}
                  -> {"results": [{"scores": [...], "boxes": [[x1,y1,x2,y2],...],
                                    "masks": {"count": N, "packed": base64}}, ...]}
                  len(results) == len(prompts), same order, one result per prompt.

Usage:
    python scripts/sam_server.py \
        --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch
from PIL import Image

_MODELS: dict[str, Any] = {}


def _load_models(device: str, *, sam2_checkpoint: str, sam2_config: str) -> None:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    dino_id = "IDEA-Research/grounding-dino-tiny"
    _MODELS["dino_processor"] = AutoProcessor.from_pretrained(dino_id)
    _MODELS["dino_model"] = (
        AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()
    )
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    _MODELS["sam2_predictor"] = SAM2ImagePredictor(sam2_model)
    _MODELS["device"] = device


def _detect_boxes(
    image: Image.Image, prompt: str, *, box_threshold: float, text_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes_xyxy, scores) for one text prompt on the current frame."""
    processor = _MODELS["dino_processor"]
    model = _MODELS["dino_model"]
    device = _MODELS["device"]
    # Grounding DINO expects lowercase, period-terminated phrases.
    text = prompt.strip().lower()
    if not text.endswith("."):
        text += "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = results["boxes"].cpu().numpy().astype(np.float64)
    scores = results["scores"].cpu().numpy().astype(np.float64)
    return boxes, scores


def _segment_boxes(image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Return a (count, H, W) bool mask array for the given boxes."""
    predictor = _MODELS["sam2_predictor"]
    if len(boxes_xyxy) == 0:
        height, width = image_rgb.shape[:2]
        return np.zeros((0, height, width), dtype=bool)
    predictor.set_image(image_rgb)
    with torch.no_grad():
        masks, _, _ = predictor.predict(box=boxes_xyxy, multimask_output=False)
    # predictor.predict returns (count, 1, H, W) for a batch of boxes, or
    # (1, H, W) for a single box; normalize to (count, H, W).
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    elif masks.ndim == 3 and len(boxes_xyxy) == 1:
        masks = masks[:1] if masks.shape[0] != 1 else masks
    return masks.astype(bool)


def _pack_masks(masks: np.ndarray) -> dict[str, Any]:
    """Encode a (count, H, W) bool array the way perception.py expects to decode it."""
    count = int(masks.shape[0])
    bits = masks.astype(np.uint8).reshape(-1)
    packed = np.packbits(bits, bitorder="little")
    return {"count": count, "packed": base64.b64encode(packed.tobytes()).decode("ascii")}


def infer(image_jpeg_b64: str, prompts: list[str]) -> dict[str, Any]:
    image_bytes = base64.b64decode(image_jpeg_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)

    results = []
    for prompt in prompts:
        boxes, scores = _detect_boxes(
            image, prompt, box_threshold=0.30, text_threshold=0.25
        )
        masks = _segment_boxes(image_rgb, boxes)
        results.append(
            {
                "scores": scores.tolist(),
                "boxes": boxes.tolist(),
                "masks": _pack_masks(masks),
            }
        )
    return {"results": results}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        if self.path != "/infer":
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            image_jpeg_b64 = str(payload["image_jpeg_b64"])
            prompts = [str(item) for item in payload["prompts"]]
            response = infer(image_jpeg_b64, prompts)
        except Exception as exc:  # noqa: BLE001 - report to caller, keep server alive
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"[sam_server] {self.address_string()} {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Grounded-SAM adapter for GRASPBENCH_SAM3_URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sam2-checkpoint",
        required=True,
        help="Path to a downloaded SAM2 checkpoint, e.g. sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--sam2-config",
        required=True,
        help="SAM2 model config name/path bundled with the sam2 package, "
        "e.g. configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    args = parser.parse_args()

    print(f"[sam_server] loading models on {args.device} ...")
    _load_models(args.device, sam2_checkpoint=args.sam2_checkpoint, sam2_config=args.sam2_config)
    print("[sam_server] models ready")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam_server] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
