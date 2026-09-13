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

# OWLv2's confidence scores are not on the same scale as Grounding DINO's --
# confirmed with a live no-threshold scan of a frame where DINO's cascade
# had already failed: OWLv2 located the correct square_tray box (matching
# DINO's own box on frames where DINO succeeds, ~13% of frame area) at only
# score=0.16, while DINO's typical scores for a correct box run 0.3-0.5+.
# Reusing DINO's 0.30 box_threshold for OWLv2 discarded that correct
# candidate, silently defeating the whole point of the fallback. 0.10 keeps
# it while still being comfortably above the oversized/spurious candidates
# in that same scan (all scored <=0.09).
OWLV2_BOX_THRESHOLD = 0.10


def _load_models(device: str, *, sam2_checkpoint: str, sam2_config: str) -> None:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Owlv2ForObjectDetection,
        Owlv2Processor,
    )

    # base has notably better recall than tiny on small/edge-of-frame
    # objects: verified in practice, tiny returned zero candidates for
    # "yellow square tray" in 5 of 6 Task 2 episodes even with the tray
    # fully visible and unoccluded, while round_tray (same scene, larger
    # apparent size) was detected every time.
    dino_id = "IDEA-Research/grounding-dino-base"
    _MODELS["dino_processor"] = AutoProcessor.from_pretrained(dino_id)
    _MODELS["dino_model"] = (
        AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()
    )
    # Fallback detector: even grounding-dino-base still returned zero
    # candidates for "yellow square tray" in 2 of 6 Task 2 episodes (fully
    # visible, unoccluded). OWLv2 is a CLIP-based region proposer with a
    # different failure mode (ViT patch classification vs. DINO's
    # transformer decoder queries), so it is used only as a per-prompt
    # rescue when Grounding DINO comes back empty, rather than replacing it
    # outright -- Grounding DINO remains more accurate overall.
    owl_id = "google/owlv2-base-patch16-ensemble"
    _MODELS["owl_processor"] = Owlv2Processor.from_pretrained(owl_id)
    _MODELS["owl_model"] = Owlv2ForObjectDetection.from_pretrained(owl_id).to(device).eval()
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    _MODELS["sam2_predictor"] = SAM2ImagePredictor(sam2_model)
    _MODELS["device"] = device
    print(f"[sam_server] detector backends loaded: dino={dino_id!r} owlv2={owl_id!r}")


def _detect_boxes_dino(
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


def _detect_boxes_owlv2(
    image: Image.Image, prompt: str, *, box_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes_xyxy, scores) for one text prompt, using OWLv2 as a
    rescue detector when Grounding DINO finds nothing for this prompt."""
    processor = _MODELS["owl_processor"]
    model = _MODELS["owl_model"]
    device = _MODELS["device"]
    text = prompt.strip().lower()
    inputs = processor(images=image, text=[[text]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=box_threshold, target_sizes=[image.size[::-1]]
    )[0]
    boxes = results["boxes"].cpu().numpy().astype(np.float64)
    scores = results["scores"].cpu().numpy().astype(np.float64)
    return boxes, scores


def _detect_boxes(
    image: Image.Image, prompt: str, *, box_threshold: float, text_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grounding DINO first; if it returns nothing usable for this prompt,
    retry with OWLv2 before giving up. Each prompt is decided independently,
    so a tray that DINO cannot find does not affect an object DINO found
    fine.

    Oversized-box filtering must happen *inside* this cascade, before the
    "did DINO find anything" check -- not afterward in infer(), as it used
    to be. Previously infer() filtered oversized boxes only after this
    function had already returned, so a DINO candidate that later turned
    out to be an oversized false-positive (>40% of the frame) still counted
    as "DINO succeeded" here, and OWLv2's rescue never fired. Confirmed live:
    sam_server's own log showed "dino found 1 candidate(s)" for "yellow
    square tray" while the client still received an empty box list, because
    that lone candidate was the oversized one and got dropped afterward.
    """
    boxes, scores = _detect_boxes_dino(
        image, prompt, box_threshold=box_threshold, text_threshold=text_threshold
    )
    boxes, scores = _drop_oversized_boxes(boxes, scores, image.size)
    if len(boxes) > 0:
        print(f"[sam_server] prompt={prompt!r}: dino found {len(boxes)} usable candidate(s)")
        return boxes, scores
    print(f"[sam_server] prompt={prompt!r}: dino found nothing usable, falling back to owlv2")
    boxes, scores = _detect_boxes_owlv2(image, prompt, box_threshold=OWLV2_BOX_THRESHOLD)
    boxes, scores = _drop_oversized_boxes(boxes, scores, image.size)
    print(f"[sam_server] prompt={prompt!r}: owlv2 found {len(boxes)} usable candidate(s)")
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


# Grounding DINO occasionally proposes a box spanning most of the frame
# instead of the actual object -- observed in practice: a "yellow square
# tray" box covering ~74% of a 256x256 overhead frame, engulfing the
# tabletop and part of the robot arm, while a second, only slightly
# lower-scoring candidate correctly boxed the real tray at ~13% of the
# frame. perception.py's client just takes the highest-scoring candidate,
# so an oversized top candidate silently wins even when a good one exists
# right behind it. Every real object/container in this scene occupies well
# under a third of the frame even at closest range, so dropping oversized
# proposals here -- before segmentation, before the client ever sees them --
# lets the next-best (and often correct) candidate take over automatically.
MAX_BOX_AREA_FRACTION = 0.40


def _drop_oversized_boxes(
    boxes: np.ndarray, scores: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Discard proposal boxes covering an implausible fraction of the frame."""
    if len(boxes) == 0:
        return boxes, scores
    width, height = image_size
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    area_fraction = (widths * heights) / float(width * height)
    keep = area_fraction <= MAX_BOX_AREA_FRACTION
    return boxes[keep], scores[keep]


def infer(image_jpeg_b64: str, prompts: list[str]) -> dict[str, Any]:
    image_bytes = base64.b64decode(image_jpeg_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(image)

    results = []
    for prompt in prompts:
        # Oversized-box filtering already happened inside _detect_boxes,
        # before the DINO/OWLv2 cascade decision -- doing it again here
        # would be redundant, not harmful, but keeping it in exactly one
        # place avoids the two filtering passes drifting out of sync.
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
