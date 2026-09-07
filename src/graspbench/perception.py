from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, ClassVar

import imageio.v3 as iio
import mujoco
import numpy as np

from .config import CONTAINER_SPECS, OBJECT_SPECS, TABLE_TOP_Z, ContainerSpec, ObjectSpec
from .types import CameraObservation, DetectedObject


class ModelServiceError(RuntimeError):
    """A model service was unavailable or returned an invalid response."""


@dataclass(frozen=True)
class SAM3DetectionDebug:
    target_id: str
    prompt: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask_pixels: int
    geometry_pixels: int
    latency_s: float
    endpoint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": "sam3",
            "target_id": self.target_id,
            "prompt": self.prompt,
            "score": self.score,
            "box_xyxy": list(self.box_xyxy),
            "mask_pixels": self.mask_pixels,
            "geometry_pixels": self.geometry_pixels,
            "latency_s": self.latency_s,
            "endpoint": self.endpoint,
        }


@dataclass(frozen=True)
class GroundingResult:
    target_id: str
    reason: str
    model: str
    latency_s: float
    raw_response: str
    service: str = "ollama"
    endpoint: str | None = None
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "model": self.model,
            "target_id": self.target_id,
            "reason": self.reason,
            "latency_s": self.latency_s,
            "raw_response": self.raw_response,
            "endpoint": self.endpoint,
            "cached": self.cached,
        }


class FoundationModelPerception:
    """SAM3 image segmentation and RGB-D geometry for the current camera frame."""

    SAM_PROMPTS: ClassVar[dict[str, str]] = {
        **{spec.name: spec.prompt for spec in OBJECT_SPECS},
        **{spec.name: spec.prompt for spec in CONTAINER_SPECS},
    }

    def __init__(
        self,
        *,
        sam3_endpoint: str | None = None,
        sam3_timeout_s: float = 30.0,
    ) -> None:
        self.sam3_endpoint = sam3_endpoint or os.getenv(
            "GRASPBENCH_SAM3_URL", "http://127.0.0.1:8765/infer"
        )
        self.sam3_timeout_s = float(sam3_timeout_s)
        self.spec_by_name: dict[str, ObjectSpec | ContainerSpec] = {
            **{spec.name: spec for spec in OBJECT_SPECS},
            **{spec.name: spec for spec in CONTAINER_SPECS},
        }

    def detect_scene(
        self, camera: CameraObservation, candidate_ids: tuple[str, ...] | None = None
    ) -> tuple[dict[str, DetectedObject], dict[str, SAM3DetectionDebug]]:
        target_ids = candidate_ids or tuple(self.SAM_PROMPTS)
        unknown = set(target_ids) - set(self.SAM_PROMPTS)
        if unknown:
            raise ModelServiceError(f"Unknown SAM3 candidate ids: {sorted(unknown)}")
        prompts = [self.SAM_PROMPTS[target_id] for target_id in target_ids]
        response, latency = self._call_sam3(camera.rgb, prompts)
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(prompts):
            raise ModelServiceError("SAM3 response does not match the requested prompt list")

        detections: dict[str, DetectedObject] = {}
        debug: dict[str, SAM3DetectionDebug] = {}
        for target_id, prompt, result in zip(target_ids, prompts, results, strict=True):
            try:
                detected, evidence = self._decode_detection(
                    camera,
                    self.spec_by_name[target_id],
                    prompt,
                    result,
                    latency_s=latency,
                    prior_xy=None,
                    surface_percentile=(
                        self.spec_by_name[target_id].surface_percentile
                        if isinstance(self.spec_by_name[target_id], ObjectSpec)
                        else 70.0
                    ),
                )
            except (TypeError, ValueError):
                continue
            detections[target_id] = detected
            debug[target_id] = evidence
        if not detections:
            raise ModelServiceError("SAM3 did not detect any tabletop candidate")
        return detections, debug

    def detect_target(
        self,
        camera: CameraObservation,
        target_id: str,
        *,
        prior_xy: np.ndarray,
        surface_percentile: float = 98.0,
    ) -> tuple[DetectedObject, SAM3DetectionDebug]:
        if target_id not in self.SAM_PROMPTS:
            raise ModelServiceError(f"Unsupported target returned by language model: {target_id}")
        prompt = self.SAM_PROMPTS[target_id]
        response, latency = self._call_sam3(camera.rgb, [prompt])
        results = response.get("results")
        if not isinstance(results, list) or len(results) != 1:
            raise ModelServiceError("SAM3 target response must contain exactly one result")
        try:
            return self._decode_detection(
                camera,
                self.spec_by_name[target_id],
                prompt,
                results[0],
                latency_s=latency,
                prior_xy=np.asarray(prior_xy, dtype=np.float64),
                surface_percentile=surface_percentile,
            )
        except (TypeError, ValueError) as exc:
            raise ModelServiceError(str(exc)) from exc

    def _call_sam3(self, rgb: np.ndarray, prompts: list[str]) -> tuple[dict[str, Any], float]:
        encoded = iio.imwrite("<bytes>", np.asarray(rgb, dtype=np.uint8), extension=".jpg", quality=92)
        payload = {
            "image_jpeg_b64": base64.b64encode(encoded).decode("ascii"),
            "prompts": prompts,
        }
        return self._post_json(self.sam3_endpoint, payload, timeout_s=self.sam3_timeout_s)

    def _decode_detection(
        self,
        camera: CameraObservation,
        spec: ObjectSpec | ContainerSpec,
        prompt: str,
        result: Any,
        *,
        latency_s: float,
        prior_xy: np.ndarray | None,
        surface_percentile: float,
    ) -> tuple[DetectedObject, SAM3DetectionDebug]:
        if not isinstance(result, dict):
            raise TypeError("SAM3 result is not an object")
        scores = np.asarray(result.get("scores", []), dtype=np.float64)
        boxes = np.asarray(result.get("boxes", []), dtype=np.float64)
        masks_payload = result.get("masks", {})
        if not isinstance(masks_payload, dict):
            raise TypeError(f"SAM3 returned invalid masks for {spec.name}")
        count = int(masks_payload.get("count", 0))
        if count <= 0 or scores.shape != (count,) or boxes.shape != (count, 4):
            raise ValueError(f"SAM3 found no valid instance for {spec.name}")
        masks = self._decode_masks(
            str(masks_payload.get("packed", "")), count, camera.rgb.shape[:2]
        )
        best = int(np.argmax(scores))
        mask = masks[best]
        rows, cols = np.nonzero(mask)
        if len(rows) < 10:
            raise ValueError(f"SAM3 mask for {spec.name} is too small")

        points = self._unproject_many(camera, cols, rows)
        # This window must cover every place an object OR a container can
        # actually be sampled, not just the object range: containers are
        # sampled at x in [0.30, 0.35] (env._sample_container_positions)
        # while objects are sampled at x in [0.54, 0.70]
        # (env._sample_placements). A previous window of x>=0.34 clipped out
        # most real container geometry (its x range mostly falls below that),
        # silently keeping only stray mask pixels that happened to fall
        # inside the window -- which produced a plausible-looking but wrong
        # centroid instead of an error. 0.20-0.75 covers both ranges with
        # margin on each side.
        keep = (
            (points[:, 0] >= 0.20)
            & (points[:, 0] <= 0.75)
            & (np.abs(points[:, 1]) <= 0.29)
            & (points[:, 2] >= 0.405)
            & (points[:, 2] <= 0.76)
        )
        if prior_xy is not None:
            keep &= np.linalg.norm(points[:, :2] - prior_xy[None, :], axis=1) <= 0.085
        points = points[keep]
        if len(points) < 10:
            raise ValueError(f"SAM3 mask for {spec.name} has no valid RGB-D geometry")

        xy = np.median(points[:, :2], axis=0)
        yaw_quaternion, elongation = self._estimate_yaw_quaternion(points[:, :2], xy)
        visible_surface_z = float(np.percentile(points[:, 2], surface_percentile))
        center_z = (
            visible_surface_z - spec.half_height
            if isinstance(spec, ObjectSpec)
            else visible_surface_z
        )
        if isinstance(spec, ObjectSpec):
            # A text prompt can segment an object of the same colour but a
            # different catalogue shape.  Its inferred centre then falls
            # implausibly below the tabletop after applying this object's
            # half-height.  Reject that alias without reading task answers.
            minimum_center_z = TABLE_TOP_Z + spec.half_height - 0.012
            if center_z < minimum_center_z:
                raise ValueError(f"SAM3 geometry is inconsistent with {spec.name}")
            if prior_xy is None and center_z > TABLE_TOP_Z + spec.half_height + 0.020:
                raise ValueError(f"SAM3 geometry is inconsistent with a resting {spec.name}")
        position = np.array([xy[0], xy[1], center_z], dtype=np.float64)
        box = tuple(float(value) for value in boxes[best])
        return (
            DetectedObject(
                name=spec.name,
                color=spec.color if isinstance(spec, ObjectSpec) else "container",
                shape=spec.shape,
                position=position,
                quaternion=yaw_quaternion,
                elongation=elongation,
            ),
            SAM3DetectionDebug(
                target_id=spec.name,
                prompt=prompt,
                score=float(scores[best]),
                box_xyxy=box,
                mask_pixels=len(rows),
                geometry_pixels=len(points),
                latency_s=latency_s,
                endpoint=self.sam3_endpoint,
            ),
        )

    @staticmethod
    def _decode_masks(packed_b64: str, count: int, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        try:
            packed = np.frombuffer(base64.b64decode(packed_b64), dtype=np.uint8)
        except ValueError as exc:
            raise ModelServiceError("SAM3 returned invalid base64 mask data") from exc
        bits = np.unpackbits(packed, bitorder="little")
        required = count * height * width
        if bits.size < required:
            raise ModelServiceError(
                f"SAM3 mask payload has {bits.size} bits; expected at least {required}"
            )
        return bits[:required].reshape(count, height, width).astype(bool)

    @staticmethod
    def _unproject_many(
        camera: CameraObservation, cols: np.ndarray, rows: np.ndarray
    ) -> np.ndarray:
        height, width = camera.depth.shape
        depth = camera.depth[rows, cols].astype(np.float64)
        focal = 0.5 * height / np.tan(np.deg2rad(camera.fovy_degrees) * 0.5)
        points_camera = np.column_stack(
            [
                (cols - (width - 1) * 0.5) * depth / focal,
                ((height - 1) * 0.5 - rows) * depth / focal,
                -depth,
            ]
        )
        return camera.position + points_camera @ camera.rotation.T

    @staticmethod
    def _estimate_yaw_quaternion(
        points_xy: np.ndarray, center_xy: np.ndarray, *, num_angles: int = 90
    ) -> tuple[np.ndarray, float]:
        """Estimate a top-down (world +Z) grasp yaw via a rotating minimum-area box.

        Unlike PCA (which is undefined for a square footprint -- both axes
        have equal variance, so its "dominant axis" is measurement noise),
        scanning candidate angles and keeping the one whose axis-aligned
        bounding box has the smallest area recovers a real edge orientation
        for both oblong footprints (a rotated blue box) and square ones (a
        rotated red cube caught on its diagonal) alike. Verified against
        ground-truth body orientation in simulation: recovers the true edge
        angle (mod its wrap period) within ~1 degree for a cube, a rectangle,
        and a circle (which correctly reports near-zero confidence).

        A square footprint only needs aligning to *some* pair of opposite
        edges (any of the 4 is equally graspable), so its angle is wrapped
        to a 90-degree period; an oblong footprint must align to its long
        axis specifically, wrapped to a 180-degree period. Which applies is
        decided from the box's own aspect ratio at the best-fit angle.

        Returns (quaternion, confidence). Confidence is the fractional
        spread between the largest and smallest scanned box area: near 0 for
        a round footprint (rotating it barely changes the bounding box, so
        the "best" angle is noise) and larger whenever some rotation clearly
        produces a tighter box (square or oblong). Callers should skip
        grasp rotation below some confidence floor.
        """
        centered = points_xy - center_xy[None, :]
        thetas = np.linspace(0.0, np.pi / 2.0, num_angles, endpoint=False)
        cos_t, sin_t = np.cos(thetas), np.sin(thetas)
        # rotated[k] = centered points expressed in the frame rotated by -thetas[k]
        rotated_x = centered[:, 0, None] * cos_t[None, :] + centered[:, 1, None] * sin_t[None, :]
        rotated_y = -centered[:, 0, None] * sin_t[None, :] + centered[:, 1, None] * cos_t[None, :]
        widths = rotated_x.max(axis=0) - rotated_x.min(axis=0)
        heights = rotated_y.max(axis=0) - rotated_y.min(axis=0)
        areas = widths * heights
        best = int(np.argmin(areas))
        max_area = float(areas.max())
        confidence = float((max_area - areas.min()) / max_area) if max_area > 1e-12 else 0.0

        long_side, short_side = max(widths[best], heights[best]), min(widths[best], heights[best])
        square_like = short_side >= 0.85 * long_side
        if square_like:
            # Any of the 4 sides is an equally good grasp face.
            wrap_period = np.pi / 2.0
            raw_yaw = float(thetas[best])
        else:
            # thetas[best] is whichever of the box's two perpendicular
            # directions is narrower; the long axis (the one the gripper
            # must align with) is 90 degrees from it when that direction
            # turned out to be the height rather than the width.
            wrap_period = np.pi
            raw_yaw = float(thetas[best]) if widths[best] >= heights[best] else float(thetas[best]) + np.pi / 2.0
        yaw = ((raw_yaw + wrap_period / 2.0) % wrap_period) - wrap_period / 2.0

        quaternion = np.empty(4)
        mujoco.mju_axisAngle2Quat(quaternion, np.array([0.0, 0.0, 1.0]), yaw)
        return quaternion, confidence

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
        if not isinstance(value, dict):
            raise TypeError("grounding response must be a JSON object")
        return value

    @staticmethod
    def _post_json(
        endpoint: str, payload: dict[str, Any], *, timeout_s: float
    ) -> tuple[dict[str, Any], float]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2_000).decode("utf-8", errors="replace")
            raise ModelServiceError(
                f"Model service {endpoint} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelServiceError(f"Model service {endpoint} failed: {exc}") from exc
        latency = time.perf_counter() - started
        if not isinstance(value, dict):
            raise ModelServiceError(f"Model service {endpoint} returned a non-object response")
        if "error" in value:
            raise ModelServiceError(f"Model service {endpoint} returned: {value['error']}")
        return value, latency
