from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


MIN_BOX_PADDING = 8
MAX_BOX_PADDING = 32
MIN_BOX_CONTAINMENT = 0.90


class ImageMaskPredictor(Protocol):
    def predict(
        self,
        *,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        box: np.ndarray,
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class MotionComponent:
    mask: np.ndarray
    area: int
    box: np.ndarray
    positive_point: np.ndarray


@dataclass(frozen=True)
class PromptedSegmentationResult:
    sam_mask: np.ndarray
    fused_mask: np.ndarray
    accepted_masks: int
    rejected_masks: int
    failed_predictions: int
    fallback_components: int
    components: tuple[dict[str, Any], ...]


def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [
        labels == label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]


def build_component_prompt(component: np.ndarray, padding_ratio: float) -> MotionComponent:
    region = np.asarray(component, dtype=bool)
    if region.ndim != 2 or not region.any():
        raise ValueError("A motion component must be a non-empty 2D mask")
    height, width = region.shape
    ys, xs = np.nonzero(region)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    box_extent = max(x_max - x_min + 1, y_max - y_min + 1)
    padding = int(np.clip(round(box_extent * padding_ratio), MIN_BOX_PADDING, MAX_BOX_PADDING))
    box = np.array(
        [
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(width - 1, x_max + padding),
            min(height - 1, y_max + padding),
        ],
        dtype=np.float32,
    )

    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    _, _, _, maximum = cv2.minMaxLoc(distance)
    positive = np.array(maximum, dtype=np.float32)
    return MotionComponent(region, int(region.sum()), box, positive)


def _candidate_mask(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    masks = np.asarray(prediction)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[0] == 0 or masks.shape[1:] != shape:
        return None
    return np.asarray(masks[0], dtype=bool)


def segment_motion_components(
    motion_mask: np.ndarray,
    predictor: ImageMaskPredictor,
    *,
    min_component_area: int,
    box_padding_ratio: float,
    min_component_coverage: float,
) -> PromptedSegmentationResult:
    """Use independent residual components to prompt SAM2 object boundaries."""
    motion = np.asarray(motion_mask, dtype=bool)
    sam_union = np.zeros_like(motion)
    fallback = np.zeros_like(motion)
    accepted = rejected = failed = fallback_count = 0
    records: list[dict[str, Any]] = []

    for index, region in enumerate(connected_components(motion, min_component_area)):
        prompt = build_component_prompt(region, box_padding_ratio)
        x, y = (int(round(value)) for value in prompt.positive_point)
        record: dict[str, Any] = {
            "index": index,
            "area": prompt.area,
            "box": [float(value) for value in prompt.box],
            "positive_point": [float(value) for value in prompt.positive_point],
        }
        try:
            masks, quality, _ = predictor.predict(
                point_coords=prompt.positive_point[None],
                point_labels=np.ones(1, dtype=np.int32),
                box=prompt.box,
                multimask_output=False,
            )
            candidate = _candidate_mask(masks, motion.shape)
        except (RuntimeError, ValueError) as error:
            candidate = None
            failed += 1
            record.update({"status": "prediction_failed", "error": str(error)})

        if candidate is None:
            if record.get("status") != "prediction_failed":
                failed += 1
                record.update({"status": "invalid_prediction"})
            fallback |= prompt.mask
            fallback_count += 1
            records.append(record)
            continue

        intersection = int((candidate & prompt.mask).sum())
        coverage = intersection / prompt.area
        candidate_area = int(candidate.sum())
        x0, y0, x1, y1 = (int(round(value)) for value in prompt.box)
        box_mask = np.zeros_like(motion)
        box_mask[y0 : y1 + 1, x0 : x1 + 1] = True
        containment = float((candidate & box_mask).sum()) / max(candidate_area, 1)
        contains_point = bool(0 <= y < motion.shape[0] and 0 <= x < motion.shape[1] and candidate[y, x])
        quality_values = np.asarray(quality).reshape(-1)
        predicted_quality = float(quality_values[0]) if quality_values.size else None
        record.update(
            {
                "candidate_area": candidate_area,
                "component_coverage": float(coverage),
                "box_containment": containment,
                "contains_positive_point": contains_point,
                "predicted_quality": predicted_quality,
            }
        )

        if (
            candidate_area > 0
            and contains_point
            and coverage >= min_component_coverage
            and containment >= MIN_BOX_CONTAINMENT
        ):
            sam_union |= candidate
            accepted += 1
            record["status"] = "accepted"
        else:
            fallback |= prompt.mask
            fallback_count += 1
            rejected += 1
            record["status"] = "rejected"
        records.append(record)

    return PromptedSegmentationResult(
        sam_union,
        sam_union | fallback,
        accepted,
        rejected,
        failed,
        fallback_count,
        tuple(records),
    )


class Sam2PromptedSegmenter:
    """Lazy, per-image wrapper around SAM2ImagePredictor; no video state is used."""

    def __init__(self, model_cfg: str, checkpoint: Path, device: str = "cuda"):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 is required only for prior generation. Install requirements-dynamic-split.txt "
                "and the official SAM2 package in the preprocessing environment."
            ) from exc
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        model = build_sam2(model_cfg, str(checkpoint), device=device)
        self._predictor = SAM2ImagePredictor(model)

    def set_image(self, rgb_image: np.ndarray) -> None:
        # PIL-backed arrays can be read-only; SAM2 converts this array to a tensor.
        image = np.array(rgb_image, dtype=np.uint8, copy=True)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"SAM2 expects an RGB image, got {image.shape}")
        self._predictor.set_image(image)

    @property
    def predictor(self) -> ImageMaskPredictor:
        return self._predictor
