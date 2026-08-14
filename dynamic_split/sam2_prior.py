from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FusionResult:
    mask: np.ndarray
    accepted_proposals: int
    fallback_components: int


def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [labels == label for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) >= min_area]


def fuse_motion_with_proposals(
    motion_mask: np.ndarray,
    proposals: Iterable[np.ndarray | dict[str, Any]],
    support_ratio: float = 0.05,
    support_pixels: int = 64,
    min_component_area: int = 64,
) -> FusionResult:
    """Expand motion support to complete object proposals without accepting unsupported objects."""
    motion = np.asarray(motion_mask, dtype=bool)
    fused = np.zeros_like(motion)
    accepted = 0
    for proposal in proposals:
        segmentation = proposal.get("segmentation") if isinstance(proposal, dict) else proposal
        region = np.asarray(segmentation, dtype=bool)
        if region.shape != motion.shape:
            raise ValueError(f"SAM2 proposal shape {region.shape} does not match motion mask {motion.shape}")
        area = int(region.sum())
        if area == 0:
            continue
        supported = int((region & motion).sum())
        if supported >= support_pixels and supported / area >= support_ratio:
            fused |= region
            accepted += 1

    fallback = 0
    unmatched = motion & ~fused
    for component in connected_components(unmatched, min_component_area):
        fused |= component
        fallback += 1
    return FusionResult(fused, accepted, fallback)


class Sam2ProposalGenerator:
    """Lazy wrapper around SAM2's automatic mask generator."""

    def __init__(self, model_cfg: str, checkpoint: Path, device: str = "cuda"):
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 is required only for prior generation. Install requirements-dynamic-split.txt "
                "and the official SAM2 package in the preprocessing environment."
            ) from exc
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        model = build_sam2(model_cfg, str(checkpoint), device=device)
        self._generator = SAM2AutomaticMaskGenerator(model)

    def generate(self, rgb_image: np.ndarray) -> list[dict[str, Any]]:
        image = np.asarray(rgb_image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"SAM2 expects an RGB image, got {image.shape}")
        return self._generator.generate(image)
