from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import PriorConfig
from .flow import flow_path, forward_backward_consistency, load_flow, motion_support, residual_flow
from .geometry import camera_flow_from_depth
from .sam2_prior import Sam2ProposalGenerator, fuse_motion_with_proposals


REQUIRED_GEOMETRY_KEYS = ("image_paths", "extrinsic", "intrinsic", "depth", "depth_conf")


def load_page4d_geometry(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"PAGE4D predictions not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_GEOMETRY_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"PAGE4D predictions are missing keys: {', '.join(missing)}")
        # Access only these arrays. In particular, do not read world_points.
        result = {key: np.asarray(archive[key]) for key in REQUIRED_GEOMETRY_KEYS}
    result["depth"] = np.asarray(result["depth"], dtype=np.float32).squeeze(-1)
    result["depth_conf"] = np.asarray(result["depth_conf"], dtype=np.float32)
    result["intrinsic"] = np.asarray(result["intrinsic"], dtype=np.float32)
    result["extrinsic"] = np.asarray(result["extrinsic"], dtype=np.float32)
    frame_count = len(result["depth"])
    for key in REQUIRED_GEOMETRY_KEYS:
        if len(result[key]) != frame_count:
            raise ValueError(f"PAGE4D frame-count mismatch for {key}: {len(result[key])} != {frame_count}")
    return result


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def save_magnitude(path: Path, magnitude: np.ndarray) -> None:
    finite = np.isfinite(magnitude)
    scaled = np.zeros_like(magnitude, dtype=np.uint8)
    if finite.any():
        high = max(float(np.percentile(magnitude[finite], 99)), 1e-6)
        scaled[finite] = np.clip(magnitude[finite] / high * 255, 0, 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colored).save(path)


def save_overlay(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    rgb = np.asarray(image, dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)


def build_motion_evidence(
    geometry: dict[str, np.ndarray],
    forward_flow_dir: Path,
    backward_flow_dir: Path,
    config: PriorConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frame_count, height, width = geometry["depth"].shape
    supports = [np.zeros((height, width), dtype=bool) for _ in range(frame_count)]
    magnitudes = [np.full((height, width), np.nan, dtype=np.float32) for _ in range(frame_count)]
    for index in range(frame_count - 1):
        forward = load_flow(flow_path(forward_flow_dir, index, index + 1), (height, width))
        backward = load_flow(flow_path(backward_flow_dir, index + 1, index), (height, width))
        forward_consistent, _ = forward_backward_consistency(
            forward, backward, config.flow_consistency_threshold
        )
        backward_consistent, _ = forward_backward_consistency(
            backward, forward, config.flow_consistency_threshold
        )

        camera_forward, valid_forward = camera_flow_from_depth(
            geometry["depth"][index],
            geometry["intrinsic"][index],
            geometry["extrinsic"][index],
            geometry["intrinsic"][index + 1],
            geometry["extrinsic"][index + 1],
        )
        camera_backward, valid_backward = camera_flow_from_depth(
            geometry["depth"][index + 1],
            geometry["intrinsic"][index + 1],
            geometry["extrinsic"][index + 1],
            geometry["intrinsic"][index],
            geometry["extrinsic"][index],
        )
        valid_forward &= geometry["depth_conf"][index] >= config.min_depth_confidence
        valid_backward &= geometry["depth_conf"][index + 1] >= config.min_depth_confidence
        valid_forward &= forward_consistent
        valid_backward &= backward_consistent
        residual_forward = residual_flow(forward, camera_forward, valid_forward)
        residual_backward = residual_flow(backward, camera_backward, valid_backward)
        supports[index] |= motion_support(residual_forward, config.residual_threshold, valid_forward)
        supports[index + 1] |= motion_support(residual_backward, config.residual_threshold, valid_backward)
        magnitudes[index] = np.fmax(magnitudes[index], np.linalg.norm(residual_forward, axis=-1))
        magnitudes[index + 1] = np.fmax(
            magnitudes[index + 1], np.linalg.norm(residual_backward, axis=-1)
        )
    return supports, magnitudes


def run_prior_pipeline(
    image_dir: Path,
    forward_flow_dir: Path,
    backward_flow_dir: Path,
    predictions: Path,
    output_dir: Path,
    sam2_model_cfg: str,
    sam2_checkpoint: Path,
    config: PriorConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    config.validate()
    images = sorted(image_dir.glob("*.png"))
    geometry = load_page4d_geometry(predictions)
    if len(images) != len(geometry["depth"]):
        raise ValueError(f"Image/geometry frame mismatch: {len(images)} != {len(geometry['depth'])}")
    expected_hw = tuple(geometry["depth"].shape[1:])
    supports, magnitudes = build_motion_evidence(
        geometry, forward_flow_dir, backward_flow_dir, config
    )
    generator = Sam2ProposalGenerator(sam2_model_cfg, sam2_checkpoint, device=device)
    frame_records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images):
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        if rgb.shape[:2] != expected_hw:
            raise ValueError(f"Image resolution mismatch for {image_path}: {rgb.shape[:2]} != {expected_hw}")
        proposals = generator.generate(rgb)
        fusion = fuse_motion_with_proposals(
            supports[index],
            proposals,
            config.sam_support_ratio,
            config.sam_support_pixels,
            config.min_component_area,
        )
        name = image_path.stem
        prior_rel = Path("masks") / f"{name}.png"
        save_binary_mask(output_dir / prior_rel, fusion.mask)
        save_binary_mask(output_dir / "motion_support" / f"{name}.png", supports[index])
        save_magnitude(output_dir / "residual_magnitude" / f"{name}.png", magnitudes[index])
        save_overlay(output_dir / "overlays" / f"{name}.png", rgb, fusion.mask)
        frame_records.append(
            {
                "index": index,
                "image_name": name,
                "prior": str(prior_rel),
                "accepted_sam2_proposals": fusion.accepted_proposals,
                "fallback_components": fusion.fallback_components,
                "dynamic_pixels": int(fusion.mask.sum()),
            }
        )
    manifest = {
        "version": 1,
        "images": str(image_dir.resolve()),
        "forward_flow": str(forward_flow_dir.resolve()),
        "backward_flow": str(backward_flow_dir.resolve()),
        "page4d_predictions": str(predictions.resolve()),
        "sam2_model_cfg": sam2_model_cfg,
        "sam2_checkpoint": str(sam2_checkpoint.resolve()),
        "config": config.to_dict(),
        "frames": frame_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
