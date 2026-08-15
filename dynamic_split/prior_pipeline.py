from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from utils.fourrc_utils import load_fourrc_prior_data

from .config import PriorConfig
from .flow import flow_path, forward_backward_consistency, load_flow, residual_flow
from .geometry import camera_flow_from_depth
from .motion import (
    AdaptiveMotionSupport,
    PoseFlowRefinement,
    adaptive_motion_support,
    depth_confidence_mask,
    refine_pose_camera_flow,
)
from .sam2_prior import PromptedSegmentationResult, Sam2PromptedSegmenter, segment_motion_components


REQUIRED_GEOMETRY_KEYS = ("image_paths", "extrinsic", "intrinsic", "depth", "depth_conf")


@dataclass(frozen=True)
class FlowObservation:
    direction: str
    target_index: int
    observed_flow: np.ndarray
    pose_camera_flow: np.ndarray
    camera_fit_valid: np.ndarray
    motion_detection_valid: np.ndarray
    confidence_threshold: float


@dataclass(frozen=True)
class MotionPass:
    support: np.ndarray
    magnitude: np.ndarray
    direction_records: tuple[dict[str, Any], ...]


def load_page4d_geometry(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"PAGE4D predictions not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_GEOMETRY_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"PAGE4D predictions are missing keys: {', '.join(missing)}")
        # Deliberately access only these arrays; world_points can be absent or corrupted.
        result = {key: np.asarray(archive[key]) for key in REQUIRED_GEOMETRY_KEYS}
    result["depth"] = np.asarray(result["depth"], dtype=np.float32)
    result["depth_conf"] = np.asarray(result["depth_conf"], dtype=np.float32)
    if result["depth"].ndim == 4 and result["depth"].shape[-1] == 1:
        result["depth"] = result["depth"][..., 0]
    if result["depth_conf"].ndim == 4 and result["depth_conf"].shape[-1] == 1:
        result["depth_conf"] = result["depth_conf"][..., 0]
    if result["depth"].ndim != 3:
        raise ValueError(f"Expected PAGE4D depth [N,H,W] or [N,H,W,1], got {result['depth'].shape}")
    if result["depth_conf"].ndim != 3:
        raise ValueError(
            f"Expected PAGE4D depth_conf [N,H,W] or [N,H,W,1], got "
            f"{result['depth_conf'].shape}"
        )
    result["intrinsic"] = np.asarray(result["intrinsic"], dtype=np.float32)
    result["extrinsic"] = np.asarray(result["extrinsic"], dtype=np.float32)
    frame_count = len(result["depth"])
    for key in REQUIRED_GEOMETRY_KEYS:
        if len(result[key]) != frame_count:
            raise ValueError(f"PAGE4D frame-count mismatch for {key}: {len(result[key])} != {frame_count}")
    if result["depth_conf"].shape != result["depth"].shape:
        raise ValueError(
            f"PAGE4D depth-confidence shape mismatch: {result['depth_conf'].shape} != "
            f"{result['depth'].shape}"
        )
    return result


def load_fourrc_geometry(path: Path) -> dict[str, np.ndarray]:
    data = load_fourrc_prior_data(path)
    return {
        "image_paths": np.asarray([f"{name}.png" for name in data.frame_names]),
        "extrinsic": data.world_to_camera,
        "intrinsic": data.intrinsics,
        "depth": data.depth,
        "depth_conf": data.depth_confidence,
    }


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


def frame_flow_specs(index: int, frame_count: int) -> list[tuple[str, int]]:
    """Return the available source-frame observations for a sequence position."""
    if index < 0 or index >= frame_count:
        raise IndexError(f"Frame index {index} is outside a sequence of {frame_count} frames")
    result: list[tuple[str, int]] = []
    if index < frame_count - 1:
        result.append(("forward", index + 1))
    if index > 0:
        result.append(("backward", index - 1))
    return result


def _load_observations(
    index: int,
    geometry: dict[str, np.ndarray],
    forward_flow_dir: Path,
    backward_flow_dir: Path,
    config: PriorConfig,
) -> list[FlowObservation]:
    frame_count, height, width = geometry["depth"].shape
    confidence_valid, confidence_threshold = depth_confidence_mask(
        geometry["depth_conf"][index], config.depth_confidence_quantile
    )
    observations: list[FlowObservation] = []
    for direction, target_index in frame_flow_specs(index, frame_count):
        if direction == "forward":
            observed_path = flow_path(forward_flow_dir, index, target_index)
            inverse_path = flow_path(backward_flow_dir, target_index, index)
        else:
            observed_path = flow_path(backward_flow_dir, index, target_index)
            inverse_path = flow_path(forward_flow_dir, target_index, index)
        observed = load_flow(observed_path, (height, width))
        inverse = load_flow(inverse_path, (height, width))
        consistent, _ = forward_backward_consistency(
            observed, inverse, config.flow_consistency_threshold
        )
        pose_flow, pose_valid = camera_flow_from_depth(
            geometry["depth"][index],
            geometry["intrinsic"][index],
            geometry["extrinsic"][index],
            geometry["intrinsic"][target_index],
            geometry["extrinsic"][target_index],
        )
        # Depth confidence protects the camera correction fit, but it must not
        # suppress motion evidence on textureless or overexposed objects.
        motion_detection_valid = consistent & pose_valid
        camera_fit_valid = motion_detection_valid & confidence_valid
        observations.append(
            FlowObservation(
                direction,
                target_index,
                observed,
                pose_flow,
                camera_fit_valid,
                motion_detection_valid,
                confidence_threshold,
            )
        )
    return observations


def _optional_number(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _direction_record(
    observation: FlowObservation,
    refinement: PoseFlowRefinement,
    support: AdaptiveMotionSupport,
) -> dict[str, Any]:
    return {
        "direction": observation.direction,
        "target_index": observation.target_index,
        "depth_confidence_threshold": _optional_number(observation.confidence_threshold),
        "valid_pixel_count": refinement.valid_pixels,
        "camera_fit_valid_pixel_count": refinement.valid_pixels,
        "motion_detection_valid_pixel_count": int(observation.motion_detection_valid.sum()),
        "low_confidence_detection_pixel_count": int(
            (observation.motion_detection_valid & ~observation.camera_fit_valid).sum()
        ),
        "correction_vector": [float(value) for value in refinement.correction],
        "correction_accepted": refinement.accepted,
        "correction_rank": refinement.equation_rank,
        "irls_iterations": refinement.iterations,
        "median_residual_before_correction": _optional_number(refinement.median_before),
        "median_residual_after_correction": _optional_number(refinement.median_after),
        "correction_status": refinement.reason,
        "residual_median": _optional_number(support.median),
        "residual_mad": _optional_number(support.mad),
        "residual_threshold": _optional_number(support.threshold),
        "residual_valid_pixel_count": support.valid_pixels,
        "motion_pixel_count": int(support.mask.sum()),
    }


def _run_motion_pass(
    observations: list[FlowObservation],
    depth: np.ndarray,
    intrinsic: np.ndarray,
    config: PriorConfig,
    exclusion_mask: np.ndarray | None = None,
) -> MotionPass:
    merged_support = np.zeros(depth.shape, dtype=bool)
    merged_magnitude = np.full(depth.shape, np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for observation in observations:
        refinement = refine_pose_camera_flow(
            observation.observed_flow,
            observation.pose_camera_flow,
            depth,
            intrinsic,
            observation.camera_fit_valid,
            exclusion_mask=exclusion_mask,
            iterations=config.irls_iterations,
            enabled=config.refine_pose_flow,
        )
        residual = residual_flow(
            observation.observed_flow,
            refinement.camera_flow,
            observation.motion_detection_valid,
        )
        support = adaptive_motion_support(
            residual, observation.motion_detection_valid, config.mad_multiplier
        )
        merged_support |= support.mask
        merged_magnitude = np.fmax(merged_magnitude, support.magnitude)
        records.append(_direction_record(observation, refinement, support))
    return MotionPass(merged_support, merged_magnitude, tuple(records))


def _segmentation_record(result: PromptedSegmentationResult) -> dict[str, Any]:
    return {
        "component_count": len(result.components),
        "sam2_acceptances": result.accepted_masks,
        "sam2_rejections": result.rejected_masks,
        "sam2_failures": result.failed_predictions,
        "residual_fallbacks": result.fallback_components,
        "sam2_pixels": int(result.sam_mask.sum()),
        "fused_pixels": int(result.fused_mask.sum()),
        "components": list(result.components),
    }


def _run_prior_core(
    frames: list[tuple[str, np.ndarray]],
    geometry: dict[str, np.ndarray],
    forward_flow_dir: Path,
    backward_flow_dir: Path,
    predictions: Path,
    output_dir: Path,
    sam2_model_cfg: str,
    sam2_checkpoint: Path,
    config: PriorConfig,
    geometry_source: str,
    image_source: str,
    device: str = "cuda",
    segmenter: Sam2PromptedSegmenter | None = None,
    raft_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config.validate()
    if not frames:
        raise ValueError("Prior generation requires at least one RGB frame")
    if len(frames) != len(geometry["depth"]):
        raise ValueError(f"Image/geometry frame mismatch: {len(frames)} != {len(geometry['depth'])}")
    expected_hw = tuple(geometry["depth"].shape[1:])
    sam2 = segmenter or Sam2PromptedSegmenter(sam2_model_cfg, sam2_checkpoint, device=device)

    frame_records: list[dict[str, Any]] = []
    for index, (name, rgb) in enumerate(frames):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape[:2] != expected_hw:
            raise ValueError(f"Image resolution mismatch for {name}: {rgb.shape[:2]} != {expected_hw}")
        observations = _load_observations(
            index, geometry, forward_flow_dir, backward_flow_dir, config
        )
        initial = _run_motion_pass(
            observations, geometry["depth"][index], geometry["intrinsic"][index], config
        )

        # SAM2 sees this frame only. Residual components provide prompts, not labels.
        sam2.set_image(rgb)
        provisional = segment_motion_components(
            initial.support,
            sam2.predictor,
            min_component_area=config.min_component_area,
            box_padding_ratio=config.sam_box_padding_ratio,
            min_component_coverage=config.sam_min_component_coverage,
        )
        refined = _run_motion_pass(
            observations,
            geometry["depth"][index],
            geometry["intrinsic"][index],
            config,
            exclusion_mask=provisional.fused_mask,
        )
        final_segmentation = segment_motion_components(
            refined.support,
            sam2.predictor,
            min_component_area=config.min_component_area,
            box_padding_ratio=config.sam_box_padding_ratio,
            min_component_coverage=config.sam_min_component_coverage,
        )

        paths = {
            "residual_support_initial": Path("residual_support_initial") / f"{name}.png",
            "residual_support_refined": Path("residual_support_refined") / f"{name}.png",
            "residual_magnitude": Path("residual_magnitude") / f"{name}.png",
            "sam2_mask": Path("sam2_masks") / f"{name}.png",
            "prior": Path("masks") / f"{name}.png",
            "overlay": Path("overlays") / f"{name}.png",
        }
        save_binary_mask(output_dir / paths["residual_support_initial"], initial.support)
        save_binary_mask(output_dir / paths["residual_support_refined"], refined.support)
        save_magnitude(output_dir / paths["residual_magnitude"], refined.magnitude)
        save_binary_mask(output_dir / paths["sam2_mask"], final_segmentation.sam_mask)
        save_binary_mask(output_dir / paths["prior"], final_segmentation.fused_mask)
        save_overlay(output_dir / paths["overlay"], rgb, final_segmentation.fused_mask)
        frame_records.append(
            {
                "index": index,
                "image_name": name,
                # Retain the established top-level key consumed by stage 3.
                "prior": str(paths["prior"]),
                "paths": {key: str(value) for key, value in paths.items()},
                "initial_motion": {
                    "motion_pixel_count": int(initial.support.sum()),
                    "directions": list(initial.direction_records),
                },
                "provisional_segmentation": _segmentation_record(provisional),
                "refined_motion": {
                    "motion_pixel_count": int(refined.support.sum()),
                    "directions": list(refined.direction_records),
                },
                "final_segmentation": _segmentation_record(final_segmentation),
            }
        )

    raft_metadata = dict(raft_metadata or {})
    manifest = {
        "version": 3,
        "method": "pose_plus_robust_flow_correction_and_per_frame_sam2",
        "geometry_source": geometry_source,
        "prediction_archive": str(predictions.resolve()),
        "images": image_source,
        "forward_flow": str(forward_flow_dir.resolve()),
        "backward_flow": str(backward_flow_dir.resolve()),
        "raft_model_weights": raft_metadata.get("model_weights", "external_or_precomputed"),
        "raft_forward_cache_dir": str(forward_flow_dir.resolve()),
        "raft_backward_cache_dir": str(backward_flow_dir.resolve()),
        "raft": raft_metadata,
        "frame_count": len(frames),
        "resolution": {"height": expected_hw[0], "width": expected_hw[1]},
        "sam2_model_cfg": sam2_model_cfg,
        "sam2_checkpoint": str(sam2_checkpoint.resolve()),
        "config": asdict(config),
        "ground_truth_used": False,
        "frames": frame_records,
    }
    if geometry_source == "page4d":
        manifest["page4d_predictions"] = str(predictions.resolve())
    elif geometry_source == "fourrc":
        manifest["fourrc_predictions"] = str(predictions.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


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
    segmenter: Sam2PromptedSegmenter | None = None,
    raft_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the prior pipeline with Page4D geometry and image files."""
    image_paths = sorted(Path(image_dir).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")
    frames = [
        (path.stem, np.asarray(Image.open(path).convert("RGB"))) for path in image_paths
    ]
    return _run_prior_core(
        frames,
        load_page4d_geometry(predictions),
        forward_flow_dir,
        backward_flow_dir,
        predictions,
        output_dir,
        sam2_model_cfg,
        sam2_checkpoint,
        config,
        "page4d",
        str(Path(image_dir).resolve()),
        device,
        segmenter,
        raft_metadata,
    )


def run_fourrc_prior_pipeline(
    predictions: Path,
    forward_flow_dir: Path,
    backward_flow_dir: Path,
    output_dir: Path,
    sam2_model_cfg: str,
    sam2_checkpoint: Path,
    config: PriorConfig,
    device: str = "cuda",
    segmenter: Sam2PromptedSegmenter | None = None,
    raft_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same prior core using images and geometry embedded in a 4RC archive."""
    data = load_fourrc_prior_data(predictions)
    frames = list(zip(data.frame_names, data.images_uint8))
    geometry = {
        "image_paths": np.asarray([f"{name}.png" for name in data.frame_names]),
        "extrinsic": data.world_to_camera,
        "intrinsic": data.intrinsics,
        "depth": data.depth,
        "depth_conf": data.depth_confidence,
    }
    return _run_prior_core(
        frames,
        geometry,
        forward_flow_dir,
        backward_flow_dir,
        predictions,
        output_dir,
        sam2_model_cfg,
        sam2_checkpoint,
        config,
        "fourrc",
        f"embedded:{Path(predictions).resolve()}",
        device,
        segmenter,
        raft_metadata,
    )
