from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from .io import camera_frame_name
from .renderer import render_dynamic_logits, render_gaussian_membership


def write_gaussian_subset(gaussians, membership: torch.Tensor, path: Path) -> None:
    selected = membership.detach().cpu().numpy().astype(bool).reshape(-1)
    xyz = gaussians._xyz.detach().cpu().numpy()[selected]
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()[selected]
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()[selected]
    opacity = gaussians._opacity.detach().cpu().numpy()[selected]
    scale = gaussians._scaling.detach().cpu().numpy()[selected]
    rotation = gaussians._rotation.detach().cpu().numpy()[selected]
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacity, scale, rotation), axis=1)
    dtype = [(attribute, "f4") for attribute in gaussians.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def _save_gray(path: Path, image: torch.Tensor) -> np.ndarray:
    array = (image.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)
    return array


def render_split_artifacts(
    cameras: list,
    gaussians,
    pipeline,
    dynamic_logits: torch.Tensor,
    threshold: float,
    output_dir: Path,
    cam_type: str | None,
    opacity_threshold: float = 0.1,
) -> None:
    dynamic = dynamic_logits.detach().reshape(-1) > threshold
    static = ~dynamic
    overlay_frames: list[np.ndarray] = []
    with torch.no_grad():
        for camera in cameras:
            name = camera_frame_name(camera)
            logit_map, _ = render_dynamic_logits(camera, gaussians, pipeline, dynamic_logits, cam_type)
            probability = torch.sigmoid(logit_map)
            dynamic_alpha = render_gaussian_membership(camera, gaussians, pipeline, dynamic, cam_type)
            static_alpha = render_gaussian_membership(camera, gaussians, pipeline, static, cam_type)
            _save_gray(output_dir / "renders" / "soft_logits" / f"{name}.png", probability)
            _save_gray(output_dir / "renders" / "dynamic" / f"{name}.png", dynamic_alpha)
            _save_gray(output_dir / "renders" / "static" / f"{name}.png", static_alpha)
            binary = dynamic_alpha > opacity_threshold
            _save_gray(output_dir / "renders" / "binary_masks" / f"{name}.png", binary.float())
            rgb = camera.original_image[:3].permute(1, 2, 0).cpu().numpy()
            overlay = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            mask = binary.detach().cpu().numpy()
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
            overlay_frames.append(overlay)
    if overlay_frames:
        video_path = output_dir / "videos" / "static_dynamic_overlay.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, overlay_frames, fps=10, macro_block_size=1)


def save_split_arrays(dynamic_logits: torch.Tensor, threshold: float, output_dir: Path) -> torch.Tensor:
    logits = dynamic_logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
    dynamic = logits > threshold
    np.savez_compressed(
        output_dir / "gaussian_dynamic_split.npz",
        dynamic_logits=logits,
        dynamic_mask=dynamic,
        static_mask=~dynamic,
        threshold=np.float32(threshold),
    )
    return torch.from_numpy(dynamic)
