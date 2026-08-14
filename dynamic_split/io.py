from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def camera_frame_name(camera) -> str:
    name = getattr(camera, "image_name", None)
    if name is None:
        raise ValueError("Camera has no image_name; priors cannot be aligned safely")
    return Path(str(name)).stem


def ordered_unique_cameras(scene) -> list:
    cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    by_name = {camera_frame_name(camera): camera for camera in cameras}
    return [by_name[name] for name in sorted(by_name)]


def load_prior_manifest(prior_dir: Path) -> tuple[dict, dict[str, Path]]:
    manifest_path = prior_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dynamic prior manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    mapping: dict[str, Path] = {}
    for frame in manifest.get("frames", []):
        name = Path(str(frame["image_name"])).stem
        path = prior_dir / frame["prior"]
        if name in mapping:
            raise ValueError(f"Duplicate prior for frame {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Prior mask listed in manifest is missing: {path}")
        mapping[name] = path
    if not mapping:
        raise ValueError(f"Prior manifest contains no frames: {manifest_path}")
    return manifest, mapping


def load_prior_tensor(path: Path, expected_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != expected_hw:
        raise ValueError(f"Prior resolution mismatch for {path}: {mask.shape} != {expected_hw}")
    return torch.from_numpy((mask > 127).astype(np.float32)).to(device)
