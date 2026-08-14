from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


FLOW_NAME = re.compile(r"flow_(\d+)_to_(\d+)\.npy$")


def load_flow(path: Path, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Optical flow file not found: {path}")
    flow = np.load(path).astype(np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow [H,W,2] in {path}, got {flow.shape}")
    if expected_hw is not None and flow.shape[:2] != expected_hw:
        raise ValueError(f"Flow resolution mismatch in {path}: {flow.shape[:2]} != {expected_hw}")
    if not np.isfinite(flow).all():
        raise ValueError(f"Flow contains non-finite values: {path}")
    return flow


def flow_path(flow_dir: Path, source: int, target: int) -> Path:
    return flow_dir / f"flow_{source:06d}_to_{target:06d}.npy"


def bilinear_sample_flow(flow: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    x = coordinates[..., 0].astype(np.float32)
    y = coordinates[..., 1].astype(np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    sampled = cv2.remap(
        flow,
        np.nan_to_num(x, nan=-1).astype(np.float32),
        np.nan_to_num(y, nan=-1).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    valid &= np.isfinite(sampled).all(axis=-1)
    sampled[~valid] = np.nan
    return sampled.astype(np.float32), valid


def forward_backward_consistency(
    forward: np.ndarray,
    backward: np.ndarray,
    threshold: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    if forward.shape != backward.shape or forward.ndim != 3 or forward.shape[-1] != 2:
        raise ValueError("Forward and backward flow must have matching [H,W,2] shapes")
    height, width = forward.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    target = np.stack((u, v), axis=-1) + forward
    sampled_backward, sample_valid = bilinear_sample_flow(backward, target)
    error = np.linalg.norm(forward + sampled_backward, axis=-1).astype(np.float32)
    valid = sample_valid & np.isfinite(forward).all(axis=-1) & (error <= threshold)
    error[~sample_valid] = np.nan
    return valid, error


def residual_flow(observed: np.ndarray, camera: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if observed.shape != camera.shape or observed.ndim != 3 or observed.shape[-1] != 2:
        raise ValueError("Observed and camera flow must have matching [H,W,2] shapes")
    result = np.asarray(observed, dtype=np.float32) - np.asarray(camera, dtype=np.float32)
    finite = np.isfinite(result).all(axis=-1)
    if valid is not None:
        finite &= np.asarray(valid, dtype=bool)
    result[~finite] = np.nan
    return result
