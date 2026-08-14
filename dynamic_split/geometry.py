from __future__ import annotations

import numpy as np


EPS = 1e-8


def as_homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.asarray(extrinsic, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 extrinsic, got {matrix.shape}")
    result = np.eye(4, dtype=np.float32)
    result[:3] = matrix
    return result


def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to world points using world-to-camera extrinsics."""
    depth_map = np.asarray(depth, dtype=np.float32).squeeze()
    if depth_map.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth_map.shape}")
    k = np.asarray(intrinsic, dtype=np.float32)
    if k.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic, got {k.shape}")
    world_to_camera = as_homogeneous_extrinsic(extrinsic)
    height, width = depth_map.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)
    rays = pixels @ np.linalg.inv(k).T
    camera_points = rays * depth_map.reshape(-1, 1)
    camera_h = np.concatenate(
        (camera_points, np.ones((camera_points.shape[0], 1), dtype=np.float32)), axis=1
    )
    world_h = camera_h @ np.linalg.inv(world_to_camera).T
    world = world_h[:, :3].reshape(height, width, 3)
    valid = np.isfinite(depth_map) & (depth_map > EPS) & np.isfinite(world).all(axis=-1)
    world[~valid] = np.nan
    return world.astype(np.float32), valid


def project_world_points(
    world_points: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(world_points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected world points [H,W,3], got {points.shape}")
    height, width = points.shape[:2]
    flat = points.reshape(-1, 3)
    flat_h = np.concatenate((flat, np.ones((flat.shape[0], 1), dtype=np.float32)), axis=1)
    camera = flat_h @ as_homogeneous_extrinsic(extrinsic).T
    projected = camera[:, :3] @ np.asarray(intrinsic, dtype=np.float32).T
    z = camera[:, 2]
    denom = projected[:, 2]
    u = projected[:, 0] / np.where(np.abs(denom) > EPS, denom, np.nan)
    v = projected[:, 1] / np.where(np.abs(denom) > EPS, denom, np.nan)
    pixels = np.stack((u, v), axis=-1).reshape(height, width, 2)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(pixels).all(axis=-1)
    valid &= z.reshape(height, width) > EPS
    pixels[~valid] = np.nan
    return pixels.astype(np.float32), valid


def camera_flow_from_depth(
    depth_source: np.ndarray,
    intrinsic_source: np.ndarray,
    extrinsic_source: np.ndarray,
    intrinsic_target: np.ndarray,
    extrinsic_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flow caused only by camera motion while source world points stay fixed."""
    world, source_valid = backproject_depth(depth_source, intrinsic_source, extrinsic_source)
    target_pixels, target_valid = project_world_points(world, intrinsic_target, extrinsic_target)
    height, width = source_valid.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    source_pixels = np.stack((u, v), axis=-1)
    flow = target_pixels - source_pixels
    valid = source_valid & target_valid & np.isfinite(flow).all(axis=-1)
    flow[~valid] = np.nan
    return flow.astype(np.float32), valid
