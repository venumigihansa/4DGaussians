from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


POSE_ATOL = 5e-3
PRINCIPAL_POINT_ATOL = 0.5


@dataclass(frozen=True)
class FourRCSceneData:
    images: tuple[np.ndarray, ...]
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    frame_names: tuple[str, ...]
    init_points: np.ndarray
    init_confidence: np.ndarray
    init_index: int
    height: int
    width: int


@dataclass(frozen=True)
class FourRCPriorData:
    images_uint8: tuple[np.ndarray, ...]
    frame_names: tuple[str, ...]
    depth: np.ndarray
    depth_confidence: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    height: int
    width: int


def fourrc_frame_name(index: int) -> str:
    return f"frame_{index:06d}"


def _frame_key(index: int, field: str) -> str:
    return f"pred_{index}_{field}"


def _view_key(index: int, field: str) -> str:
    return f"view_{index}_{field}"


def _read_frame_count(archive, path: Path) -> int:
    if "n_frames" not in archive.files:
        raise KeyError(f"4RC archive is missing n_frames: {path}")
    value = np.asarray(archive["n_frames"])
    if value.ndim != 0:
        raise ValueError(f"4RC n_frames must be a scalar, got {value.shape}")
    frame_count = int(value)
    if float(value) != frame_count:
        raise ValueError(f"4RC n_frames must be an integer, got {value}")
    if frame_count < 1:
        raise ValueError(f"4RC n_frames must be positive, got {frame_count}")
    required = []
    for index in range(frame_count):
        required.extend(
            (
                _frame_key(index, "pts"),
                _frame_key(index, "conf"),
                _frame_key(index, "intrinsic"),
                _frame_key(index, "extrinsic"),
                _view_key(index, "img"),
            )
        )
    missing = [key for key in required if key not in archive.files]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise KeyError(f"4RC archive is missing required arrays: {preview}{suffix}")
    return frame_count


def _decode_image(array: np.ndarray, index: int) -> np.ndarray:
    image = np.asarray(array, dtype=np.float32)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(
            f"Expected view_{index}_img [1,3,H,W], got {image.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError(f"4RC image {index} contains non-finite values")
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < -1.1 or maximum > 1.1:
        raise ValueError(
            f"4RC image {index} must use normalized [-1,1] values, got [{minimum}, {maximum}]"
        )
    return np.clip(image[0] * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


def image_chw_to_uint8(image: np.ndarray) -> np.ndarray:
    chw = np.asarray(image, dtype=np.float32)
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"Expected image [3,H,W], got {chw.shape}")
    return np.rint(np.clip(chw.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)


def _validate_intrinsic(array: np.ndarray, index: int, height: int, width: int) -> np.ndarray:
    intrinsic = np.asarray(array, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected pred_{index}_intrinsic [3,3], got {intrinsic.shape}")
    if not np.isfinite(intrinsic).all():
        raise ValueError(f"4RC intrinsic {index} contains non-finite values")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f"4RC intrinsic {index} has non-positive focal lengths")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-5):
        raise ValueError(f"4RC intrinsic {index} has an invalid homogeneous row")
    if abs(float(intrinsic[0, 1])) > 1e-5 or abs(float(intrinsic[1, 0])) > 1e-5:
        raise ValueError(f"4RC intrinsic {index} has unsupported skew")
    expected_cx, expected_cy = width / 2.0, height / 2.0
    if (
        abs(float(intrinsic[0, 2]) - expected_cx) > PRINCIPAL_POINT_ATOL
        or abs(float(intrinsic[1, 2]) - expected_cy) > PRINCIPAL_POINT_ATOL
    ):
        raise ValueError(
            f"4RC intrinsic {index} has off-center principal point "
            f"({intrinsic[0, 2]}, {intrinsic[1, 2]}); expected ({expected_cx}, {expected_cy})"
        )
    return intrinsic


def _validate_camera_to_world(array: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    camera_to_world = np.asarray(array, dtype=np.float32)
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"Expected pred_{index}_extrinsic [4,4], got {camera_to_world.shape}")
    if not np.isfinite(camera_to_world).all():
        raise ValueError(f"4RC pose {index} contains non-finite values")
    if not np.allclose(camera_to_world[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        raise ValueError(f"4RC pose {index} has an invalid homogeneous row")
    rotation = camera_to_world[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=POSE_ATOL):
        raise ValueError(f"4RC pose {index} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > POSE_ATOL:
        raise ValueError(f"4RC pose {index} rotation determinant is {determinant}, expected 1")
    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    return camera_to_world, world_to_camera


def _read_points_and_confidence(
    archive, index: int, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(archive[_frame_key(index, "pts")], dtype=np.float32)
    confidence = np.asarray(archive[_frame_key(index, "conf")], dtype=np.float32)
    if points.shape != (1, height, width, 3):
        raise ValueError(
            f"Expected pred_{index}_pts [1,{height},{width},3], got {points.shape}"
        )
    if confidence.shape != (1, height, width):
        raise ValueError(
            f"Expected pred_{index}_conf [1,{height},{width}], got {confidence.shape}"
        )
    return points[0], confidence[0]


def world_points_to_depth(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    point_map = np.asarray(points, dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError(f"Expected world points [H,W,3], got {point_map.shape}")
    flat = point_map.reshape(-1, 3)
    flat_h = np.concatenate(
        (flat, np.ones((flat.shape[0], 1), dtype=np.float32)), axis=1
    )
    camera = flat_h @ np.asarray(world_to_camera, dtype=np.float32).T
    return camera[:, 2].reshape(point_map.shape[:2]).astype(np.float32)


def _read_common_frames(archive, path: Path):
    frame_count = _read_frame_count(archive, path)
    images = []
    intrinsics = []
    camera_to_world = []
    world_to_camera = []
    height = width = None
    for index in range(frame_count):
        image = _decode_image(archive[_view_key(index, "img")], index)
        current_height, current_width = image.shape[1:]
        if height is None:
            height, width = current_height, current_width
        elif (current_height, current_width) != (height, width):
            raise ValueError(
                f"4RC image resolution mismatch at frame {index}: "
                f"{(current_height, current_width)} != {(height, width)}"
            )
        intrinsic = _validate_intrinsic(
            archive[_frame_key(index, "intrinsic")], index, height, width
        )
        c2w, w2c = _validate_camera_to_world(
            archive[_frame_key(index, "extrinsic")], index
        )
        images.append(image)
        intrinsics.append(intrinsic)
        camera_to_world.append(c2w)
        world_to_camera.append(w2c)
    return (
        tuple(images),
        np.stack(intrinsics),
        np.stack(camera_to_world),
        np.stack(world_to_camera),
        int(height),
        int(width),
    )


def load_fourrc_scene(path: Path | str, init_frame: int = 0) -> FourRCSceneData:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"4RC archive not found: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        images, intrinsics, c2w, w2c, height, width = _read_common_frames(
            archive, archive_path
        )
        frame_count = len(images)
        if init_frame < 0 or init_frame >= frame_count:
            raise ValueError(f"fourrc_init_frame must be in [0, {frame_count - 1}]")
        points, confidence = _read_points_and_confidence(
            archive, init_frame, height, width
        )
    return FourRCSceneData(
        images=images,
        intrinsics=intrinsics,
        camera_to_world=c2w,
        world_to_camera=w2c,
        frame_names=tuple(fourrc_frame_name(index) for index in range(frame_count)),
        init_points=points,
        init_confidence=confidence,
        init_index=init_frame,
        height=height,
        width=width,
    )


def load_fourrc_prior_data(path: Path | str) -> FourRCPriorData:
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"4RC archive not found: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        images, intrinsics, c2w, w2c, height, width = _read_common_frames(
            archive, archive_path
        )
        depth = []
        confidence = []
        for index in range(len(images)):
            points, frame_confidence = _read_points_and_confidence(
                archive, index, height, width
            )
            frame_depth = world_points_to_depth(points, w2c[index])
            invalid = (
                ~np.isfinite(points).all(axis=-1)
                | ~np.isfinite(frame_depth)
                | (frame_depth <= 0)
            )
            frame_depth[invalid] = np.nan
            depth.append(frame_depth)
            confidence.append(frame_confidence)
    frame_count = len(images)
    return FourRCPriorData(
        images_uint8=tuple(image_chw_to_uint8(image) for image in images),
        frame_names=tuple(fourrc_frame_name(index) for index in range(frame_count)),
        depth=np.stack(depth).astype(np.float32),
        depth_confidence=np.stack(confidence).astype(np.float32),
        intrinsics=intrinsics,
        camera_to_world=c2w,
        world_to_camera=w2c,
        height=height,
        width=width,
    )
