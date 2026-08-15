from pathlib import Path

import numpy as np

from scene.dataset_readers import readFourRCSceneInfo
from utils.fourrc_utils import load_fourrc_prior_data, load_fourrc_scene


def write_fourrc_archive(path: Path, frame_count=3, height=4, width=6, mutate=None):
    intrinsic = np.array(
        [[8.0, 0.0, width / 2], [0.0, 9.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    camera_points = np.stack(
        (
            (pixel_x - intrinsic[0, 2]) / intrinsic[0, 0],
            (pixel_y - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones((height, width)),
        ),
        axis=-1,
    ).astype(np.float32)
    arrays = {"n_frames": np.asarray(frame_count)}
    for index in range(frame_count):
        camera_to_world = np.eye(4, dtype=np.float32)
        camera_to_world[0, 3] = index * 0.1
        world_points = camera_points + camera_to_world[:3, 3]
        image = np.empty((1, 3, height, width), dtype=np.float32)
        image[:, 0] = -1.0 + index / max(frame_count - 1, 1)
        image[:, 1] = 0.0
        image[:, 2] = 1.0
        arrays[f"view_{index}_img"] = image
        arrays[f"pred_{index}_pts"] = world_points[None]
        arrays[f"pred_{index}_conf"] = (
            np.arange(height * width, dtype=np.float32).reshape(1, height, width) + index
        )
        arrays[f"pred_{index}_intrinsic"] = intrinsic.copy()
        arrays[f"pred_{index}_extrinsic"] = camera_to_world
    if mutate is not None:
        mutate(arrays)
    np.savez(path, **arrays)
    return path


def test_archive_images_cameras_depth_names_and_selective_initialization(tmp_path):
    archive = write_fourrc_archive(tmp_path / "scene.npz")
    scene_data = load_fourrc_scene(archive, init_frame=1)
    assert scene_data.frame_names == ("frame_000000", "frame_000001", "frame_000002")
    assert (scene_data.height, scene_data.width) == (4, 6)
    assert scene_data.images[1].shape == (3, 4, 6)
    assert np.allclose(scene_data.images[1][0], 0.25)
    assert np.allclose(
        scene_data.world_to_camera[1] @ scene_data.camera_to_world[1], np.eye(4), atol=1e-6
    )

    prior_data = load_fourrc_prior_data(archive)
    assert prior_data.depth.shape == (3, 4, 6)
    assert np.allclose(prior_data.depth, 1.0, atol=1e-6)
    assert prior_data.images_uint8[0].dtype == np.uint8


def test_scene_uses_dense_selected_pointmap_colors_cameras_and_times(tmp_path):
    archive = write_fourrc_archive(tmp_path / "scene.npz")
    info = readFourRCSceneInfo(archive, init_frame=1)
    assert len(info.point_cloud.points) == 24
    assert np.allclose(info.point_cloud.colors[:, 0], 0.25)
    assert np.all(info.point_cloud.normals == 0)
    assert [camera.image_name for camera in info.train_cameras] == [
        "frame_000000", "frame_000001", "frame_000002"
    ]
    assert [camera.time for camera in info.train_cameras] == [0.0, 1 / 3, 2 / 3]
    expected_w2c = np.linalg.inv(load_fourrc_scene(archive).camera_to_world[1])
    assert np.allclose(info.train_cameras[1].R, expected_w2c[:3, :3].T)
    assert np.allclose(info.train_cameras[1].T, expected_w2c[:3, 3])
    assert info.test_cameras == []
    assert len(info.video_cameras) == 3


def test_confidence_quantile_holdout_and_fixed_camera_point_radius(tmp_path):
    def fixed_cameras(arrays):
        for index in range(3):
            arrays[f"pred_{index}_extrinsic"] = np.eye(4, dtype=np.float32)

    archive = write_fourrc_archive(tmp_path / "scene.npz", mutate=fixed_cameras)
    info = readFourRCSceneInfo(
        archive, confidence_quantile=0.5, holdout_stride=2
    )
    assert len(info.point_cloud.points) == 12
    assert [camera.uid for camera in info.train_cameras] == [1]
    assert [camera.uid for camera in info.test_cameras] == [0, 2]
    assert info.nerf_normalization["radius"] > 0
    retain_one = readFourRCSceneInfo(archive, holdout_stride=1)
    assert len(retain_one.train_cameras) == 1
    assert len(retain_one.test_cameras) == 2


def test_invalid_archives_fail_clearly(tmp_path):
    missing = write_fourrc_archive(
        tmp_path / "missing.npz", mutate=lambda arrays: arrays.pop("pred_1_pts")
    )
    try:
        load_fourrc_scene(missing)
    except KeyError as error:
        assert "pred_1_pts" in str(error)
    else:
        raise AssertionError("Expected a missing-key error")

    def off_center(arrays):
        arrays["pred_0_intrinsic"][0, 2] += 1.0

    malformed = write_fourrc_archive(tmp_path / "off_center.npz", mutate=off_center)
    try:
        load_fourrc_scene(malformed)
    except ValueError as error:
        assert "off-center principal point" in str(error)
    else:
        raise AssertionError("Expected an off-center-intrinsic error")

    def non_rigid(arrays):
        arrays["pred_2_extrinsic"][0, 0] = 2.0

    malformed = write_fourrc_archive(tmp_path / "non_rigid.npz", mutate=non_rigid)
    try:
        load_fourrc_scene(malformed)
    except ValueError as error:
        assert "not orthonormal" in str(error)
    else:
        raise AssertionError("Expected a non-rigid-pose error")

    def wrong_batch(arrays):
        arrays["view_0_img"] = np.repeat(arrays["view_0_img"], 2, axis=0)

    malformed = write_fourrc_archive(tmp_path / "batch.npz", mutate=wrong_batch)
    try:
        load_fourrc_scene(malformed)
    except ValueError as error:
        assert "[1,3,H,W]" in str(error)
    else:
        raise AssertionError("Expected an image-batch error")

    def inconsistent_resolution(arrays):
        arrays["view_2_img"] = np.zeros((1, 3, 5, 6), dtype=np.float32)

    malformed = write_fourrc_archive(tmp_path / "resolution.npz", mutate=inconsistent_resolution)
    try:
        load_fourrc_scene(malformed)
    except ValueError as error:
        assert "resolution mismatch" in str(error)
    else:
        raise AssertionError("Expected a resolution-mismatch error")
