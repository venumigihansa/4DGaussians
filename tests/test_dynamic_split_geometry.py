import numpy as np

from dynamic_split.geometry import backproject_depth, camera_flow_from_depth


def test_backproject_identity_camera():
    depth = np.full((2, 3), 2.0, dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    points, valid = backproject_depth(depth, intrinsic, np.eye(4, dtype=np.float32))
    assert valid.all()
    np.testing.assert_allclose(points[1, 2], [4.0, 2.0, 2.0])


def test_identical_camera_has_zero_flow():
    depth = np.ones((3, 4), dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    flow, valid = camera_flow_from_depth(depth, intrinsic, extrinsic, intrinsic, extrinsic)
    assert valid.all()
    np.testing.assert_allclose(flow, 0.0, atol=1e-6)


def test_invalid_depth_is_invalid_flow():
    depth = np.ones((2, 2), dtype=np.float32)
    depth[0, 0] = 0.0
    intrinsic = np.eye(3, dtype=np.float32)
    flow, valid = camera_flow_from_depth(depth, intrinsic, np.eye(4), intrinsic, np.eye(4))
    assert not valid[0, 0]
    assert np.isnan(flow[0, 0]).all()


def test_camera_translation_matches_projection():
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    source = np.eye(4, dtype=np.float32)
    target = np.eye(4, dtype=np.float32)
    target[0, 3] = 1.0
    flow, valid = camera_flow_from_depth(depth, intrinsic, source, intrinsic, target)
    assert valid.all()
    np.testing.assert_allclose(flow[..., 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(flow[..., 1], 0.0, atol=1e-6)
