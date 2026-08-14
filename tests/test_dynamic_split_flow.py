import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from dynamic_split.flow import forward_backward_consistency, load_flow, motion_support, residual_flow


def test_inverse_flows_are_consistent():
    forward = np.zeros((4, 5, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    valid, error = forward_backward_consistency(forward, backward)
    assert valid.all()
    np.testing.assert_allclose(error, 0.0)


def test_pure_camera_motion_has_zero_residual():
    observed = np.full((3, 3, 2), 2.0, dtype=np.float32)
    result = residual_flow(observed, observed.copy())
    np.testing.assert_allclose(result, 0.0)
    assert not motion_support(result, 1.0).any()


def test_residual_threshold_is_strict():
    residual = np.zeros((1, 2, 2), dtype=np.float32)
    residual[0, 0, 0] = 1.0
    residual[0, 1, 0] = 1.01
    np.testing.assert_array_equal(motion_support(residual, 1.0), [[False, True]])


def test_missing_flow_fails_clearly():
    with TemporaryDirectory() as directory:
        try:
            load_flow(Path(directory) / "missing.npy")
        except FileNotFoundError as error:
            assert "Optical flow file not found" in str(error)
        else:
            raise AssertionError("Expected a missing-flow error")
