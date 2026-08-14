import numpy as np

from dynamic_split.motion import (
    adaptive_motion_support,
    depth_confidence_mask,
    image_jacobian_from_depth,
    refine_pose_camera_flow,
)


def _scene(height=40, width=40):
    depth = np.full((height, width), 2.0, dtype=np.float32)
    intrinsic = np.array(
        [[100.0, 0.0, width / 2], [0.0, 100.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return depth, intrinsic


def test_image_jacobian_center_translation_and_rotation():
    depth, intrinsic = _scene(3, 3)
    intrinsic[0, 2] = intrinsic[1, 2] = 1.0
    jacobian = image_jacobian_from_depth(depth, intrinsic)
    np.testing.assert_allclose(jacobian[1, 1, :, 0], [-50.0, 0.0])
    np.testing.assert_allclose(jacobian[1, 1, :, 1], [0.0, -50.0])
    np.testing.assert_allclose(jacobian[1, 1, :, 4], [-100.0, 0.0])
    np.testing.assert_allclose(jacobian[1, 1, :, 3], [0.0, 100.0])


def test_recovers_known_six_dimensional_correction():
    depth, intrinsic = _scene()
    jacobian = image_jacobian_from_depth(depth, intrinsic)
    expected = np.array([0.01, -0.015, 0.005, 0.002, -0.003, 0.001], dtype=np.float32)
    observed = np.einsum("hwij,j->hwi", jacobian, expected)
    result = refine_pose_camera_flow(
        observed, np.zeros_like(observed), depth, intrinsic, np.ones(depth.shape, bool)
    )
    assert result.accepted
    np.testing.assert_allclose(result.correction, expected, atol=1e-5)


def test_cauchy_irls_resists_large_moving_region():
    depth, intrinsic = _scene(50, 50)
    jacobian = image_jacobian_from_depth(depth, intrinsic)
    expected = np.array([0.008, -0.004, 0.002, 0.001, -0.002, 0.0005], dtype=np.float32)
    observed = np.einsum("hwij,j->hwi", jacobian, expected)
    observed[10:25, 10:25] += np.array([80.0, -60.0], dtype=np.float32)
    result = refine_pose_camera_flow(
        observed, np.zeros_like(observed), depth, intrinsic, np.ones(depth.shape, bool)
    )
    assert result.accepted
    np.testing.assert_allclose(result.correction, expected, atol=2e-3)


def test_insufficient_support_and_rank_deficiency_fall_back():
    depth, intrinsic = _scene(30, 30)
    observed = np.ones((30, 30, 2), dtype=np.float32)
    sparse = np.zeros(depth.shape, dtype=bool)
    sparse[:10, :10] = True
    insufficient = refine_pose_camera_flow(
        observed, np.zeros_like(observed), depth, intrinsic, sparse
    )
    assert not insufficient.accepted
    assert insufficient.reason == "insufficient_equations"

    one_pixel_depth = np.ones((1, 500), dtype=np.float32)
    degenerate_intrinsic = np.eye(3, dtype=np.float32)
    degenerate = refine_pose_camera_flow(
        np.ones((1, 500, 2), dtype=np.float32),
        np.zeros((1, 500, 2), dtype=np.float32),
        one_pixel_depth,
        degenerate_intrinsic,
        np.ones((1, 500), dtype=bool),
    )
    assert not degenerate.accepted
    assert degenerate.reason == "rank_deficient"
    assert degenerate.equation_rank < 6


def test_non_improving_correction_retains_pose_flow():
    depth, intrinsic = _scene()
    baseline = np.zeros((*depth.shape, 2), dtype=np.float32)
    result = refine_pose_camera_flow(
        baseline.copy(), baseline, depth, intrinsic, np.ones(depth.shape, bool)
    )
    assert not result.accepted
    assert result.reason == "no_median_improvement"
    np.testing.assert_array_equal(result.camera_flow, baseline)


def test_adaptive_threshold_handles_zero_mad_noise_and_outlier():
    residual = np.zeros((10, 10, 2), dtype=np.float32)
    residual[0, 0, 0] = 20.0
    result = adaptive_motion_support(residual, np.ones((10, 10), bool), 5.0)
    assert result.median == 0.0
    assert result.mad == 0.0
    assert result.mask.sum() == 1

    rng = np.random.default_rng(7)
    noisy = rng.normal(0.0, 0.2, (60, 60, 2)).astype(np.float32)
    noisy[20:30, 20:30] += 8.0
    noisy_result = adaptive_motion_support(noisy, np.ones((60, 60), bool), 5.0)
    assert noisy_result.threshold > noisy_result.median
    assert noisy_result.mask[20:30, 20:30].mean() > 0.95


def test_depth_confidence_excludes_bottom_decile():
    confidence = np.arange(100, dtype=np.float32).reshape(10, 10)
    valid, threshold = depth_confidence_mask(confidence, 0.10)
    assert threshold == np.quantile(confidence, 0.10)
    assert valid.sum() == 90
