from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MAD_NORMAL_SCALE = 1.4826
CAUCHY_DENOMINATOR = 2.0
EPS = 1e-6


@dataclass(frozen=True)
class PoseFlowRefinement:
    camera_flow: np.ndarray
    correction: np.ndarray
    accepted: bool
    valid_pixels: int
    equation_rank: int
    iterations: int
    median_before: float | None
    median_after: float | None
    reason: str


@dataclass(frozen=True)
class AdaptiveMotionSupport:
    mask: np.ndarray
    magnitude: np.ndarray
    median: float | None
    mad: float | None
    threshold: float | None
    valid_pixels: int


def image_jacobian_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Return the 2x6 image Jacobian for a small camera twist at every pixel."""
    depth_map = np.asarray(depth, dtype=np.float64)
    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    k = np.asarray(intrinsic, dtype=np.float64)
    if depth_map.ndim != 2:
        raise ValueError(f"Expected depth [H,W], got {depth_map.shape}")
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsic [3,3], got {k.shape}")
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if abs(fx) < EPS or abs(fy) < EPS:
        raise ValueError("Focal lengths must be non-zero")

    height, width = depth_map.shape
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    x = pixel_x - cx
    y = pixel_y - cy
    inverse_depth = np.divide(
        1.0,
        depth_map,
        out=np.full_like(depth_map, np.nan),
        where=np.isfinite(depth_map) & (depth_map > EPS),
    )

    jacobian = np.empty((height, width, 2, 6), dtype=np.float64)
    jacobian[..., 0, 0] = -fx * inverse_depth
    jacobian[..., 0, 1] = 0.0
    jacobian[..., 0, 2] = x * inverse_depth
    jacobian[..., 0, 3] = x * y / fy
    jacobian[..., 0, 4] = -(fx + x * x / fx)
    jacobian[..., 0, 5] = y
    jacobian[..., 1, 0] = 0.0
    jacobian[..., 1, 1] = -fy * inverse_depth
    jacobian[..., 1, 2] = y * inverse_depth
    jacobian[..., 1, 3] = fy + y * y / fy
    jacobian[..., 1, 4] = -x * y / fx
    jacobian[..., 1, 5] = -x
    return jacobian.astype(np.float32)


def depth_confidence_mask(confidence: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    values = np.asarray(confidence, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected depth confidence [H,W], got {values.shape}")
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=bool), float("nan")
    threshold = float(np.quantile(values[finite], quantile))
    return finite & (values >= threshold), threshold


def _median_magnitude(vector_field: np.ndarray, valid: np.ndarray) -> float | None:
    magnitude = np.linalg.norm(vector_field, axis=-1)
    values = magnitude[np.asarray(valid, dtype=bool) & np.isfinite(magnitude)]
    return float(np.median(values)) if values.size else None


def refine_pose_camera_flow(
    observed_flow: np.ndarray,
    pose_camera_flow: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray,
    *,
    exclusion_mask: np.ndarray | None = None,
    iterations: int = 30,
    min_equations: int = 1_000,
    enabled: bool = True,
) -> PoseFlowRefinement:
    """Robustly fit a small 6-DoF correction around geometry-derived camera flow."""
    observed = np.asarray(observed_flow, dtype=np.float32)
    baseline = np.asarray(pose_camera_flow, dtype=np.float32)
    if observed.shape != baseline.shape or observed.ndim != 3 or observed.shape[-1] != 2:
        raise ValueError("Observed and pose camera flow must have matching [H,W,2] shapes")
    fit_valid = np.asarray(valid, dtype=bool).copy()
    fit_valid &= np.isfinite(observed).all(axis=-1) & np.isfinite(baseline).all(axis=-1)
    if exclusion_mask is not None:
        exclusion = np.asarray(exclusion_mask, dtype=bool)
        if exclusion.shape != fit_valid.shape:
            raise ValueError("exclusion_mask must match the flow resolution")
        fit_valid &= ~exclusion

    pixel_count = int(fit_valid.sum())
    zero = np.zeros(6, dtype=np.float32)
    before_field = observed - baseline
    median_before = _median_magnitude(before_field, fit_valid)
    if not enabled:
        return PoseFlowRefinement(
            baseline.copy(), zero, False, pixel_count, 0, 0, median_before, median_before, "disabled"
        )
    if pixel_count * 2 < min_equations:
        return PoseFlowRefinement(
            baseline.copy(), zero, False, pixel_count, 0, 0, median_before, median_before,
            "insufficient_equations",
        )

    jacobian = image_jacobian_from_depth(depth, intrinsic)
    fit_valid &= np.isfinite(jacobian).all(axis=(-1, -2))
    pixel_count = int(fit_valid.sum())
    if pixel_count * 2 < min_equations:
        return PoseFlowRefinement(
            baseline.copy(), zero, False, pixel_count, 0, 0, median_before, median_before,
            "insufficient_equations",
        )

    design_by_pixel = np.asarray(jacobian[fit_valid], dtype=np.float64)
    design = design_by_pixel.reshape(-1, 6)
    target_by_pixel = np.asarray(before_field[fit_valid], dtype=np.float64)
    target = target_by_pixel.reshape(-1)
    rank = int(np.linalg.matrix_rank(design))
    if rank < 6:
        return PoseFlowRefinement(
            baseline.copy(), zero, False, pixel_count, rank, 0, median_before, median_before,
            "rank_deficient",
        )

    correction = np.zeros(6, dtype=np.float64)
    completed_iterations = 0
    try:
        for completed_iterations in range(1, iterations + 1):
            error = target_by_pixel - np.einsum("nij,j->ni", design_by_pixel, correction)
            error_norm = np.linalg.norm(error, axis=1)
            center = float(np.median(error_norm))
            mad = float(np.median(np.abs(error_norm - center)))
            scale = max(MAD_NORMAL_SCALE * mad, EPS)
            weights = 1.0 / (1.0 + (error_norm / (CAUCHY_DENOMINATOR * scale)) ** 2)
            row_scale = np.repeat(np.sqrt(weights), 2)
            weighted_design = design * row_scale[:, None]
            weighted_target = target * row_scale
            updated, _, updated_rank, _ = np.linalg.lstsq(
                weighted_design, weighted_target, rcond=None
            )
            if int(updated_rank) < 6 or not np.isfinite(updated).all():
                raise np.linalg.LinAlgError("weighted system became rank deficient")
            if np.linalg.norm(updated - correction) <= EPS * (1.0 + np.linalg.norm(correction)):
                correction = updated
                break
            correction = updated
    except np.linalg.LinAlgError:
        return PoseFlowRefinement(
            baseline.copy(), zero, False, pixel_count, rank, completed_iterations,
            median_before, median_before, "solve_failed",
        )

    correction_field = np.einsum("hwij,j->hwi", jacobian, correction).astype(np.float32)
    corrected = baseline + correction_field
    median_after = _median_magnitude(observed - corrected, fit_valid)
    accepted = (
        median_before is not None
        and median_after is not None
        and np.isfinite(median_after)
        and median_after < median_before
    )
    if not accepted:
        return PoseFlowRefinement(
            baseline.copy(), zero, False, pixel_count, rank, completed_iterations,
            median_before, median_after, "no_median_improvement",
        )
    return PoseFlowRefinement(
        corrected.astype(np.float32), correction.astype(np.float32), True, pixel_count, rank,
        completed_iterations, median_before, median_after, "accepted",
    )


def adaptive_motion_support(
    residual: np.ndarray,
    valid: np.ndarray,
    mad_multiplier: float = 5.0,
) -> AdaptiveMotionSupport:
    field = np.asarray(residual, dtype=np.float32)
    validity = np.asarray(valid, dtype=bool) & np.isfinite(field).all(axis=-1)
    magnitude = np.linalg.norm(field, axis=-1).astype(np.float32)
    values = magnitude[validity]
    if values.size == 0:
        magnitude[~validity] = np.nan
        return AdaptiveMotionSupport(
            np.zeros_like(validity), magnitude, None, None, None, 0
        )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + mad_multiplier * MAD_NORMAL_SCALE * mad
    mask = validity & (magnitude > threshold)
    magnitude[~validity] = np.nan
    return AdaptiveMotionSupport(mask, magnitude, median, mad, float(threshold), int(values.size))
