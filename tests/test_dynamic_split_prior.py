import numpy as np

from dynamic_split.sam2_prior import (
    build_component_prompt,
    connected_components,
    segment_motion_components,
)


class MockPredictor:
    def __init__(self, masks=None, fail=False):
        self.masks = list(masks or [])
        self.fail = fail
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("mock SAM2 failure")
        mask = self.masks.pop(0)
        return mask[None], np.array([0.9], dtype=np.float32), np.zeros((1, 1, 1))


def _segment(motion, predictor, minimum=16):
    return segment_motion_components(
        motion,
        predictor,
        min_component_area=minimum,
        box_padding_ratio=0.05,
        min_component_coverage=0.5,
    )


def test_component_prompt_uses_interior_point_and_clamped_border_box():
    component = np.zeros((60, 80), dtype=bool)
    component[:20, :30] = True
    prompt = build_component_prompt(component, 0.05)
    np.testing.assert_array_equal(prompt.box[:2], [0, 0])
    x, y = prompt.positive_point.astype(int)
    assert component[y, x]


def test_small_components_are_filtered():
    motion = np.zeros((30, 30), dtype=bool)
    motion[2:5, 2:5] = True
    assert connected_components(motion, min_area=10) == []


def test_prompted_mask_is_accepted_and_kept_separate_from_residual():
    motion = np.zeros((60, 60), dtype=bool)
    motion[20:30, 20:30] = True
    prediction = np.zeros_like(motion)
    prediction[15:35, 15:35] = True
    result = _segment(motion, MockPredictor([prediction]))
    assert result.accepted_masks == 1
    assert result.sam_mask.sum() == 400
    np.testing.assert_array_equal(result.fused_mask, result.sam_mask)


def test_oversized_mask_is_rejected_and_residual_is_fallback():
    motion = np.zeros((100, 100), dtype=bool)
    motion[40:55, 40:55] = True
    prediction = np.ones_like(motion)
    result = _segment(motion, MockPredictor([prediction]))
    assert result.rejected_masks == 1
    assert not result.sam_mask.any()
    np.testing.assert_array_equal(result.fused_mask, motion)


def test_mask_with_poor_component_coverage_is_rejected():
    motion = np.zeros((80, 80), dtype=bool)
    motion[20:40, 20:40] = True
    prediction = np.zeros_like(motion)
    prediction[29:32, 29:32] = True
    result = _segment(motion, MockPredictor([prediction]))
    assert result.rejected_masks == 1
    np.testing.assert_array_equal(result.fused_mask, motion)


def test_empty_prediction_and_exception_both_retain_residual():
    motion = np.zeros((40, 40), dtype=bool)
    motion[10:25, 10:25] = True
    empty = np.empty((0, 40, 40), dtype=bool)
    empty_result = _segment(motion, MockPredictor([empty]))
    failed_result = _segment(motion, MockPredictor(fail=True))
    assert empty_result.failed_predictions == 1
    assert failed_result.failed_predictions == 1
    np.testing.assert_array_equal(empty_result.fused_mask, motion)
    np.testing.assert_array_equal(failed_result.fused_mask, motion)
