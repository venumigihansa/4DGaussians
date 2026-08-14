import numpy as np

from dynamic_split.sam2_prior import fuse_motion_with_proposals


def test_supported_proposal_expands_motion_to_full_region():
    motion = np.zeros((20, 20), dtype=bool)
    motion[5:9, 5:9] = True
    proposal = np.zeros_like(motion)
    proposal[3:13, 3:13] = True
    result = fuse_motion_with_proposals(motion, [proposal], 0.05, 8, 8)
    assert result.accepted_proposals == 1
    assert result.mask.sum() == 100


def test_unsupported_proposal_is_rejected():
    motion = np.zeros((20, 20), dtype=bool)
    motion[0:2, 0:2] = True
    proposal = np.zeros_like(motion)
    proposal[10:18, 10:18] = True
    result = fuse_motion_with_proposals(motion, [proposal], 0.05, 8, 8)
    assert result.accepted_proposals == 0
    assert not result.mask.any()


def test_large_unmatched_motion_component_is_retained():
    motion = np.zeros((20, 20), dtype=bool)
    motion[2:8, 2:8] = True
    result = fuse_motion_with_proposals(motion, [], min_component_area=16)
    assert result.fallback_components == 1
    np.testing.assert_array_equal(result.mask, motion)


def test_small_unmatched_component_is_removed():
    motion = np.zeros((20, 20), dtype=bool)
    motion[2:4, 2:4] = True
    result = fuse_motion_with_proposals(motion, [], min_component_area=16)
    assert result.fallback_components == 0
    assert not result.mask.any()
