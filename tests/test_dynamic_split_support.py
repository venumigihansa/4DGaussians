from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dynamic_split.renderer import alpha_composite_logits
from dynamic_split.support import (
    accumulate_prior_support,
    finalize_support,
    support_classification_logits,
    support_supervision_loss,
)


def test_probe_gradients_match_analytic_alpha_transmittance():
    alpha = torch.tensor(
        [
            [[0.50, 1.00]],
            [[0.40, 0.80]],
            [[0.25, 0.60]],
        ]
    )
    probes = torch.zeros((3, 3), requires_grad=True)
    rendered = torch.stack(
        [alpha_composite_logits(probes[:, channel], alpha) for channel in range(3)]
    )
    prior = torch.tensor([[1.0, 0.0]])
    objective = (rendered[0] * prior).sum() + rendered[1].sum()
    objective.backward()

    weights = torch.tensor(
        [
            [0.50, 1.00],
            [0.40 * 0.50, 0.80 * 0.00],
            [0.25 * 0.50 * 0.60, 0.60 * 0.00 * 0.20],
        ]
    )
    torch.testing.assert_close(probes.grad[:, 0], (weights * prior).sum(dim=1))
    torch.testing.assert_close(probes.grad[:, 1], weights.sum(dim=1))
    assert probes.grad[1, 1].item() == pytest.approx(0.2)
    assert probes.grad[2, 1].item() == pytest.approx(0.075)


def test_finalize_support_mass_confidence_and_zero_visibility():
    result = finalize_support(
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([1.0, 2.0, 4.0, 0.0]),
    )
    torch.testing.assert_close(result.negative_mass, torch.tensor([0.0, 1.0, 4.0, 0.0]))
    torch.testing.assert_close(result.support_score, torch.tensor([1.0, 0.5, 0.0, 0.0]))
    # Median positive visibility is 2.
    torch.testing.assert_close(
        result.confidence, torch.tensor([1 / 3, 1 / 2, 2 / 3, 0.0])
    )
    assert result.visibility_reference == pytest.approx(2.0)


def test_finalize_support_rejects_zero_visibility_and_nonfinite_values():
    with pytest.raises(ValueError, match="zero visibility"):
        finalize_support(torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        finalize_support(torch.tensor([float("nan")]), torch.ones(1))


def test_support_loss_gradient_directions_and_threshold_calibration():
    logits = torch.tensor([[8.0], [6.0], [9.0]], requires_grad=True)
    support = torch.tensor([0.1, 0.9, 0.0])
    confidence = torch.tensor([1.0, 1.0, 0.0])
    calibrated = support_classification_logits(torch.tensor([[7.0]]), 7.0, 1.0)
    torch.testing.assert_close(torch.sigmoid(calibrated), torch.tensor([0.5]))

    loss = support_supervision_loss(logits, support, confidence, 7.0, 1.0)
    loss.backward()
    assert logits.grad[0].item() > 0  # Gradient descent decreases d_i when p_i > s_i.
    assert logits.grad[1].item() < 0  # Gradient descent increases d_i when p_i < s_i.
    assert logits.grad[2].item() == 0  # No visibility confidence means no support update.


class _Camera:
    def __init__(self, name: str, height: int = 2, width: int = 3):
        self.image_name = name
        self.image_height = height
        self.image_width = width


class _Gaussians:
    get_xyz = torch.zeros((2, 3))


def _write_mask(path: Path, shape=(2, 3)) -> None:
    Image.fromarray(np.zeros(shape, dtype=np.uint8), mode="L").save(path)


def test_accumulate_support_across_frames(tmp_path):
    paths = {}
    for name in ("a", "b"):
        path = tmp_path / f"{name}.png"
        _write_mask(path)
        paths[name] = path

    masses = iter(
        [
            (torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0])),
            (torch.tensor([0.0, 2.0]), torch.tensor([1.0, 3.0])),
        ]
    )

    def probe(*_args):
        return next(masses)

    result = accumulate_prior_support(
        [_Camera("a"), _Camera("b")], paths, _Gaussians(), object(), probe_fn=probe
    )
    torch.testing.assert_close(result.positive_mass, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result.total_visibility, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(result.support_score, torch.tensor([1 / 3, 1 / 2]))


def test_accumulate_support_rejects_prior_resolution_mismatch(tmp_path):
    path = tmp_path / "a.png"
    _write_mask(path, shape=(1, 1))
    with pytest.raises(ValueError, match="resolution mismatch"):
        accumulate_prior_support(
            [_Camera("a")], {"a": path}, _Gaussians(), object(), probe_fn=lambda *_: None
        )
