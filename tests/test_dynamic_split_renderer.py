import torch

from dynamic_split.renderer import alpha_composite_logits


def test_alpha_composite_matches_swift4d_equation():
    logits = torch.tensor([[2.0], [-1.0], [4.0]], requires_grad=True)
    alpha = torch.tensor([[0.5], [0.25], [0.1]])
    rendered = alpha_composite_logits(logits, alpha)
    expected = 2.0 * 0.5 + (-1.0) * 0.25 * 0.5 + 4.0 * 0.1 * 0.5 * 0.75
    torch.testing.assert_close(rendered.squeeze(), torch.tensor(expected))
    rendered.sum().backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 3


def test_opaque_front_layer_occludes_later_logits():
    logits = torch.tensor([[3.0], [100.0]])
    alpha = torch.tensor([[1.0], [1.0]])
    rendered = alpha_composite_logits(logits, alpha)
    torch.testing.assert_close(rendered.squeeze(), torch.tensor(3.0))
