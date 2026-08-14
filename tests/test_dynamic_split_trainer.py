import torch

from dynamic_split.trainer import fingerprint_reconstruction, freeze_reconstruction


class DummyGaussians:
    def __init__(self):
        self._xyz = torch.nn.Parameter(torch.ones(2, 3))
        self._features_dc = torch.nn.Parameter(torch.ones(2, 1, 3))
        self._features_rest = torch.nn.Parameter(torch.ones(2, 2, 3))
        self._scaling = torch.nn.Parameter(torch.ones(2, 3))
        self._rotation = torch.nn.Parameter(torch.ones(2, 4))
        self._opacity = torch.nn.Parameter(torch.ones(2, 1))
        self._deformation = torch.nn.Linear(3, 3)
        self.optimizer = torch.optim.Adam(
            [self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity]
        )


def test_freeze_preserves_bytes_and_disables_gradients():
    gaussians = DummyGaussians()
    before = fingerprint_reconstruction(gaussians)
    freeze_reconstruction(gaussians)
    after = fingerprint_reconstruction(gaussians)
    assert before == after
    assert not gaussians._xyz.requires_grad
    assert all(not parameter.requires_grad for parameter in gaussians._deformation.parameters())
