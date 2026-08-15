from pathlib import Path

import numpy as np
import torch

from dynamic_split.raft import ensure_bidirectional_raft_flows


class FakeRaft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, source, target):
        self.calls.append((tuple(source.shape), float(source.mean()), float(target.mean())))
        difference = target.mean(dim=(1, 2, 3)) - source.mean(dim=(1, 2, 3))
        output = torch.zeros(
            (source.shape[0], 2, source.shape[2], source.shape[3]),
            dtype=torch.float32,
            device=source.device,
        )
        output[:, 0] = difference[:, None, None]
        output[:, 1] = -difference[:, None, None]
        return [output]


def preprocess(source, target):
    return source.float(), target.float()


def test_raft_bidirectional_names_padding_crop_and_dtype(tmp_path: Path):
    images = [np.full((5, 7, 3), value, dtype=np.uint8) for value in (0, 10, 30)]
    forward = tmp_path / "forward"
    backward = tmp_path / "backward"
    model = FakeRaft()
    metadata = ensure_bidirectional_raft_flows(
        images,
        forward,
        backward,
        device="cpu",
        model=model,
        preprocess=preprocess,
        weights_identifier="fake-raft",
    )
    assert sorted(path.name for path in forward.iterdir()) == [
        "flow_000000_to_000001.npy", "flow_000001_to_000002.npy"
    ]
    assert sorted(path.name for path in backward.iterdir()) == [
        "flow_000001_to_000000.npy", "flow_000002_to_000001.npy"
    ]
    assert all(shape == (1, 3, 8, 8) for shape, _, _ in model.calls)
    flow = np.load(forward / "flow_000000_to_000001.npy")
    assert flow.shape == (5, 7, 2)
    assert flow.dtype == np.float32
    assert np.allclose(flow[..., 0], 10.0)
    backward_flow = np.load(backward / "flow_000001_to_000000.npy")
    assert np.allclose(backward_flow[..., 0], -10.0)
    assert metadata["generated_flow_count"] == 4
    assert metadata["model_weights"] == "fake-raft"


def test_valid_caches_skip_and_invalid_caches_regenerate(tmp_path: Path):
    images = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8)]
    forward = tmp_path / "forward"
    backward = tmp_path / "backward"
    first_model = FakeRaft()
    ensure_bidirectional_raft_flows(
        images, forward, backward, device="cpu", model=first_model, preprocess=preprocess
    )
    metadata = ensure_bidirectional_raft_flows(images, forward, backward, device="cpu")
    assert metadata["generated_flow_count"] == 0
    assert metadata["reused_flow_count"] == 2

    np.save(forward / "flow_000000_to_000001.npy", np.zeros((2, 2), dtype=np.float32))
    repair_model = FakeRaft()
    metadata = ensure_bidirectional_raft_flows(
        images, forward, backward, device="cpu", model=repair_model, preprocess=preprocess
    )
    assert metadata["generated_flow_count"] == 1
    assert len(repair_model.calls) == 1
    assert np.load(forward / "flow_000000_to_000001.npy").shape == (4, 6, 2)
