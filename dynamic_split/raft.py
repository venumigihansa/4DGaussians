from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .flow import flow_path


RAFT_WEIGHTS_IDENTIFIER = "torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT"


def _valid_cached_flow(path: Path, height: int, width: int) -> bool:
    if not path.is_file():
        return False
    try:
        flow = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        flow.shape == (height, width, 2)
        and flow.dtype == np.float32
        and np.isfinite(flow).all()
    )


def _validate_images(images: Sequence[np.ndarray]) -> tuple[int, int]:
    if len(images) < 1:
        raise ValueError("RAFT requires at least one RGB frame")
    height = width = None
    for index, image in enumerate(images):
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB frame {index} [H,W,3], got {rgb.shape}")
        if not np.isfinite(rgb).all():
            raise ValueError(f"RGB frame {index} contains non-finite values")
        if height is None:
            height, width = rgb.shape[:2]
        elif rgb.shape[:2] != (height, width):
            raise ValueError(
                f"RGB resolution mismatch at frame {index}: {rgb.shape[:2]} != {(height, width)}"
            )
    return int(height), int(width)


def _load_default_raft(device: str):
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).eval().to(torch.device(device))
    return model, weights.transforms(), RAFT_WEIGHTS_IDENTIFIER


def _predict_flow(
    source: np.ndarray,
    target: np.ndarray,
    model,
    preprocess: Callable,
    device: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as torch_functional

    height, width = source.shape[:2]
    source_tensor = torch.from_numpy(np.ascontiguousarray(source)).permute(2, 0, 1).unsqueeze(0)
    target_tensor = torch.from_numpy(np.ascontiguousarray(target)).permute(2, 0, 1).unsqueeze(0)
    pad_height = (-height) % 8
    pad_width = (-width) % 8
    if pad_height or pad_width:
        padding = (0, pad_width, 0, pad_height)
        source_tensor = torch_functional.pad(source_tensor, padding, mode="replicate")
        target_tensor = torch_functional.pad(target_tensor, padding, mode="replicate")
    source_tensor, target_tensor = preprocess(source_tensor, target_tensor)
    source_tensor = source_tensor.to(device)
    target_tensor = target_tensor.to(device)
    with torch.inference_mode():
        predictions = model(source_tensor, target_tensor)
    prediction = predictions[-1] if isinstance(predictions, (tuple, list)) else predictions
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != 2:
        raise ValueError(f"RAFT model returned an invalid flow tensor: {tuple(prediction.shape)}")
    flow = prediction[0, :, :height, :width].permute(1, 2, 0).detach().cpu().numpy()
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
        raise ValueError(f"RAFT produced an invalid flow array: {flow.shape}")
    return flow


def ensure_bidirectional_raft_flows(
    images: Sequence[np.ndarray],
    forward_flow_dir: Path,
    backward_flow_dir: Path,
    *,
    device: str = "cuda",
    overwrite: bool = False,
    model=None,
    preprocess: Callable | None = None,
    weights_identifier: str | None = None,
) -> dict:
    """Generate missing consecutive forward/backward RAFT flows and reuse valid caches."""
    height, width = _validate_images(images)
    forward_flow_dir = Path(forward_flow_dir)
    backward_flow_dir = Path(backward_flow_dir)
    forward_flow_dir.mkdir(parents=True, exist_ok=True)
    backward_flow_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for index in range(len(images) - 1):
        jobs.extend(
            (
                (index, index + 1, forward_flow_dir),
                (index + 1, index, backward_flow_dir),
            )
        )
    missing = [
        job for job in jobs
        if overwrite or not _valid_cached_flow(flow_path(job[2], job[0], job[1]), height, width)
    ]
    identifier = weights_identifier or RAFT_WEIGHTS_IDENTIFIER
    if missing:
        if model is None:
            model, preprocess, identifier = _load_default_raft(device)
        elif preprocess is None:
            raise ValueError("preprocess must be supplied with an injected RAFT model")
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "to"):
            model.to(device)
        for source_index, target_index, directory in missing:
            output_path = flow_path(directory, source_index, target_index)
            flow = _predict_flow(
                np.asarray(images[source_index]),
                np.asarray(images[target_index]),
                model,
                preprocess,
                device,
            )
            np.save(output_path, flow)

    return {
        "model_weights": identifier,
        "forward_cache_dir": str(forward_flow_dir.resolve()),
        "backward_cache_dir": str(backward_flow_dir.resolve()),
        "generated_flow_count": len(missing),
        "reused_flow_count": len(jobs) - len(missing),
    }
