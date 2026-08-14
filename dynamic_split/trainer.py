from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import DynamicSplitConfig
from .export import render_split_artifacts, save_split_arrays, write_gaussian_subset
from .io import camera_frame_name, load_prior_manifest, load_prior_tensor, ordered_unique_cameras
from .renderer import render_dynamic_logits


def reconstruction_tensors(gaussians):
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        yield name, getattr(gaussians, name)
    for name, value in gaussians._deformation.state_dict().items():
        yield f"deformation.{name}", value


def fingerprint_reconstruction(gaussians) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, tensor in reconstruction_tensors(gaussians):
        data = tensor.detach().contiguous().cpu().numpy().tobytes()
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def freeze_reconstruction(gaussians) -> None:
    for _, tensor in reconstruction_tensors(gaussians):
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            tensor.requires_grad_(False)
            tensor.grad = None
    for parameter in gaussians._deformation.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if gaussians.optimizer is not None:
        gaussians.optimizer.zero_grad(set_to_none=True)


def _learning_rate(config: DynamicSplitConfig, iteration: int) -> float:
    if config.iterations == 1:
        return config.lr_final
    progress = (iteration - 1) / (config.iterations - 1)
    return config.lr_init * math.exp(math.log(config.lr_final / config.lr_init) * progress)


def run_dynamic_stage(scene, gaussians, pipeline, config: DynamicSplitConfig, tb_writer=None):
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prior_manifest, prior_paths = load_prior_manifest(config.prior_dir)
    all_cameras = ordered_unique_cameras(scene)
    cameras = [camera for camera in all_cameras if camera_frame_name(camera) in prior_paths]
    missing = sorted(set(prior_paths) - {camera_frame_name(camera) for camera in cameras})
    if missing:
        raise ValueError(f"Priors have no matching scene cameras: {', '.join(missing[:10])}")
    if not cameras:
        raise ValueError("No scene cameras matched the dynamic priors")

    before = fingerprint_reconstruction(gaussians)
    freeze_reconstruction(gaussians)
    device = gaussians.get_xyz.device
    dynamic_logits = torch.nn.Parameter(
        torch.zeros((gaussians.get_xyz.shape[0], 1), dtype=gaussians.get_xyz.dtype, device=device)
    )
    optimizer = torch.optim.Adam([dynamic_logits], lr=config.lr_init, eps=1e-15)
    criterion = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(config.seed)
    order = list(range(len(cameras)))
    rng.shuffle(order)
    history: list[tuple[int, float, float, str]] = []
    progress = tqdm(range(1, config.iterations + 1), desc="Dynamic split stage")
    for iteration in progress:
        if (iteration - 1) % len(order) == 0 and iteration > 1:
            rng.shuffle(order)
        camera = cameras[order[(iteration - 1) % len(order)]]
        name = camera_frame_name(camera)
        target = load_prior_tensor(
            prior_paths[name], (camera.image_height, camera.image_width), device
        )
        lr = _learning_rate(config, iteration)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        rendered_logits, _ = render_dynamic_logits(
            camera, gaussians, pipeline, dynamic_logits, scene.dataset_type
        )
        loss = criterion(rendered_logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite dynamic loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        history.append((iteration, loss_value, lr, name))
        if tb_writer is not None:
            tb_writer.add_scalar("dynamic_split/loss", loss_value, iteration)
            tb_writer.add_scalar("dynamic_split/learning_rate", lr, iteration)
        if iteration % 10 == 0 or iteration == 1:
            progress.set_postfix(loss=f"{loss_value:.6f}")

    after = fingerprint_reconstruction(gaussians)
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise RuntimeError(f"Frozen reconstruction changed during dynamic optimization: {changed[:5]}")

    torch.save(
        {"dynamic_logits": dynamic_logits.detach().cpu(), "threshold": config.threshold},
        config.output_dir / "dynamic_logits.pt",
    )
    with (config.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("iteration", "loss", "learning_rate", "image_name"))
        writer.writerows(history)
    dynamic = save_split_arrays(dynamic_logits, config.threshold, config.output_dir)
    write_gaussian_subset(gaussians, dynamic, config.output_dir / "dynamic_gaussians.ply")
    write_gaussian_subset(gaussians, ~dynamic, config.output_dir / "static_gaussians.ply")
    render_split_artifacts(
        cameras,
        gaussians,
        pipeline,
        dynamic_logits,
        config.threshold,
        config.output_dir,
        scene.dataset_type,
    )
    metadata = {
        **config.to_dict(),
        "prior_manifest": str((config.prior_dir / "manifest.json").resolve()),
        "prior_configuration": prior_manifest.get("config"),
        "camera_count": len(cameras),
        "gaussian_count": int(dynamic.numel()),
        "dynamic_gaussian_count": int(dynamic.sum()),
        "static_gaussian_count": int((~dynamic).sum()),
        "final_loss": history[-1][1],
        "reconstruction_sha256": after,
    }
    with (config.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return dynamic_logits.detach(), metadata
