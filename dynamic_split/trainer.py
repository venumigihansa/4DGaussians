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
from .evaluation import evaluate_prior_agreement
from .export import render_split_artifacts, save_split_arrays, write_gaussian_subset
from .io import camera_frame_name, load_prior_manifest, load_prior_tensor, ordered_unique_cameras
from .renderer import render_dynamic_logits
from .support import accumulate_prior_support, support_supervision_loss


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


def reconstruction_gradient_names(gaussians) -> list[str]:
    names: list[str] = []
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        if getattr(gaussians, name).grad is not None:
            names.append(name)
    for name, parameter in gaussians._deformation.named_parameters():
        if parameter.grad is not None:
            names.append(f"deformation.{name}")
    return names


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
    support = None
    support_statistics = None
    if config.support_weight > 0:
        support = accumulate_prior_support(
            cameras,
            prior_paths,
            gaussians,
            pipeline,
            scene.dataset_type,
        )
        support.save(config.output_dir / "gaussian_prior_support.npz")
        support_statistics = support.statistics()
        unexpected_gradients = reconstruction_gradient_names(gaussians)
        if unexpected_gradients:
            raise RuntimeError(
                "Support precomputation populated frozen reconstruction gradients: "
                f"{unexpected_gradients[:5]}"
            )
    dynamic_logits = torch.nn.Parameter(
        torch.zeros((gaussians.get_xyz.shape[0], 1), dtype=gaussians.get_xyz.dtype, device=device)
    )
    optimizer = torch.optim.Adam([dynamic_logits], lr=config.lr_init, eps=1e-15)
    criterion = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(config.seed)
    order = list(range(len(cameras)))
    rng.shuffle(order)
    history: list[dict[str, float | int | str]] = []
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
        pixel_loss = criterion(rendered_logits, target)
        if support is None:
            support_loss = None
            weighted_support_loss = None
            loss = pixel_loss
        else:
            support_loss = support_supervision_loss(
                dynamic_logits,
                support.support_score,
                support.confidence,
                config.threshold,
                config.support_temperature,
            )
            weighted_support_loss = config.support_weight * support_loss
            loss = pixel_loss + weighted_support_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite dynamic loss at iteration {iteration}")
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        pixel_loss_value = float(pixel_loss.detach())
        support_loss_value = float(support_loss.detach()) if support_loss is not None else 0.0
        weighted_support_loss_value = (
            float(weighted_support_loss.detach()) if weighted_support_loss is not None else 0.0
        )
        history.append(
            {
                "iteration": iteration,
                "total_loss": loss_value,
                "pixel_loss": pixel_loss_value,
                "support_loss": support_loss_value,
                "weighted_support_loss": weighted_support_loss_value,
                "learning_rate": lr,
                "image_name": name,
            }
        )
        if tb_writer is not None:
            tb_writer.add_scalar("dynamic_split/total_loss", loss_value, iteration)
            tb_writer.add_scalar("dynamic_split/pixel_loss", pixel_loss_value, iteration)
            tb_writer.add_scalar("dynamic_split/support_loss", support_loss_value, iteration)
            tb_writer.add_scalar(
                "dynamic_split/weighted_support_loss", weighted_support_loss_value, iteration
            )
            tb_writer.add_scalar("dynamic_split/learning_rate", lr, iteration)
        if iteration % 10 == 0 or iteration == 1:
            progress.set_postfix(loss=f"{loss_value:.6f}")

    after = fingerprint_reconstruction(gaussians)
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise RuntimeError(f"Frozen reconstruction changed during dynamic optimization: {changed[:5]}")
    unexpected_gradients = reconstruction_gradient_names(gaussians)
    if unexpected_gradients:
        raise RuntimeError(
            "Frozen reconstruction received gradients during dynamic optimization: "
            f"{unexpected_gradients[:5]}"
        )

    torch.save(
        {"dynamic_logits": dynamic_logits.detach().cpu(), "threshold": config.threshold},
        config.output_dir / "dynamic_logits.pt",
    )
    with (config.output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
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
    prior_agreement = evaluate_prior_agreement(
        config.output_dir / "renders" / "binary_masks",
        config.prior_dir,
        config.output_dir / "evaluation",
    )
    metadata = {
        **config.to_dict(),
        "prior_manifest": str((config.prior_dir / "manifest.json").resolve()),
        "prior_configuration": prior_manifest.get("config"),
        "camera_count": len(cameras),
        "gaussian_count": int(dynamic.numel()),
        "dynamic_gaussian_count": int(dynamic.sum()),
        "static_gaussian_count": int((~dynamic).sum()),
        "final_loss": history[-1]["total_loss"],
        "final_pixel_loss": history[-1]["pixel_loss"],
        "final_support_loss": history[-1]["support_loss"],
        "final_weighted_support_loss": history[-1]["weighted_support_loss"],
        "support_statistics": support_statistics,
        "prior_agreement": prior_agreement,
        "reconstruction_sha256": after,
    }
    with (config.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return dynamic_logits.detach(), metadata
