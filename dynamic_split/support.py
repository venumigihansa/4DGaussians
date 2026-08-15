from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .io import camera_frame_name, load_prior_tensor


@dataclass(frozen=True)
class GaussianPriorSupport:
    positive_mass: torch.Tensor
    negative_mass: torch.Tensor
    total_visibility: torch.Tensor
    support_score: torch.Tensor
    confidence: torch.Tensor
    visibility_reference: float

    def statistics(self, epsilon: float = 1e-8) -> dict[str, object]:
        visible = self.total_visibility > epsilon
        return {
            "visible_gaussian_count": int(visible.sum().item()),
            "zero_visibility_gaussian_count": int((~visible).sum().item()),
            "visibility_reference": self.visibility_reference,
            "positive_mass_quantiles": _quantiles(self.positive_mass[visible]),
            "negative_mass_quantiles": _quantiles(self.negative_mass[visible]),
            "total_visibility_quantiles": _quantiles(self.total_visibility[visible]),
            "support_score_quantiles": _quantiles(self.support_score[visible]),
            "confidence_quantiles": _quantiles(self.confidence[visible]),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            positive_mass=self.positive_mass.detach().cpu().numpy().astype(np.float32),
            negative_mass=self.negative_mass.detach().cpu().numpy().astype(np.float32),
            total_visibility=self.total_visibility.detach().cpu().numpy().astype(np.float32),
            support_score=self.support_score.detach().cpu().numpy().astype(np.float32),
            confidence=self.confidence.detach().cpu().numpy().astype(np.float32),
            visibility_reference=np.float32(self.visibility_reference),
        )


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {}
    probabilities = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        dtype=torch.float32,
        device=values.device,
    )
    result = torch.quantile(values.float(), probabilities).cpu().tolist()
    return {
        name: float(value)
        for name, value in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), result)
    }


def render_probe_masses(viewpoint, gaussians, pipeline, prior: torch.Tensor, cam_type=None):
    """Recover per-Gaussian alpha-transmittance mass with color gradients.

    Channel 0 is integrated only inside the prior, while channel 1 is
    integrated over the complete image. The rasterizer's color gradient is
    exactly the visible alpha contribution of each Gaussian.
    """
    from gaussian_renderer import render

    expected_hw = (int(viewpoint.image_height), int(viewpoint.image_width))
    if tuple(prior.shape) != expected_hw:
        raise ValueError(f"Prior resolution mismatch: {tuple(prior.shape)} != {expected_hw}")
    gaussian_count = int(gaussians.get_xyz.shape[0])
    probes = torch.zeros(
        (gaussian_count, 3),
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
        requires_grad=True,
    )
    background = torch.zeros(3, dtype=probes.dtype, device=probes.device)
    rendered = render(
        viewpoint,
        gaussians,
        pipeline,
        background,
        override_color=probes,
        stage="fine",
        cam_type=cam_type,
    )["render"]
    objective = (rendered[0] * prior).sum() + rendered[1].sum()
    (probe_gradient,) = torch.autograd.grad(objective, probes, create_graph=False)
    positive = probe_gradient[:, 0].detach()
    visibility = probe_gradient[:, 1].detach()
    if not torch.isfinite(positive).all() or not torch.isfinite(visibility).all():
        raise FloatingPointError("Non-finite Gaussian support probe gradient")
    return positive, visibility


def finalize_support(
    positive_mass: torch.Tensor,
    total_visibility: torch.Tensor,
    epsilon: float = 1e-8,
) -> GaussianPriorSupport:
    positive = positive_mass.detach().reshape(-1)
    visibility = total_visibility.detach().reshape(-1)
    if positive.shape != visibility.shape:
        raise ValueError("positive_mass and total_visibility must have identical shapes")
    if positive.numel() == 0:
        raise ValueError("Cannot calculate support for zero Gaussians")
    if not torch.isfinite(positive).all() or not torch.isfinite(visibility).all():
        raise ValueError("Support masses must contain only finite values")
    if (positive < -1e-6).any() or (visibility < -1e-6).any():
        raise ValueError("Support masses cannot be negative")

    visibility = visibility.clamp_min(0)
    positive = torch.minimum(positive.clamp_min(0), visibility)
    visible = visibility > epsilon
    if not visible.any():
        raise ValueError("All Gaussians have zero visibility in the prior cameras")

    negative = (visibility - positive).clamp_min(0)
    support = torch.zeros_like(visibility)
    support[visible] = positive[visible] / (visibility[visible] + epsilon)
    support.clamp_(0, 1)
    visibility_reference_tensor = torch.median(visibility[visible])
    if not torch.isfinite(visibility_reference_tensor) or visibility_reference_tensor <= 0:
        raise ValueError("Could not derive a positive visibility reference")
    confidence = torch.zeros_like(visibility)
    confidence[visible] = visibility[visible] / (
        visibility[visible] + visibility_reference_tensor
    )
    return GaussianPriorSupport(
        positive_mass=positive,
        negative_mass=negative,
        total_visibility=visibility,
        support_score=support,
        confidence=confidence,
        visibility_reference=float(visibility_reference_tensor.item()),
    )


def accumulate_prior_support(
    cameras: list,
    prior_paths: dict[str, Path],
    gaussians,
    pipeline,
    cam_type=None,
    probe_fn: Callable = render_probe_masses,
    epsilon: float = 1e-8,
) -> GaussianPriorSupport:
    device = gaussians.get_xyz.device
    gaussian_count = int(gaussians.get_xyz.shape[0])
    positive = torch.zeros(gaussian_count, dtype=gaussians.get_xyz.dtype, device=device)
    visibility = torch.zeros_like(positive)
    for camera in cameras:
        name = camera_frame_name(camera)
        if name not in prior_paths:
            raise KeyError(f"No dynamic prior path for camera {name}")
        prior = load_prior_tensor(
            prior_paths[name], (camera.image_height, camera.image_width), device
        )
        frame_positive, frame_visibility = probe_fn(
            camera, gaussians, pipeline, prior, cam_type
        )
        if frame_positive.numel() != gaussian_count or frame_visibility.numel() != gaussian_count:
            raise ValueError("Support probe returned the wrong number of Gaussian values")
        positive.add_(frame_positive.reshape(-1))
        visibility.add_(frame_visibility.reshape(-1))
    return finalize_support(positive, visibility, epsilon)


def support_classification_logits(
    dynamic_logits: torch.Tensor,
    threshold: float,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Support temperature must be positive")
    return (dynamic_logits.reshape(-1) - threshold) / temperature


def support_supervision_loss(
    dynamic_logits: torch.Tensor,
    support_score: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float,
    temperature: float,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    logits = support_classification_logits(dynamic_logits, threshold, temperature)
    target = support_score.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    weights = confidence.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if logits.shape != target.shape or logits.shape != weights.shape:
        raise ValueError("Logits, support scores and confidence must have one value per Gaussian")
    if not torch.isfinite(target).all() or not torch.isfinite(weights).all():
        raise ValueError("Support scores and confidence must be finite")
    if (target < 0).any() or (target > 1).any() or (weights < 0).any():
        raise ValueError("Support targets must be in [0,1] and confidence must be non-negative")
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (weights * losses).sum() / (weights.sum() + epsilon)
