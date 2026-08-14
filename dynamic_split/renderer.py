from __future__ import annotations

import torch


def alpha_composite_logits(dynamic_logits: torch.Tensor, alpha_layers: torch.Tensor) -> torch.Tensor:
    """Reference front-to-back implementation used to test Swift4D's equation."""
    logits = dynamic_logits.reshape(-1)
    if alpha_layers.ndim < 2 or alpha_layers.shape[0] != logits.shape[0]:
        raise ValueError("alpha_layers must start with the Gaussian/layer dimension")
    one_minus = 1.0 - alpha_layers
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha_layers[:1]), one_minus[:-1]), dim=0), dim=0
    )
    view_shape = (logits.shape[0],) + (1,) * (alpha_layers.ndim - 1)
    return (logits.view(view_shape) * alpha_layers * transmittance).sum(dim=0)


def render_dynamic_logits(viewpoint, gaussians, pipeline, dynamic_logits, cam_type=None):
    from gaussian_renderer import render

    if dynamic_logits.ndim != 2 or dynamic_logits.shape != (gaussians.get_xyz.shape[0], 1):
        raise ValueError(
            f"Expected dynamic logits [{gaussians.get_xyz.shape[0]},1], got {tuple(dynamic_logits.shape)}"
        )
    override = dynamic_logits.expand(-1, 3)
    background = torch.zeros(3, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    package = render(
        viewpoint,
        gaussians,
        pipeline,
        background,
        override_color=override,
        stage="fine",
        cam_type=cam_type,
    )
    return package["render"][0], package


def render_gaussian_membership(
    viewpoint,
    gaussians,
    pipeline,
    membership: torch.Tensor,
    cam_type=None,
):
    """Render the visible alpha contribution of a selected Gaussian subset."""
    from gaussian_renderer import render

    selected = membership.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    if selected.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError("membership must contain one value per Gaussian")
    if not selected.any():
        return torch.zeros(
            (viewpoint.image_height, viewpoint.image_width),
            dtype=gaussians.get_xyz.dtype,
            device=gaussians.get_xyz.device,
        )
    override = torch.ones((gaussians.get_xyz.shape[0], 3), device=gaussians.get_xyz.device)
    background = torch.zeros(3, dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)
    return render(
        viewpoint,
        gaussians,
        pipeline,
        background,
        override_color=override,
        stage="fine",
        cam_type=cam_type,
        gaussian_mask=selected,
    )["render"][0]
