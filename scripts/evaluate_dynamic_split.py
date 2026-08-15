#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams, get_combined_args
from dynamic_split.evaluation import evaluate_prior_agreement, evaluate_split_masks
from dynamic_split.export import render_split_artifacts, save_split_arrays, write_gaussian_subset
from dynamic_split.io import ordered_unique_cameras
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and evaluate a learned Gaussian dynamic split.")
    model = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    hidden = ModelHiddenParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Load a saved point-cloud iteration; omit when restoring --fine-checkpoint.",
    )
    parser.add_argument("--dynamic-output-dir", type=Path, required=True)
    parser.add_argument(
        "--export-output-dir",
        type=Path,
        help="Write thresholded exports here instead of overwriting the learned-logit directory.",
    )
    parser.add_argument(
        "--dynamic-threshold",
        type=float,
        help="Override the threshold stored with the learned dynamic logits.",
    )
    parser.add_argument("--fine-checkpoint", type=Path)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hidden.extract(args))
    scene = Scene(dataset, gaussians, load_iteration=getattr(args, "iteration", None), shuffle=False)
    export_output_dir = args.export_output_dir or args.dynamic_output_dir
    export_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.dynamic_output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    fine_checkpoint = getattr(args, "fine_checkpoint", None) or metadata.get("checkpoint_path")
    if fine_checkpoint:
        fine_checkpoint = Path(fine_checkpoint)
        if not fine_checkpoint.is_file():
            raise FileNotFoundError(f"Fine checkpoint not found: {fine_checkpoint}")
        model_parameters, _ = torch.load(fine_checkpoint, map_location="cuda")
        gaussians.restore(model_parameters, optimization.extract(args))
    state = torch.load(args.dynamic_output_dir / "dynamic_logits.pt", map_location="cuda")
    logits = state["dynamic_logits"].to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    if logits.shape[0] != gaussians.get_xyz.shape[0]:
        raise ValueError(
            "Dynamic-logit/Gaussian count mismatch. Evaluate with the exact fine checkpoint "
            "recorded by the dynamic stage or pass --fine-checkpoint explicitly."
        )
    threshold = (
        float(args.dynamic_threshold)
        if args.dynamic_threshold is not None
        else float(state.get("threshold", 7.0))
    )
    torch.save(
        {"dynamic_logits": logits.detach().cpu(), "threshold": threshold},
        export_output_dir / "dynamic_logits.pt",
    )
    dynamic = save_split_arrays(logits, threshold, export_output_dir)
    write_gaussian_subset(gaussians, dynamic, export_output_dir / "dynamic_gaussians.ply")
    write_gaussian_subset(gaussians, ~dynamic, export_output_dir / "static_gaussians.ply")
    render_split_artifacts(
        ordered_unique_cameras(scene),
        gaussians,
        pipeline_params.extract(args),
        logits,
        threshold,
        export_output_dir,
        scene.dataset_type,
        args.opacity_threshold,
    )
    export_metadata = {
        **metadata,
        "threshold": threshold,
        "opacity_threshold": float(args.opacity_threshold),
        "gaussian_count": int(dynamic.numel()),
        "dynamic_gaussian_count": int(dynamic.sum()),
        "static_gaussian_count": int((~dynamic).sum()),
        "source_dynamic_output_dir": str(args.dynamic_output_dir.resolve()),
        "posthoc_reexport": True,
    }
    with (export_output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(export_metadata, handle, indent=2)
    ground_truth_dir = getattr(args, "ground_truth_dir", None)
    prior_dir = getattr(args, "prior_dir", None)
    if prior_dir:
        proxy_report = evaluate_prior_agreement(
            export_output_dir / "renders" / "binary_masks",
            prior_dir,
            export_output_dir / "evaluation",
        )
        print({"prior_agreement": proxy_report})
    if ground_truth_dir:
        report = evaluate_split_masks(
            export_output_dir / "renders" / "binary_masks",
            ground_truth_dir,
            export_output_dir / "evaluation",
            prior_dir,
        )
        print(report)


if __name__ == "__main__":
    main()
