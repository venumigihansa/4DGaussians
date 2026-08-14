#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams, get_combined_args
from dynamic_split.evaluation import evaluate_split_masks
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
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--dynamic-output-dir", type=Path, required=True)
    parser.add_argument("--fine-checkpoint", type=Path)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hidden.extract(args))
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
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
    threshold = float(state.get("threshold", 7.0))
    dynamic = save_split_arrays(logits, threshold, args.dynamic_output_dir)
    write_gaussian_subset(gaussians, dynamic, args.dynamic_output_dir / "dynamic_gaussians.ply")
    write_gaussian_subset(gaussians, ~dynamic, args.dynamic_output_dir / "static_gaussians.ply")
    render_split_artifacts(
        ordered_unique_cameras(scene),
        gaussians,
        pipeline_params.extract(args),
        logits,
        threshold,
        args.dynamic_output_dir,
        scene.dataset_type,
        args.opacity_threshold,
    )
    if args.ground_truth_dir:
        report = evaluate_split_masks(
            args.dynamic_output_dir / "renders" / "binary_masks",
            args.ground_truth_dir,
            args.dynamic_output_dir / "evaluation",
            args.prior_dir,
        )
        print(report)


if __name__ == "__main__":
    main()
