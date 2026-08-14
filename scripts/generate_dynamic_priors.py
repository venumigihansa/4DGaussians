#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_split.config import PriorConfig
from dynamic_split.prior_pipeline import run_prior_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate robust camera-corrected residual and per-frame SAM2 priors."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--forward-flow-dir", type=Path, required=True)
    parser.add_argument("--backward-flow-dir", type=Path, required=True)
    parser.add_argument("--page4d-predictions", type=Path, required=True)
    parser.add_argument("--sam2-model-cfg", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mad-multiplier", type=float, default=5.0)
    parser.add_argument("--flow-consistency-threshold", type=float, default=1.5)
    parser.add_argument("--depth-confidence-quantile", type=float, default=0.10)
    parser.add_argument("--irls-iterations", type=int, default=30)
    parser.add_argument("--min-component-area", type=int, default=256)
    parser.add_argument("--sam-box-padding-ratio", type=float, default=0.05)
    parser.add_argument("--sam-min-component-coverage", type=float, default=0.50)
    parser.add_argument(
        "--disable-pose-flow-refinement",
        action="store_true",
        help="Use exact PAGE4D pose reprojection without fitting the six-DoF correction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PriorConfig(
        mad_multiplier=args.mad_multiplier,
        flow_consistency_threshold=args.flow_consistency_threshold,
        depth_confidence_quantile=args.depth_confidence_quantile,
        irls_iterations=args.irls_iterations,
        min_component_area=args.min_component_area,
        sam_box_padding_ratio=args.sam_box_padding_ratio,
        sam_min_component_coverage=args.sam_min_component_coverage,
        refine_pose_flow=not args.disable_pose_flow_refinement,
    )
    manifest = run_prior_pipeline(
        args.image_dir,
        args.forward_flow_dir,
        args.backward_flow_dir,
        args.page4d_predictions,
        args.output_dir,
        args.sam2_model_cfg,
        args.sam2_checkpoint,
        config,
        args.device,
    )
    print(f"Wrote {len(manifest['frames'])} dynamic priors to {args.output_dir}")


if __name__ == "__main__":
    main()
