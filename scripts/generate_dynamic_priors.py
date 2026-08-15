#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_split.config import PriorConfig
from dynamic_split.prior_pipeline import run_fourrc_prior_pipeline, run_prior_pipeline
from dynamic_split.raft import ensure_bidirectional_raft_flows
from utils.fourrc_utils import load_fourrc_prior_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate robust camera-corrected residual and per-frame SAM2 priors."
    )
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--forward-flow-dir", type=Path)
    parser.add_argument("--backward-flow-dir", type=Path)
    prediction_group = parser.add_mutually_exclusive_group(required=True)
    prediction_group.add_argument("--page4d-predictions", type=Path)
    prediction_group.add_argument("--fourrc-predictions", type=Path)
    parser.add_argument("--sam2-model-cfg", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--raft-cache-dir",
        type=Path,
        help="Cache root used when explicit flow directories are omitted.",
    )
    parser.add_argument("--overwrite-raft", action="store_true")
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
        help="Use pose reprojection without fitting the six-DoF camera-flow correction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.page4d_predictions is not None and args.image_dir is None:
        raise SystemExit("--image-dir is required with --page4d-predictions")
    if (args.forward_flow_dir is None) != (args.backward_flow_dir is None):
        raise SystemExit(
            "--forward-flow-dir and --backward-flow-dir must be supplied together"
        )
    if args.forward_flow_dir is None:
        raft_root = args.raft_cache_dir or (args.output_dir / "raft")
        forward_flow_dir = raft_root / "forward"
        backward_flow_dir = raft_root / "backward"
    else:
        forward_flow_dir = args.forward_flow_dir
        backward_flow_dir = args.backward_flow_dir

    if args.fourrc_predictions is not None:
        fourrc_data = load_fourrc_prior_data(args.fourrc_predictions)
        rgb_frames = list(fourrc_data.images_uint8)
    else:
        image_paths = sorted(args.image_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No PNG images found in {args.image_dir}")
        rgb_frames = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]

    raft_metadata = ensure_bidirectional_raft_flows(
        rgb_frames,
        forward_flow_dir,
        backward_flow_dir,
        device=args.device,
        overwrite=args.overwrite_raft,
    )
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
    if args.fourrc_predictions is not None:
        manifest = run_fourrc_prior_pipeline(
            args.fourrc_predictions,
            forward_flow_dir,
            backward_flow_dir,
            args.output_dir,
            args.sam2_model_cfg,
            args.sam2_checkpoint,
            config,
            args.device,
            raft_metadata=raft_metadata,
        )
    else:
        manifest = run_prior_pipeline(
            args.image_dir,
            forward_flow_dir,
            backward_flow_dir,
            args.page4d_predictions,
            args.output_dir,
            args.sam2_model_cfg,
            args.sam2_checkpoint,
            config,
            args.device,
            raft_metadata=raft_metadata,
        )
    print(f"Wrote {len(manifest['frames'])} dynamic priors to {args.output_dir}")


if __name__ == "__main__":
    main()
