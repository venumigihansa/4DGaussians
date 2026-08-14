#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_split.config import PriorConfig
from dynamic_split.prior_pipeline import run_prior_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate camera-compensated, flow-gated SAM2 priors.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--forward-flow-dir", type=Path, required=True)
    parser.add_argument("--backward-flow-dir", type=Path, required=True)
    parser.add_argument("--page4d-predictions", type=Path, required=True)
    parser.add_argument("--sam2-model-cfg", required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-threshold", type=float, default=1.0)
    parser.add_argument("--flow-consistency-threshold", type=float, default=1.5)
    parser.add_argument("--min-depth-confidence", type=float, default=0.0)
    parser.add_argument("--sam-support-ratio", type=float, default=0.05)
    parser.add_argument("--sam-support-pixels", type=int, default=64)
    parser.add_argument("--min-component-area", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PriorConfig(
        residual_threshold=args.residual_threshold,
        flow_consistency_threshold=args.flow_consistency_threshold,
        min_depth_confidence=args.min_depth_confidence,
        sam_support_ratio=args.sam_support_ratio,
        sam_support_pixels=args.sam_support_pixels,
        min_component_area=args.min_component_area,
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
