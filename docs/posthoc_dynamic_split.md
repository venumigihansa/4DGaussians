# Post-hoc static/dynamic Gaussian split

This fork optionally adds a third stage after the normal coarse and fine 4DGS
stages. The third stage freezes the completed reconstruction and learns one
global dynamic logit, `d_i`, for each Gaussian.

The renderer computes the original Swift4D logit equation:

```
z_t(x) = sum_i d_i * alpha_i,t(x) * product_{j<i}(1 - alpha_j,t(x))
```

`BCEWithLogitsLoss` compares this rendered logit with a binary 2D prior. The
reconstruction, deformation network, opacity, appearance, scale, and rotation
remain frozen. Manual annotations are never loaded during optimization.

## Generate 2D priors

Install the optional packages from `requirements-dynamic-split.txt` and the
official SAM2 package in a preprocessing environment. Generate priors with:

```
python scripts/generate_dynamic_priors.py \
  --image-dir /external/balloon2/images \
  --forward-flow-dir /external/balloon2/optical_flow_raft \
  --backward-flow-dir /external/balloon2/optical_flow_raft_backward \
  --page4d-predictions /external/page4d/predictions.npz \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /external/weights/sam2.1_hiera_large.pt \
  --output-dir dynamic_split/outputs/balloon2_priors_v2 \
  --mad-multiplier 5.0 \
  --flow-consistency-threshold 1.5 \
  --depth-confidence-quantile 0.10 \
  --irls-iterations 30 \
  --min-component-area 256 \
  --sam-box-padding-ratio 0.05 \
  --sam-min-component-coverage 0.50
```

The command preserves PAGE4D pose-and-depth reprojection as its camera-motion
baseline, then robustly fits only a six-dimensional correction against RAFT.
It filters forward/backward-inconsistent flow and uses the highest-confidence
90% of PAGE4D depth pixels to fit the camera-flow correction. Residual-motion
detection remains enabled on every geometrically valid, flow-consistent pixel,
including low-depth-confidence pixels, so textureless moving objects are not
silently erased. Motion support uses a per-direction median/MAD threshold
instead of a fixed pixel threshold.

For each residual connected component, SAM2 is independently prompted in that
single frame with a padded box and an interior point. SAM2 supplies an object
boundary; it never independently declares an object dynamic. A provisional
SAM2/fallback mask is excluded during a second camera-correction fit, after
which residual components are recomputed and prompted again. There are no
class labels, automatic SAM2 proposals, video memory, or temporal propagation.

The generated layers remain separate:

- `residual_support_initial/`: adaptive flow evidence before provisional exclusion
- `residual_support_refined/`: adaptive flow evidence after the second fit
- `residual_magnitude/`: refined residual-flow visualization
- `sam2_masks/`: accepted final SAM2 boundaries only
- `masks/`: final SAM2 masks plus unmatched residual-component fallbacks
- `overlays/`: fused priors over the RGB frames

`manifest.json` records correction vectors, acceptance decisions, residual
statistics, component decisions, and every generated path. The loader reads
only `image_paths`, `depth`, `depth_conf`, `intrinsic`, and `extrinsic` from
PAGE4D predictions; it never accesses `world_points`.

Manual annotations are not accepted by this command and are never used to
generate or tune an optimization prior. Keep the earlier
`dynamic_split/outputs/balloon2_priors_v1` directory unchanged for visual
comparison.

## Run stage 3 from the 14k checkpoint

All large inputs remain outside this repository:

```
export PYTHON_BIN=/path/to/4dgs/python
export DATASET_DIR=/external/balloon2/colmap_dataset
export FINE_CHECKPOINT=/external/checkpoints/chkpnt_fine_14000.pth
export PRIOR_DIR=/external/balloon2/dynamic_priors
export MODEL_OUTPUT_DIR=/external/experiments/balloon2_posthoc_split
scripts/run_balloon2_dynamic_split.sh
```

The normal `train.py` interface can also enable the stage with
`--train_dynamic_split`. Defaults reproduce Swift4D stage-2 settings: 3,000
iterations, Adam learning rate 0.05 to 0.005, and split threshold `d_i > 7`.

## Evaluate

```
python scripts/evaluate_dynamic_split.py \
  -s /external/balloon2/colmap_dataset \
  -m /external/checkpoints/model_directory \
  --iteration 14000 \
  --fine-checkpoint /external/checkpoints/chkpnt_fine_14000.pth \
  --dynamic-output-dir /external/experiments/balloon2_posthoc_split/dynamic_split \
  --prior-dir /external/balloon2/dynamic_priors \
  --ground-truth-dir /external/balloon2/dynamic_masks_gt
```

Evaluation renders only Gaussians with `d_i > 7`, thresholds their opacity at
0.1, and reports IoU, F1, precision, and recall. Fused-prior and final-split
metrics are stored separately.
