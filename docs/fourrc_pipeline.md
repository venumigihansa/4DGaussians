# Native 4RC pipeline

4DGaussians can use a 4RC prediction archive directly as its scene source. No
COLMAP conversion, image export, sparse point cloud, or generated PLY is
needed.

## Archive contract and frame semantics

The archive must contain a scalar `n_frames` and, for every frame index `i`:

- `view_i_img`: `[1,3,H,W]` RGB in `[-1,1]`
- `pred_i_pts`: `[1,H,W,3]` world-space dense point map
- `pred_i_conf`: `[1,H,W]` point/depth confidence
- `pred_i_intrinsic`: `[3,3]` pinhole intrinsic matrix
- `pred_i_extrinsic`: `[4,4]` camera-to-world rigid transform

All frames must have the same resolution. Focal lengths must be positive, and
the principal point must be within 0.5 pixels of `(W/2,H/2)`, because the
current Gaussian rasterizer does not represent an arbitrary principal point.

Frame `i` is named `frame_XXXXXX`, and its training time is `i/n_frames`.
Some 4RC archives also contain trajectory arrays whose names describe a query
frame. A query frame is the frame at which those trajectory points were
selected or anchored; it is not a special RGB frame for ordinary scene
loading. Trajectory arrays are deliberately neither loaded nor used by this
implementation. Stage-3 trajectory supervision is deferred.

## Dense canonical initialization

The canonical Gaussian positions come directly from one `pred_i_pts` map,
with colors from the corresponding embedded image. By default frame 0 is used
and every finite point with positive camera-space Z is retained, with no
downsampling:

```bash
python train.py \
  -s /path/to/bear.npz \
  -m /path/to/bear_4dgs \
  --fourrc_init_frame 0
```

`--fourrc_confidence_quantile Q` optionally retains points at or above the
`Q` confidence quantile after geometric validity filtering. Its default is
`0.0`, which disables confidence filtering. `--fourrc_max_init_points N`
then deterministically samples at most `N` valid points across the full image,
preserving spatial coverage; `0` keeps all valid points. All frames train by
default.
`--fourrc_holdout_stride N` reserves frames `0,N,2N,...` for testing while
requiring at least one frame to remain for training.

Training densification defaults to the historical 360,000-Gaussian ceiling.
Use `--max_gaussians N` to set a different hard ceiling; individual clone and
split operations are budgeted so the trained model does not grow past it. For
example, initialize from 40,000 points and cap training at 100,000 Gaussians:

```bash
python train.py \
  -s /path/to/car_turn.npz \
  -m /path/to/car_turn_4dgs \
  --fourrc_max_init_points 40000 \
  --max_gaussians 100000
```

The loader inverts every camera-to-world pose for rendering. It derives the
scene radius from both camera centers and the 90th-percentile distance from
the point-cloud median, so a fixed-camera sequence still has a useful spatial
learning-rate and densification scale.

## RAFT and SAM2 prior generation

For 4RC input, RGB, camera poses, intrinsics, confidence, and depth all come
from the archive. Depth is recovered by transforming each world-space point
map into its camera and taking positive camera-space Z. Missing or malformed
consecutive flows are generated in both directions with torchvision RAFT
Large and cached under `<output-dir>/raft/{forward,backward}`. Valid float32
`[H,W,2]` caches are reused. Use `--overwrite-raft` to recompute them, or
`--raft-cache-dir` to select another cache root.

```bash
python scripts/generate_dynamic_priors.py \
  --fourrc-predictions /path/to/bear.npz \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --output-dir /path/to/bear_priors \
  --device cuda
```

RAFT and SAM2 use the same `--device`. Images are passed to RAFT in memory;
they are not exported. RAFT pads right/bottom to multiples of eight and crops
flow back to the exact archive resolution. The existing forward/backward
consistency, pose-flow reprojection, confidence masking, two-pass Cauchy IRLS,
residual detection, and per-frame SAM2 prompting are unchanged.

The final priors are written as `masks/frame_XXXXXX.png`, matching the native
scene camera names. Manifest schema 3 records the geometry source, prediction
archive, RAFT model/cache information, frame count, and resolution.

## End-to-end training

```bash
python train.py \
  -s /path/to/bear.npz \
  -m /path/to/bear_4dgs \
  --train_dynamic_split \
  --dynamic_prior_dir /path/to/bear_priors
```

This runs the existing coarse, fine, and `d_i` split stages. The third-stage
loss and Gaussian/prior association are unchanged; no 4RC trajectories are
used.

## Page4D compatibility

Page4D geometry remains supported. It still requires `--image-dir`, while
forward/backward flow directories are now optional because missing flows can
be generated automatically:

```bash
python scripts/generate_dynamic_priors.py \
  --image-dir /path/to/images \
  --page4d-predictions /path/to/page4d_predictions.npz \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt \
  --output-dir /path/to/page4d_priors \
  --device cuda
```

Existing explicit `--forward-flow-dir` and `--backward-flow-dir` arguments are
also accepted when supplied together.
