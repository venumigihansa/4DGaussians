#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${PYTHON_BIN:?Set PYTHON_BIN to the 4DGaussians Python executable}"
: "${DATASET_DIR:?Set DATASET_DIR to the external Balloon2 COLMAP dataset}"
: "${FINE_CHECKPOINT:?Set FINE_CHECKPOINT to chkpnt_fine_14000.pth}"
: "${PRIOR_DIR:?Set PRIOR_DIR to the generated dynamic-prior directory}"
: "${MODEL_OUTPUT_DIR:?Set MODEL_OUTPUT_DIR to a new output directory}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
for required in "$FINE_CHECKPOINT" "$PRIOR_DIR/manifest.json"; do
    if [[ ! -e "$required" ]]; then
        echo "Required input not found: $required" >&2
        exit 1
    fi
done

cd "$repo_dir"
"$PYTHON_BIN" train.py \
    -s "$DATASET_DIR" \
    -m "$MODEL_OUTPUT_DIR" \
    --iterations 14000 \
    --coarse_iterations 3000 \
    --start_checkpoint "$FINE_CHECKPOINT" \
    --test_iterations 999999 \
    --save_iterations 14000 \
    --train_dynamic_split \
    --dynamic_prior_dir "$PRIOR_DIR" \
    --dynamic_output_dir "$MODEL_OUTPUT_DIR/dynamic_split" \
    --dynamic_iterations "${DYNAMIC_ITERATIONS:-3000}" \
    --dynamic_lr_init "${DYNAMIC_LR_INIT:-0.05}" \
    --dynamic_lr_final "${DYNAMIC_LR_FINAL:-0.005}" \
    --dynamic_threshold "${DYNAMIC_THRESHOLD:-7.0}" \
    --dynamic_support_weight "${DYNAMIC_SUPPORT_WEIGHT:-0.1}" \
    --dynamic_support_temperature "${DYNAMIC_SUPPORT_TEMPERATURE:-1.0}"
