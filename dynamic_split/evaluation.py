from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .io import load_prior_manifest


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} != {gt.shape}")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def evaluate_split_masks(
    rendered_mask_dir: Path,
    ground_truth_dir: Path,
    output_dir: Path,
    prior_dir: Path | None = None,
) -> dict:
    ground_truth = {path.stem: path for path in sorted(ground_truth_dir.glob("*.png"))}
    if not ground_truth:
        raise FileNotFoundError(f"No PNG ground-truth masks found in {ground_truth_dir}")
    prior_mapping: dict[str, Path] = {}
    if prior_dir is not None:
        _, prior_mapping = load_prior_manifest(prior_dir)
    rows: list[dict] = []
    final_total = np.zeros(4, dtype=np.int64)
    prior_total = np.zeros(4, dtype=np.int64)
    for name, gt_path in ground_truth.items():
        predicted_path = rendered_mask_dir / f"{name}.png"
        if not predicted_path.is_file():
            raise FileNotFoundError(f"Rendered split mask missing for annotated frame: {predicted_path}")
        target = load_binary(gt_path)
        final_counts = confusion_counts(load_binary(predicted_path), target)
        final_total += final_counts
        row = {"image_name": name, **{f"final_{k}": v for k, v in metrics_from_counts(*final_counts).items()}}
        if name in prior_mapping:
            prior_counts = confusion_counts(load_binary(prior_mapping[name]), target)
            prior_total += prior_counts
            row.update({f"prior_{k}": v for k, v in metrics_from_counts(*prior_counts).items()})
        rows.append(row)
    report = {
        "annotated_frames": len(rows),
        "final_split": metrics_from_counts(*map(int, final_total)),
    }
    if prior_mapping:
        report["fused_prior"] = metrics_from_counts(*map(int, prior_total))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report
