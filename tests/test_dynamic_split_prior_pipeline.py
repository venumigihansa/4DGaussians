from pathlib import Path

import numpy as np
from PIL import Image

from dynamic_split.config import PriorConfig
from dynamic_split.io import load_prior_manifest
from dynamic_split.prior_pipeline import (
    _load_observations,
    frame_flow_specs,
    load_fourrc_geometry,
    load_page4d_geometry,
    run_fourrc_prior_pipeline,
    run_prior_pipeline,
)


class BoxSegmenter:
    def __init__(self):
        self.predictor = self
        self.images = 0
        self.shape = None

    def set_image(self, image):
        self.images += 1
        self.shape = image.shape[:2]

    def predict(self, **kwargs):
        mask = np.zeros(self.shape, dtype=bool)
        x0, y0, x1, y1 = np.rint(kwargs["box"]).astype(int)
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        return mask[None], np.array([0.9], dtype=np.float32), np.zeros((1, 1, 1))


def test_first_interior_and_last_frames_use_available_directions():
    assert frame_flow_specs(0, 4) == [("forward", 1)]
    assert frame_flow_specs(2, 4) == [("forward", 3), ("backward", 1)]
    assert frame_flow_specs(3, 4) == [("backward", 2)]


def _write_two_frame_inputs(root: Path):
    image_dir = root / "images"
    forward_dir = root / "forward"
    backward_dir = root / "backward"
    image_dir.mkdir()
    forward_dir.mkdir()
    backward_dir.mkdir()
    height = width = 40
    for index in range(2):
        Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(
            image_dir / f"{index:06d}.png"
        )
    zero_flow = np.zeros((height, width, 2), dtype=np.float32)
    np.save(forward_dir / "flow_000000_to_000001.npy", zero_flow)
    np.save(backward_dir / "flow_000001_to_000000.npy", zero_flow)
    predictions = root / "predictions.npz"
    np.savez(
        predictions,
        image_paths=np.array(["0.png", "1.png"]),
        extrinsic=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        intrinsic=np.repeat(
            np.array([[100.0, 0, 20.0], [0, 100.0, 20.0], [0, 0, 1.0]], dtype=np.float32)[None],
            2,
            axis=0,
        ),
        depth=np.ones((2, height, width, 1), dtype=np.float32),
        depth_conf=np.ones((2, height, width, 1), dtype=np.float32),
        world_points=np.array(["this entry must never be read"], dtype=object),
    )
    return image_dir, forward_dir, backward_dir, predictions


def test_pipeline_manifest_keeps_residual_sam_and_fused_outputs_distinct(tmp_path):
    image_dir, forward_dir, backward_dir, predictions = _write_two_frame_inputs(tmp_path)
    output = tmp_path / "output"
    segmenter = BoxSegmenter()
    manifest = run_prior_pipeline(
        image_dir,
        forward_dir,
        backward_dir,
        predictions,
        output,
        "unused.yaml",
        tmp_path / "unused.pt",
        PriorConfig(min_component_area=1),
        device="cpu",
        segmenter=segmenter,
    )
    assert manifest["version"] == 3
    assert manifest["geometry_source"] == "page4d"
    assert manifest["frame_count"] == 2
    assert manifest["resolution"] == {"height": 40, "width": 40}
    assert manifest["ground_truth_used"] is False
    assert segmenter.images == 2
    expected_directories = {
        "residual_support_initial",
        "residual_support_refined",
        "residual_magnitude",
        "sam2_masks",
        "masks",
        "overlays",
    }
    assert expected_directories <= {path.name for path in output.iterdir() if path.is_dir()}
    paths = manifest["frames"][0]["paths"]
    assert paths["residual_support_refined"] != paths["sam2_mask"] != paths["prior"]
    _, prior_mapping = load_prior_manifest(output)
    assert sorted(prior_mapping) == ["000000", "000001"]
    assert len(manifest["frames"][0]["initial_motion"]["directions"]) == 1
    assert len(manifest["frames"][1]["initial_motion"]["directions"]) == 1


def test_low_depth_confidence_is_excluded_from_fit_but_retained_for_motion(tmp_path):
    image_dir, forward_dir, backward_dir, predictions = _write_two_frame_inputs(tmp_path)
    geometry = load_page4d_geometry(predictions)
    geometry["depth_conf"][0, 0, 0] = 0.0
    observations = _load_observations(
        0, geometry, forward_dir, backward_dir, PriorConfig(depth_confidence_quantile=0.10)
    )
    observation = observations[0]
    assert not observation.camera_fit_valid[0, 0]
    assert observation.motion_detection_valid[0, 0]
    assert observation.camera_fit_valid.sum() < observation.motion_detection_valid.sum()


def test_missing_flow_pair_fails_clearly(tmp_path):
    image_dir, forward_dir, backward_dir, predictions = _write_two_frame_inputs(tmp_path)
    (backward_dir / "flow_000001_to_000000.npy").unlink()
    try:
        run_prior_pipeline(
            image_dir,
            forward_dir,
            backward_dir,
            predictions,
            tmp_path / "output",
            "unused.yaml",
            tmp_path / "unused.pt",
            PriorConfig(),
            segmenter=BoxSegmenter(),
        )
    except FileNotFoundError as error:
        assert "Optical flow file not found" in str(error)
    else:
        raise AssertionError("Expected a missing-flow error")


def test_incomplete_page4d_arrays_fail_without_accessing_world_points(tmp_path):
    incomplete = tmp_path / "incomplete.npz"
    np.savez(incomplete, depth=np.ones((2, 3, 4), dtype=np.float32))
    try:
        load_page4d_geometry(incomplete)
    except KeyError as error:
        assert "missing keys" in str(error)
    else:
        raise AssertionError("Expected incomplete PAGE4D arrays to fail")


def test_page4d_depth_confidence_shape_mismatch_fails(tmp_path):
    predictions = tmp_path / "mismatch.npz"
    np.savez(
        predictions,
        image_paths=np.array(["0.png", "1.png"]),
        extrinsic=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        intrinsic=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        depth=np.ones((2, 3, 4), dtype=np.float32),
        depth_conf=np.ones((2, 2, 4), dtype=np.float32),
    )
    try:
        load_page4d_geometry(predictions)
    except ValueError as error:
        assert "depth-confidence shape mismatch" in str(error)
    else:
        raise AssertionError("Expected mismatched PAGE4D arrays to fail")


def _write_two_frame_fourrc(path: Path):
    height = width = 40
    intrinsic = np.array(
        [[100.0, 0, width / 2], [0, 100.0, height / 2], [0, 0, 1.0]],
        dtype=np.float32,
    )
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack(
        ((x - width / 2) / 100.0, (y - height / 2) / 100.0, np.ones_like(x)),
        axis=-1,
    ).astype(np.float32)
    arrays = {"n_frames": np.asarray(2)}
    for index in range(2):
        arrays[f"view_{index}_img"] = np.zeros((1, 3, height, width), dtype=np.float32)
        arrays[f"pred_{index}_pts"] = points[None]
        arrays[f"pred_{index}_conf"] = np.ones((1, height, width), dtype=np.float32)
        arrays[f"pred_{index}_intrinsic"] = intrinsic
        arrays[f"pred_{index}_extrinsic"] = np.eye(4, dtype=np.float32)
    np.savez(path, **arrays)
    return path


def test_fourrc_geometry_reaches_common_prior_core_with_matching_names(tmp_path):
    predictions = _write_two_frame_fourrc(tmp_path / "fourrc.npz")
    forward = tmp_path / "forward"
    backward = tmp_path / "backward"
    forward.mkdir()
    backward.mkdir()
    zero_flow = np.zeros((40, 40, 2), dtype=np.float32)
    np.save(forward / "flow_000000_to_000001.npy", zero_flow)
    np.save(backward / "flow_000001_to_000000.npy", zero_flow)
    output = tmp_path / "output"
    geometry = load_fourrc_geometry(predictions)
    assert np.allclose(geometry["depth"], 1.0)
    manifest = run_fourrc_prior_pipeline(
        predictions,
        forward,
        backward,
        output,
        "unused.yaml",
        tmp_path / "unused.pt",
        PriorConfig(min_component_area=1),
        device="cpu",
        segmenter=BoxSegmenter(),
        raft_metadata={"model_weights": "fake-raft"},
    )
    assert manifest["version"] == 3
    assert manifest["geometry_source"] == "fourrc"
    assert manifest["fourrc_predictions"] == str(predictions.resolve())
    assert manifest["raft_model_weights"] == "fake-raft"
    assert [frame["image_name"] for frame in manifest["frames"]] == [
        "frame_000000", "frame_000001"
    ]
    assert (output / "masks" / "frame_000000.png").is_file()
    _, mapping = load_prior_manifest(output)
    assert sorted(mapping) == ["frame_000000", "frame_000001"]
