import numpy as np
from PIL import Image

from dynamic_split.evaluation import confusion_counts, evaluate_prior_agreement, metrics_from_counts


def test_mask_metrics():
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    counts = confusion_counts(prediction, target)
    assert counts == (1, 1, 1, 1)
    metrics = metrics_from_counts(*counts)
    assert metrics["iou"] == 1 / 3
    assert metrics["f1"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_prior_agreement_is_labeled_as_proxy(tmp_path):
    rendered = tmp_path / "rendered"
    priors = tmp_path / "priors"
    output = tmp_path / "evaluation"
    rendered.mkdir()
    (priors / "masks").mkdir(parents=True)
    prediction = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    target = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    Image.fromarray(prediction).save(rendered / "frame.png")
    Image.fromarray(target).save(priors / "masks" / "frame.png")
    (priors / "manifest.json").write_text(
        '{"frames": [{"image_name": "frame.png", "prior": "masks/frame.png"}]}'
    )

    report = evaluate_prior_agreement(rendered, priors, output)
    assert "not ground truth" in report["reference_type"]
    assert report["matching_frames"] == 1
    assert report["iou"] == 1 / 3
    assert report["predicted_dynamic_fraction"] == 0.5
    assert report["prior_dynamic_fraction"] == 0.5
    assert (output / "prior_agreement_metrics.json").is_file()
    assert (output / "per_frame_prior_agreement.csv").is_file()
