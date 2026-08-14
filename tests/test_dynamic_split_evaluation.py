import numpy as np

from dynamic_split.evaluation import confusion_counts, metrics_from_counts


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
