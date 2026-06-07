from pathlib import Path

from patchdistill.distill import fit_detector_from_features, fit_proxy_from_files
from patchdistill.io import write_jsonl


def test_fit_proxy_from_mock_files(tmp_path: Path):
    features = [
        {"id": "a", "label": 1, "n_tokens": 10, "layer_0_last_norm": 1.0},
        {"id": "b", "label": 1, "n_tokens": 12, "layer_0_last_norm": 2.0},
        {"id": "c", "label": 1, "n_tokens": 8, "layer_0_last_norm": 0.5},
        {"id": "d", "label": 1, "n_tokens": 9, "layer_0_last_norm": 0.7},
    ]
    patches = [
        {"id": row["id"], "patch_signature": {"layer_0_pos_1": {"patch_effect": float(i)}}}
        for i, row in enumerate(features)
    ]
    feature_path = tmp_path / "features.jsonl"
    patch_path = tmp_path / "patch.jsonl"
    out = tmp_path / "out"
    write_jsonl(feature_path, features)
    write_jsonl(patch_path, patches)
    metrics = fit_proxy_from_files(feature_path, patch_path, out)
    assert metrics["n_aligned"] == 4
    assert (out / "proxy_metrics.json").exists()


def test_fit_detector_from_mock_features(tmp_path: Path):
    features = [
        {"id": "a", "label": 1, "n_tokens": 10, "layer_0_last_norm": 2.0},
        {"id": "b", "label": 1, "n_tokens": 12, "layer_0_last_norm": 2.2},
        {"id": "c", "label": 0, "n_tokens": 8, "layer_0_last_norm": 0.5},
        {"id": "d", "label": 0, "n_tokens": 9, "layer_0_last_norm": 0.7},
        {"id": "e", "label": 1, "n_tokens": 11, "layer_0_last_norm": 2.1},
        {"id": "f", "label": 0, "n_tokens": 7, "layer_0_last_norm": 0.4},
    ]
    feature_path = tmp_path / "features.jsonl"
    out = tmp_path / "detector"
    write_jsonl(feature_path, features)
    metrics = fit_detector_from_features(feature_path, out, test_size=0.33)
    assert metrics["model"] == "hf_features_logreg"
    assert "recall_at_fpr_0_1" in metrics["metrics"]
    assert (out / "detector_metrics.json").exists()
    assert (out / "predictions.csv").exists()


def test_fit_detector_supports_group_split(tmp_path: Path):
    features = [
        {"id": "a", "label": 1, "split_group": "frame_a", "n_tokens": 10, "layer_0_last_norm": 2.0},
        {"id": "b", "label": 0, "split_group": "frame_a", "n_tokens": 9, "layer_0_last_norm": 0.6},
        {"id": "c", "label": 1, "split_group": "frame_b", "n_tokens": 12, "layer_0_last_norm": 2.2},
        {"id": "d", "label": 0, "split_group": "frame_b", "n_tokens": 8, "layer_0_last_norm": 0.5},
        {"id": "e", "label": 1, "split_group": "frame_c", "n_tokens": 11, "layer_0_last_norm": 2.1},
        {"id": "f", "label": 0, "split_group": "frame_c", "n_tokens": 7, "layer_0_last_norm": 0.4},
        {"id": "g", "label": 1, "split_group": "frame_d", "n_tokens": 13, "layer_0_last_norm": 2.3},
        {"id": "h", "label": 0, "split_group": "frame_d", "n_tokens": 6, "layer_0_last_norm": 0.3},
    ]
    feature_path = tmp_path / "features.jsonl"
    out = tmp_path / "detector_group"
    write_jsonl(feature_path, features)
    metrics = fit_detector_from_features(feature_path, out, split="group", test_size=0.5, random_state=3)
    assert metrics["split"] == "group"
    assert "label" not in metrics["feature_names"]
    assert "recall_at_fpr_0_1" in metrics["metrics"]
    assert (out / "detector_metrics.json").exists()
    assert (out / "predictions.csv").exists()
