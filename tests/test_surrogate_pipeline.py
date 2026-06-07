from pathlib import Path

from patchdistill.data import write_synthetic_dataset
from patchdistill.distill import numeric_matrix
from patchdistill.experiments import run_surrogate_experiment


def test_surrogate_pipeline_writes_outputs(tmp_path: Path):
    data_path = tmp_path / "data.jsonl"
    out_dir = tmp_path / "run"
    write_synthetic_dataset(data_path, n=80, seed=4)
    result = run_surrogate_experiment(data_path, out_dir, split="random", test_size=0.25, random_state=4)
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "proxy_metrics.json").exists()
    assert (out_dir / "predictions.csv").exists()
    assert "patchdistill_surrogate" in result["metrics"]["models"]


def test_numeric_matrix_excludes_label_and_metadata():
    rows = [
        {
            "id": "a",
            "label": 0,
            "template_id": "t1",
            "split_group": "g1",
            "n_tokens": 10,
            "score": 0.25,
        },
        {
            "id": "b",
            "label": 1,
            "template_id": "t2",
            "split_group": "g2",
            "n_tokens": 12,
            "score": 0.75,
        },
    ]
    x, names = numeric_matrix(rows)
    assert names == ["n_tokens", "score"]
    assert x.shape == (2, 2)
