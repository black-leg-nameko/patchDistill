from pathlib import Path

from patchdistill.data import write_synthetic_dataset
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

