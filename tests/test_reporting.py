from pathlib import Path

from patchdistill.io import write_json, write_jsonl
from patchdistill.reporting import archive_results, collect_results


def test_collect_results_skips_previous_summary(tmp_path: Path):
    runs = tmp_path / "runs"
    write_json(runs / "summary.json", {"old": True})
    write_json(runs / "model" / "detector_metrics.json", {"model": "x", "metrics": {"f1": 1.0}})
    summary = collect_results(runs)
    assert [item["path"] for item in summary["artifacts"]] == ["model/detector_metrics.json"]


def test_archive_results_copies_and_gzips_jsonl(tmp_path: Path):
    runs = tmp_path / "runs"
    archive_root = tmp_path / "artifacts"
    write_json(runs / "model" / "detector_metrics.json", {"model": "x", "metrics": {"f1": 1.0}})
    write_jsonl(runs / "features.jsonl", [{"id": "a", "value": 1}])
    manifest = archive_results(runs, archive_root, name="unit_test")
    archive_dir = archive_root / "unit_test"
    assert manifest["summary_artifacts"] == 1
    assert (archive_dir / "analysis.md").exists()
    assert (archive_dir / "features.jsonl.gz").exists()
    assert (archive_dir / "model" / "detector_metrics.json").exists()
