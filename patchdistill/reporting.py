"""Collect experiment artifacts into compact JSON and Markdown summaries."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .io import write_json


def _load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def collect_results(runs_dir: str | Path = "runs", out_path: str | Path | None = None) -> dict:
    root = Path(runs_dir)
    summary: dict = {"runs_dir": str(root), "artifacts": []}
    for path in sorted(root.rglob("*.json")):
        if path == root / "summary.json":
            continue
        payload = _load_json(path)
        if payload is None:
            continue
        rel = str(path.relative_to(root))
        kind = path.name.removesuffix(".json")
        item = {"path": rel, "kind": kind}
        if "models" in payload:
            item["models"] = payload["models"]
            item["n_total"] = payload.get("n_total")
            item["split"] = payload.get("split")
        elif "metrics" in payload and isinstance(payload["metrics"], dict):
            item["model"] = payload.get("model")
            item["metrics"] = payload["metrics"]
            item["n_total"] = payload.get("n_total")
        elif "mae" in payload or "mse" in payload:
            item["mae"] = payload.get("mae")
            item["mse"] = payload.get("mse")
            item["n_aligned"] = payload.get("n_aligned")
        else:
            item["keys"] = sorted(payload.keys())
        summary["artifacts"].append(item)

    if out_path is not None:
        write_json(out_path, summary)
    return summary


def write_markdown_summary(summary: dict, out_path: str | Path) -> None:
    lines = ["# PatchDistill Experiment Summary", ""]
    for item in summary.get("artifacts", []):
        lines.append(f"## `{item['path']}`")
        if "models" in item:
            lines.append(f"- split: `{item.get('split')}`")
            lines.append(f"- n_total: `{item.get('n_total')}`")
            for name, metrics in item["models"].items():
                f1 = metrics.get("f1")
                auroc = metrics.get("auroc")
                fnr = metrics.get("false_negative_rate")
                lines.append(f"- {name}: F1={f1}, AUROC={auroc}, FNR={fnr}")
        elif "metrics" in item:
            metrics = item["metrics"]
            lines.append(f"- model: `{item.get('model')}`")
            lines.append(f"- F1: `{metrics.get('f1')}`")
            lines.append(f"- AUROC: `{metrics.get('auroc')}`")
            lines.append(f"- FNR: `{metrics.get('false_negative_rate')}`")
        elif "mae" in item:
            lines.append(f"- MAE: `{item.get('mae')}`")
            lines.append(f"- MSE: `{item.get('mse')}`")
            lines.append(f"- n_aligned: `{item.get('n_aligned')}`")
        else:
            lines.append(f"- keys: `{item.get('keys')}`")
        lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _safe_archive_name(name: str | None) -> str:
    if not name:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    return safe or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iter_result_files(root: Path, include_jsonl: bool, include_csv: bool) -> Iterable[Path]:
    suffixes = {".json", ".md"}
    if include_csv:
        suffixes.add(".csv")
    if include_jsonl:
        suffixes.add(".jsonl")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            yield path


def archive_results(
    runs_dir: str | Path = "runs",
    archive_root: str | Path = "artifacts/colab_runs",
    name: str | None = None,
    include_jsonl: bool = True,
    include_csv: bool = True,
    max_file_mb: float = 50.0,
) -> dict:
    """Copy run outputs into a Git-trackable archive directory.

    JSONL files can grow quickly, so they are gzipped in the archive. Files
    larger than `max_file_mb` are skipped and listed in the manifest.
    """

    runs = Path(runs_dir)
    archive_name = _safe_archive_name(name)
    archive_dir = Path(archive_root) / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_results(runs_dir=runs, out_path=runs / "summary.json")
    write_markdown_summary(summary, runs / "summary.md")
    write_markdown_summary(summary, archive_dir / "analysis.md")
    write_json(archive_dir / "summary.json", summary)

    max_bytes = int(max_file_mb * 1024 * 1024)
    copied: list[dict] = []
    skipped: list[dict] = []
    for src in _iter_result_files(runs, include_jsonl=include_jsonl, include_csv=include_csv):
        rel = src.relative_to(runs)
        size = src.stat().st_size
        if size > max_bytes:
            skipped.append({"path": str(rel), "bytes": size, "reason": "larger_than_max_file_mb"})
            continue

        if src.suffix == ".jsonl":
            dest = archive_dir / rel.with_suffix(rel.suffix + ".gz")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src.open("rb") as f_in, gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            dest = archive_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        copied.append({"source": str(rel), "archive": str(dest.relative_to(archive_dir)), "bytes": size})

    manifest = {
        "archive_name": archive_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs),
        "archive_dir": str(archive_dir),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "copied": copied,
        "skipped": skipped,
        "summary_artifacts": len(summary.get("artifacts", [])),
    }
    write_json(archive_dir / "manifest.json", manifest)
    return manifest
