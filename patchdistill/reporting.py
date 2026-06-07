"""Collect experiment artifacts into compact JSON and Markdown summaries."""

from __future__ import annotations

import json
from pathlib import Path

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

