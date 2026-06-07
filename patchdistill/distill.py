"""Proxy and detector training on HF feature and patch-signature files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .experiments import classification_metrics
from .experiments import split_indices
from .io import ensure_parent, read_jsonl, write_json


ID_KEYS = {
    "id",
    "label",
    "template_id",
    "attack_template_id",
    "split_group",
    "pair_role",
    "profile",
    "source",
    "language",
    "malicious_span",
    "patch_signature",
}


def numeric_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in ID_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    x = np.asarray([[float(row.get(name, 0.0)) for name in names] for row in rows], dtype=np.float64)
    return x, names


def flatten_patch_signatures(rows: list[dict]) -> tuple[dict[str, np.ndarray], list[str]]:
    component_names = sorted(
        {
            component
            for row in rows
            for component, payload in row.get("patch_signature", {}).items()
            if isinstance(payload, dict) and "patch_effect" in payload
        }
    )
    by_id: dict[str, np.ndarray] = {}
    for row in rows:
        signature = row.get("patch_signature", {})
        by_id[row["id"]] = np.asarray(
            [float(signature.get(name, {}).get("patch_effect", 0.0)) for name in component_names],
            dtype=np.float64,
        )
    return by_id, component_names


def fit_proxy_from_files(
    features_path: str | Path,
    patch_path: str | Path,
    out_dir: str | Path,
    test_size: float = 0.25,
    random_state: int = 13,
) -> dict:
    features = read_jsonl(features_path)
    patch_rows = read_jsonl(patch_path)
    patch_by_id, patch_names = flatten_patch_signatures(patch_rows)
    feature_by_id = {row["id"]: row for row in features}
    ids = sorted(set(feature_by_id) & set(patch_by_id))
    if len(ids) < 2:
        raise ValueError("Need at least two IDs present in both feature and patch files")
    if not patch_names:
        raise ValueError("Patch file does not contain any patch_effect components")

    aligned_features = [feature_by_id[id_] for id_ in ids]
    x, feature_names = numeric_matrix(aligned_features)
    y = np.vstack([patch_by_id[id_] for id_ in ids])
    if len(ids) < 4:
        train_idx = np.arange(len(ids))
        test_idx = np.arange(len(ids))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(ids)), test_size=test_size, random_state=random_state)

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_state))
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    metrics = {
        "features_path": str(features_path),
        "patch_path": str(patch_path),
        "n_aligned": int(len(ids)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_names": feature_names,
        "patch_component_names": patch_names,
        "mae": float(mean_absolute_error(y[test_idx], pred)),
        "mse": float(mean_squared_error(y[test_idx], pred)),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "proxy_metrics.json", metrics)
    return metrics


def fit_detector_from_features(
    features_path: str | Path,
    out_dir: str | Path,
    patch_path: str | Path | None = None,
    test_size: float = 0.25,
    random_state: int = 13,
    split: str = "random",
) -> dict:
    rows = read_jsonl(features_path)
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    x, feature_names = numeric_matrix(rows)
    model_name = "hf_features_logreg"

    proxy_note = None
    if patch_path is not None:
        patch_rows = read_jsonl(patch_path)
        patch_by_id, patch_names = flatten_patch_signatures(patch_rows)
        feature_by_id = {row["id"]: i for i, row in enumerate(rows)}
        aligned = sorted(set(feature_by_id) & set(patch_by_id))
        if len(aligned) >= 2 and patch_names:
            aligned_idx = np.asarray([feature_by_id[id_] for id_ in aligned], dtype=int)
            pe = np.vstack([patch_by_id[id_] for id_ in aligned])
            proxy = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_state))
            proxy.fit(x[aligned_idx], pe)
            pe_hat = proxy.predict(x)
            x = np.hstack([x, pe_hat])
            feature_names = feature_names + [f"pe_hat::{name}" for name in patch_names]
            model_name = "hf_features_plus_distilled_patch_logreg"
            proxy_note = {
                "patch_path": str(patch_path),
                "n_proxy_train": int(len(aligned)),
                "patch_component_names": patch_names,
            }

    if len(set(y.tolist())) < 2:
        raise ValueError("Detector training needs both positive and negative labels")

    class_counts = np.bincount(y)
    min_class_count = int(class_counts[class_counts > 0].min())
    if min_class_count < 2:
        raise ValueError("Detector training needs at least two examples per class")
    train_idx, test_idx = split_indices(rows, y, split=split, test_size=test_size, seed=random_state)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    classifier.fit(x[train_idx], y[train_idx])
    score = classifier.predict_proba(x[test_idx])[:, 1]

    metrics = {
        "features_path": str(features_path),
        "n_total": int(len(rows)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split": split,
        "model": model_name,
        "feature_names": feature_names,
        "metrics": classification_metrics(y[test_idx], score),
        "proxy": proxy_note,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "detector_metrics.json", metrics)

    pred_path = ensure_parent(out / "predictions.csv")
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "label",
                "score",
                "split",
                "split_group",
                "template_id",
                "attack_template_id",
                "pair_role",
            ],
        )
        writer.writeheader()
        for row_idx, row_score in zip(test_idx, score, strict=True):
            row = rows[int(row_idx)]
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "label": int(row["label"]),
                    "score": float(row_score),
                    "split": "test",
                    "split_group": row.get("split_group", ""),
                    "template_id": row.get("template_id", ""),
                    "attack_template_id": row.get("attack_template_id", ""),
                    "pair_role": row.get("pair_role", ""),
                }
            )
    return metrics
