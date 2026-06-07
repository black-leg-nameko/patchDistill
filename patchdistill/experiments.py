"""Local PatchDistill experiment pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .io import ensure_parent, read_jsonl, write_json
from .lexical import build_rule_features, pseudo_patch_signature


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    fnr = float(((y_pred == 0) & positives).sum() / max(positives.sum(), 1))
    fpr = float(((y_pred == 1) & negatives).sum() / max(negatives.sum(), 1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": _safe_auc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
    }


def _split_key(row: dict, split: str) -> str:
    if split == "template":
        return f"{row['template_id']}::{row['attack_template_id']}"
    if split == "group":
        return str(row.get("split_group") or f"{row['template_id']}::{row['attack_template_id']}")
    raise ValueError(f"Unknown split key type: {split}")


def split_indices(rows: list[dict], y: np.ndarray, split: str, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if split == "random":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros_like(y), y))
        return train_idx, test_idx

    if split not in {"template", "group"}:
        raise ValueError(f"Unknown split: {split}")

    keys = sorted({_split_key(row, split) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test_keys = max(1, int(round(len(keys) * test_size)))
    test_keys = set(keys[:n_test_keys])
    train_idx: list[int] = []
    test_idx: list[int] = []
    for i, row in enumerate(rows):
        key = _split_key(row, split)
        if key in test_keys:
            test_idx.append(i)
        else:
            train_idx.append(i)

    if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(splitter.split(np.zeros_like(y), y))
    return np.asarray(train_idx), np.asarray(test_idx)


def _positive_probability(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def run_surrogate_experiment(
    data_path: str | Path,
    out_dir: str | Path,
    split: str = "template",
    test_size: float = 0.25,
    random_state: int = 13,
    pseudo_signature_dim: int = 12,
) -> dict:
    rows = read_jsonl(data_path)
    texts = [row["text"] for row in rows]
    y = np.asarray([row["label"] for row in rows], dtype=int)
    train_idx, test_idx = split_indices(rows, y, split=split, test_size=test_size, seed=random_state)

    rule_x, rule_names = build_rule_features(texts)
    pe = pseudo_patch_signature(rule_x, dim=pseudo_signature_dim, seed=random_state)

    tfidf_model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    tfidf_model.fit([texts[i] for i in train_idx], y[train_idx])
    tfidf_score = _positive_probability(tfidf_model, [texts[i] for i in test_idx])

    rule_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    rule_model.fit(rule_x[train_idx], y[train_idx])
    rule_score = _positive_probability(rule_model, rule_x[test_idx])

    proxy_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_state))
    proxy_model.fit(rule_x[train_idx], pe[train_idx])
    pe_hat_train = proxy_model.predict(rule_x[train_idx])
    pe_hat_test = proxy_model.predict(rule_x[test_idx])

    distilled_train = np.hstack([rule_x[train_idx], pe_hat_train])
    distilled_test = np.hstack([rule_x[test_idx], pe_hat_test])
    distilled_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
    )
    distilled_model.fit(distilled_train, y[train_idx])
    distilled_score = _positive_probability(distilled_model, distilled_test)

    metrics = {
        "data_path": str(data_path),
        "split": split,
        "test_size": test_size,
        "random_state": random_state,
        "n_total": len(rows),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_names": rule_names,
        "models": {
            "tfidf_logreg": classification_metrics(y[test_idx], tfidf_score),
            "rule_logreg": classification_metrics(y[test_idx], rule_score),
            "patchdistill_surrogate": classification_metrics(y[test_idx], distilled_score),
        },
    }

    proxy_metrics = {
        "target": "pseudo_patch_signature",
        "mae": float(mean_absolute_error(pe[test_idx], pe_hat_test)),
        "mse": float(mean_squared_error(pe[test_idx], pe_hat_test)),
        "note": "Use hf-patch outputs for real causal patch signature targets.",
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    write_json(out / "proxy_metrics.json", proxy_metrics)

    pred_path = ensure_parent(out / "predictions.csv")
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "label",
                "pair_role",
                "template_id",
                "attack_template_id",
                "split_group",
                "tfidf_score",
                "rule_score",
                "patchdistill_surrogate_score",
            ],
        )
        writer.writeheader()
        for row_idx, tfidf_s, rule_s, dist_s in zip(test_idx, tfidf_score, rule_score, distilled_score, strict=True):
            row = rows[int(row_idx)]
            writer.writerow(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "pair_role": row["pair_role"],
                    "template_id": row["template_id"],
                    "attack_template_id": row["attack_template_id"],
                    "split_group": row.get("split_group", ""),
                    "tfidf_score": float(tfidf_s),
                    "rule_score": float(rule_s),
                    "patchdistill_surrogate_score": float(dist_s),
                }
            )

    return {"metrics": metrics, "proxy_metrics": proxy_metrics}
