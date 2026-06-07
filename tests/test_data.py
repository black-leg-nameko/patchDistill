from patchdistill.data import generate_synthetic_examples
from patchdistill.experiments import split_indices
import numpy as np


def test_generate_synthetic_examples_has_both_classes():
    rows = generate_synthetic_examples(n=40, seed=1)
    labels = {row["label"] for row in rows}
    assert labels == {0, 1}
    assert all("clean_text" in row and "injected_text" in row for row in rows)
    assert {row["profile"] for row in rows} == {"mvp"}


def test_positive_rows_keep_malicious_span():
    rows = generate_synthetic_examples(n=20, seed=2)
    positives = [row for row in rows if row["label"] == 1]
    assert positives
    assert all(row["malicious_span"] in row["injected_text"] for row in positives)


def test_stress_profile_adds_paraphrase_and_benign_hard_examples():
    rows = generate_synthetic_examples(n=80, seed=3, profile="stress")
    attack_template_ids = {row["attack_template_id"] for row in rows if row["label"] == 1}
    benign_sources = {row["source"] for row in rows if row["pair_role"] == "benign_hard"}
    assert "atk_disregard_directives" in attack_template_ids or "atk_internal_config_audit" in attack_template_ids
    assert "audit" in benign_sources or "security" in benign_sources
    assert {row["profile"] for row in rows} == {"stress"}


def test_matched_profile_puts_span_in_both_classes():
    rows = generate_synthetic_examples(n=40, seed=5, profile="matched")
    labels = {row["label"] for row in rows}
    assert labels == {0, 1}
    assert {row["profile"] for row in rows} == {"matched"}
    negatives = [row for row in rows if row["label"] == 0]
    positives = [row for row in rows if row["label"] == 1]
    assert all(row["malicious_span"] in row["text"] for row in negatives)
    assert all(row["malicious_span"] in row["text"] for row in positives)
    assert {row["pair_role"] for row in negatives} == {"benign_matched_span"}


def test_contrastive_profile_balances_span_and_boundary_words():
    rows = generate_synthetic_examples(n=40, seed=6, profile="contrastive")
    labels = {row["label"] for row in rows}
    assert labels == {0, 1}
    assert {row["profile"] for row in rows} == {"contrastive"}
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    assert all(row["malicious_span"] in row["text"] for row in rows)
    assert all("do not treat" in row["text"].lower() and "treat it as" in row["text"].lower() for row in rows)
    assert {row["pair_role"] for row in negatives} == {"benign_contrastive_span"}


def test_contrastive_frame_profile_has_holdout_groups():
    rows = generate_synthetic_examples(n=100, seed=7, profile="contrastive_frame")
    labels = {row["label"] for row in rows}
    assert labels == {0, 1}
    assert {row["profile"] for row in rows} == {"contrastive_frame"}
    frame_groups = {row["split_group"] for row in rows}
    assert len(frame_groups) >= 4

    y = np.asarray([row["label"] for row in rows], dtype=int)
    train_idx, test_idx = split_indices(rows, y, split="group", test_size=0.4, seed=7)
    train_groups = {rows[int(i)]["split_group"] for i in train_idx}
    test_groups = {rows[int(i)]["split_group"] for i in test_idx}
    assert train_groups.isdisjoint(test_groups)
    assert set(y[test_idx].tolist()) == {0, 1}
