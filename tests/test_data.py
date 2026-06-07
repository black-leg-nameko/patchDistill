from patchdistill.data import generate_synthetic_examples


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
