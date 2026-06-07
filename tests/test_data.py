from patchdistill.data import generate_synthetic_examples


def test_generate_synthetic_examples_has_both_classes():
    rows = generate_synthetic_examples(n=40, seed=1)
    labels = {row["label"] for row in rows}
    assert labels == {0, 1}
    assert all("clean_text" in row and "injected_text" in row for row in rows)


def test_positive_rows_keep_malicious_span():
    rows = generate_synthetic_examples(n=20, seed=2)
    positives = [row for row in rows if row["label"] == 1]
    assert positives
    assert all(row["malicious_span"] in row["injected_text"] for row in positives)

