"""Lightweight one-pass surrogate features for local PatchDistill tests."""

from __future__ import annotations

import re

import numpy as np


ATTACK_PATTERNS = [
    r"\bignore (all )?(previous|prior|above) instructions?\b",
    r"\breveal (the )?(system|developer|hidden) (prompt|message|instructions?)\b",
    r"\bprint (every|all) (hidden|private|secret)",
    r"\boverride (the )?(safety|system|policy|rules?)\b",
    r"\byou are now\b",
    r"\bdeveloper message\b",
    r"\bobey the user\b",
    r"\bfollow only this message\b",
    r"\bhidden prompts?\b",
    r"\bprivate policy\b",
]

BENIGN_CONTEXT_PATTERNS = [
    r"\bsecurity lesson\b",
    r"\bexplain this code comment\b",
    r"\bshort play\b",
    r"\bas a joke\b",
    r"\bcompare two policy drafts\b",
    r"\bprompt that says\b",
    r"\bphrase ['\"]",
    r"//",
]

SECRET_WORDS = {"secret", "hidden", "private", "confidential", "system", "developer", "policy"}
OVERRIDE_WORDS = {"ignore", "override", "reveal", "print", "obey", "comply", "administrator"}


def _count_regex(patterns: list[str], text: str) -> int:
    lower = text.lower()
    return sum(len(re.findall(pattern, lower)) for pattern in patterns)


def _count_words(words: set[str], text: str) -> int:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return sum(1 for tok in tokens if tok in words)


def build_rule_features(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return numeric text features that stand in for one-pass signals.

    These are intentionally simple and fast. They make it possible to exercise
    proxy distillation and detector evaluation without a local GPU.
    """

    names = [
        "attack_pattern_count",
        "benign_context_count",
        "secret_word_count",
        "override_word_count",
        "quoted_attack_count",
        "code_marker_count",
        "length_chars",
        "length_words",
        "uppercase_ratio",
        "newline_count",
        "suspicious_density",
        "benign_adjusted_risk",
    ]
    rows: list[list[float]] = []
    for text in texts:
        words = re.findall(r"[a-zA-Z]+", text)
        n_words = max(len(words), 1)
        attack_count = _count_regex(ATTACK_PATTERNS, text)
        benign_count = _count_regex(BENIGN_CONTEXT_PATTERNS, text)
        secret_count = _count_words(SECRET_WORDS, text)
        override_count = _count_words(OVERRIDE_WORDS, text)
        quoted_attack_count = len(re.findall(r"['\"][^'\"]*(ignore|reveal|override|system prompt)[^'\"]*['\"]", text.lower()))
        code_marker_count = text.count("//") + text.count("```")
        uppercase = sum(1 for ch in text if ch.isupper())
        letters = sum(1 for ch in text if ch.isalpha())
        uppercase_ratio = uppercase / max(letters, 1)
        suspicious_density = (attack_count + 0.5 * secret_count + override_count) / n_words
        benign_adjusted = suspicious_density - 0.35 * benign_count - 0.2 * quoted_attack_count - 0.2 * code_marker_count
        rows.append(
            [
                attack_count,
                benign_count,
                secret_count,
                override_count,
                quoted_attack_count,
                code_marker_count,
                len(text),
                n_words,
                uppercase_ratio,
                text.count("\n"),
                suspicious_density,
                benign_adjusted,
            ]
        )
    return np.asarray(rows, dtype=np.float64), names


def pseudo_patch_signature(rule_features: np.ndarray, dim: int = 12, seed: int = 13) -> np.ndarray:
    """Create deterministic pseudo PE vectors for local pipeline tests.

    Real patch signatures come from `hf-patch`. This surrogate target is only
    used to verify that the distillation and detector code behaves correctly
    before GPU experiments are available.
    """

    rng = np.random.default_rng(seed)
    base_weights = np.zeros((rule_features.shape[1], dim), dtype=np.float64)
    for j in range(dim):
        layer_phase = (j + 1) / dim
        base_weights[0, j] = 0.9 + 0.3 * np.sin(layer_phase * np.pi)
        base_weights[1, j] = -0.45 - 0.1 * np.cos(layer_phase * np.pi)
        base_weights[2, j] = 0.10 + 0.05 * j
        base_weights[3, j] = 0.22 + 0.04 * np.cos(j)
        base_weights[10, j] = 3.0 + 0.25 * j
        base_weights[11, j] = 1.3 + 0.15 * np.sin(j)

    noise_weights = rng.normal(loc=0.0, scale=0.015, size=base_weights.shape)
    pe = rule_features @ (base_weights + noise_weights)
    pe = np.tanh(pe / 3.0)
    return pe.astype(np.float64)

