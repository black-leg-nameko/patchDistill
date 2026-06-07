"""Optional Hugging Face feature extraction for GPU/Colab experiments."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .io import read_jsonl, write_jsonl
from .spans import char_span, token_positions_from_offsets


def _require_hf():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face experiments require torch and transformers. "
            "Install with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _select_layers(n_layers: int, layer_spec: str | None) -> list[int]:
    if n_layers <= 0:
        return []
    if not layer_spec or layer_spec == "auto":
        return sorted({0, n_layers // 2, n_layers - 1})
    layers: list[int] = []
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer {part} is out of range for {n_layers} layers")
        layers.append(idx)
    return sorted(set(layers))


def _entropy(prob: np.ndarray, axis: int = -1) -> np.ndarray:
    clipped = np.clip(prob, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=axis)


class HFOnePassExtractor:
    """Extract hidden-state, attention, and logit summary features."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: str = "auto",
        layers: str | None = "auto",
        max_length: int = 512,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _require_hf()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.max_length = max_length
        n_layers = getattr(self.model.config, "num_hidden_layers", None) or getattr(self.model.config, "n_layer", 0)
        self.layers = _select_layers(int(n_layers), layers)

    def _span_positions(self, text: str, span_text: str) -> list[int]:
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        return token_positions_from_offsets(offsets, char_span(text, span_text))

    def extract_one(self, row: dict) -> dict:
        torch = self.torch
        text = row["text"]
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        span_positions = self._span_positions(text, row.get("malicious_span", ""))
        with torch.no_grad():
            out = self.model(**encoded, output_hidden_states=True, output_attentions=True, use_cache=False)

        result: dict[str, float | int | str] = {
            "id": row["id"],
            "label": int(row["label"]),
            "template_id": row.get("template_id", ""),
            "attack_template_id": row.get("attack_template_id", ""),
            "split_group": row.get("split_group", ""),
            "pair_role": row.get("pair_role", ""),
            "profile": row.get("profile", ""),
            "source": row.get("source", ""),
            "language": row.get("language", ""),
            "n_tokens": int(encoded["input_ids"].shape[1]),
            "n_span_tokens": int(len(span_positions)),
        }

        hidden_states = out.hidden_states[1:] if out.hidden_states is not None else []
        for layer in self.layers:
            hs = hidden_states[layer][0].detach().float().cpu().numpy()
            last = hs[-1]
            result[f"layer_{layer}_last_norm"] = float(np.linalg.norm(last))
            result[f"layer_{layer}_mean_norm"] = float(np.linalg.norm(hs.mean(axis=0)))
            result[f"layer_{layer}_token_std"] = float(hs.std(axis=0).mean())
            if span_positions:
                valid = [pos for pos in span_positions if pos < hs.shape[0]]
                if valid:
                    span_hs = hs[valid]
                    result[f"layer_{layer}_span_mean_norm"] = float(np.linalg.norm(span_hs.mean(axis=0)))
                    result[f"layer_{layer}_span_token_std"] = float(span_hs.std(axis=0).mean())

        attentions = out.attentions or []
        for layer in self.layers:
            if layer >= len(attentions):
                continue
            attn = attentions[layer][0].detach().float().cpu().numpy()
            # shape: heads, query, key
            ent = _entropy(attn, axis=-1)
            result[f"layer_{layer}_attn_entropy_mean"] = float(ent.mean())
            result[f"layer_{layer}_attn_entropy_last"] = float(ent[:, -1].mean())
            if span_positions:
                valid = [pos for pos in span_positions if pos < attn.shape[-1]]
                if valid:
                    result[f"layer_{layer}_span_attn_mass_mean"] = float(attn[:, :, valid].sum(axis=-1).mean())
                    result[f"layer_{layer}_span_attn_mass_last"] = float(attn[:, -1, valid].sum(axis=-1).mean())

        logits = out.logits[0, -1].detach().float()
        log_probs = torch.log_softmax(logits, dim=-1)
        for name, token in {"refuse": " refuse", "reveal": " reveal", "ignore": " ignore"}.items():
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if ids:
                result[f"logprob_{name}"] = float(log_probs[ids[0]].cpu())
        return result


def extract_file(
    model_name: str,
    data_path: str | Path,
    out_path: str | Path,
    max_examples: int | None = None,
    layers: str | None = "auto",
    device: str | None = None,
    dtype: str = "auto",
    max_length: int = 512,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
) -> list[dict]:
    rows = read_jsonl(data_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    extractor = HFOnePassExtractor(
        model_name=model_name,
        device=device,
        dtype=dtype,
        layers=layers,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    features = [extractor.extract_one(row) for row in rows]
    write_jsonl(out_path, features)
    return features
