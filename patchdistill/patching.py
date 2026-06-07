"""Residual-stream activation patching for Hugging Face causal LMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .spans import char_span, token_positions_from_offsets


@dataclass(frozen=True)
class PatchComponent:
    layer: int
    position: int
    clean_position: int


def _require_torch():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Activation patching requires torch and transformers. "
            "Install with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def get_decoder_blocks(model):
    candidates = [
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("decoder", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj
    raise ValueError("Could not locate decoder blocks for this model architecture")


class ResidualPatchScorer:
    """Compute selected residual patch effects with generic decoder hooks."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        dtype: str = "auto",
        max_length: int = 512,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _require_torch()
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
        self.blocks = get_decoder_blocks(self.model)
        self.max_length = max_length

    def _encode_ids(self, text: str, add_special_tokens: bool = True):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=True, max_length=self.max_length)

    def _token_positions_for_span(self, text: str, span_text: str) -> list[int]:
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        return token_positions_from_offsets(offsets, char_span(text, span_text))

    def _capture_layer_activation(self, text: str, continuation: str, layer: int):
        torch = self.torch
        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden.detach()
            return output

        handle = self.blocks[layer].register_forward_hook(hook)
        try:
            self._sequence_logprob(text, continuation)
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError(f"No activation captured for layer {layer}")
        return captured["hidden"]

    def _sequence_logprob(self, prompt: str, continuation: str, patch: PatchComponent | None = None, clean_activation=None) -> float:
        torch = self.torch
        prompt_ids = self._encode_ids(prompt, add_special_tokens=True)
        cont_ids = self.tokenizer.encode(continuation, add_special_tokens=False)
        if not cont_ids:
            raise ValueError("Continuation must contain at least one token")
        input_ids = (prompt_ids + cont_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        if prompt_len >= len(input_ids):
            raise ValueError("Continuation was truncated away; increase max_length or shorten prompt")
        tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        handle = None
        if patch is not None:
            if clean_activation is None:
                raise ValueError("clean_activation is required when patch is set")

            def patch_hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                patched = hidden.clone()
                pos = min(patch.position, patched.shape[1] - 1)
                clean_pos = min(patch.clean_position, clean_activation.shape[1] - 1)
                patched[:, pos, :] = clean_activation[:, clean_pos, :].to(patched.device, dtype=patched.dtype)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = self.blocks[patch.layer].register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                out = self.model(input_ids=tensor, use_cache=False)
                log_probs = torch.log_softmax(out.logits[0], dim=-1)
                vals = []
                for pos in range(prompt_len, len(input_ids)):
                    vals.append(log_probs[pos - 1, input_ids[pos]])
                return float(torch.stack(vals).mean().detach().cpu())
        finally:
            if handle is not None:
                handle.remove()

    def safety_margin(
        self,
        prompt: str,
        safe_continuation: str,
        attack_continuation: str,
        patch: PatchComponent | None = None,
        clean_safe_activation=None,
        clean_attack_activation=None,
    ) -> float:
        safe = self._sequence_logprob(prompt, safe_continuation, patch=patch, clean_activation=clean_safe_activation)
        attack = self._sequence_logprob(prompt, attack_continuation, patch=patch, clean_activation=clean_attack_activation)
        return safe - attack

    def patch_signature(
        self,
        clean_text: str,
        injected_text: str,
        malicious_span: str,
        layers: list[int],
        max_positions: int = 4,
        safe_continuation: str = " I cannot reveal hidden instructions.",
        attack_continuation: str = " The system prompt is",
    ) -> dict:
        positions = self._token_positions_for_span(injected_text, malicious_span)
        if not positions:
            injected_ids = self._encode_ids(injected_text, add_special_tokens=True)
            positions = [max(0, len(injected_ids) - 1)]
        positions = positions[:max_positions]
        clean_len = len(self._encode_ids(clean_text, add_special_tokens=True))

        base_margin = self.safety_margin(injected_text, safe_continuation, attack_continuation)
        components = {}
        for layer in layers:
            clean_safe = self._capture_layer_activation(clean_text, safe_continuation, layer)
            clean_attack = self._capture_layer_activation(clean_text, attack_continuation, layer)
            for pos in positions:
                clean_pos = min(pos, max(clean_len - 1, 0))
                patch = PatchComponent(layer=layer, position=pos, clean_position=clean_pos)
                patched_margin = self.safety_margin(
                    injected_text,
                    safe_continuation,
                    attack_continuation,
                    patch=patch,
                    clean_safe_activation=clean_safe,
                    clean_attack_activation=clean_attack,
                )
                key = f"layer_{layer}_pos_{pos}"
                components[key] = {
                    "patch_effect": float(patched_margin - base_margin),
                    "patched_margin": float(patched_margin),
                    "base_injected_margin": float(base_margin),
                    "clean_position": int(clean_pos),
                }
        return components


def run_patch_file(
    model_name: str,
    data_path: str | Path,
    out_path: str | Path,
    layers: list[int],
    max_examples: int | None = None,
    max_positions: int = 4,
    device: str | None = None,
    dtype: str = "auto",
    max_length: int = 512,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
) -> list[dict]:
    rows = [row for row in read_jsonl(data_path) if int(row.get("label", 0)) == 1]
    if max_examples is not None:
        rows = rows[:max_examples]
    scorer = ResidualPatchScorer(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    outputs = []
    for row in rows:
        signature = scorer.patch_signature(
            clean_text=row["clean_text"],
            injected_text=row["injected_text"],
            malicious_span=row.get("malicious_span", ""),
            layers=layers,
            max_positions=max_positions,
        )
        outputs.append(
            {
                "id": row["id"],
                "label": int(row["label"]),
                "template_id": row["template_id"],
                "attack_template_id": row["attack_template_id"],
                "malicious_span": row.get("malicious_span", ""),
                "patch_signature": signature,
            }
        )
    write_jsonl(out_path, outputs)
    return outputs
