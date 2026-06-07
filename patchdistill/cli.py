"""Command-line entry points for PatchDistill experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import write_synthetic_dataset
from .experiments import run_surrogate_experiment


def _parse_layers(value: str) -> list[int]:
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            layers.append(int(part))
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return layers


def cmd_make_data(args: argparse.Namespace) -> None:
    rows = write_synthetic_dataset(args.out, n=args.n, seed=args.seed)
    print(json.dumps({"out": str(args.out), "n": len(rows)}, ensure_ascii=False))


def cmd_run_surrogate(args: argparse.Namespace) -> None:
    result = run_surrogate_experiment(
        data_path=args.data,
        out_dir=args.out,
        split=args.split,
        test_size=args.test_size,
        random_state=args.seed,
        pseudo_signature_dim=args.pseudo_signature_dim,
    )
    print(json.dumps({"out": str(args.out), "models": result["metrics"]["models"]}, ensure_ascii=False, indent=2))


def cmd_hf_extract(args: argparse.Namespace) -> None:
    from .hf import extract_file

    rows = extract_file(
        model_name=args.model,
        data_path=args.data,
        out_path=args.out,
        max_examples=args.max_examples,
        layers=args.layers,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps({"out": str(args.out), "n": len(rows)}, ensure_ascii=False))


def cmd_hf_patch(args: argparse.Namespace) -> None:
    from .patching import run_patch_file

    rows = run_patch_file(
        model_name=args.model,
        data_path=args.data,
        out_path=args.out,
        layers=args.layers,
        max_examples=args.max_examples,
        max_positions=args.max_positions,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps({"out": str(args.out), "n": len(rows)}, ensure_ascii=False))


def cmd_fit_proxy(args: argparse.Namespace) -> None:
    from .distill import fit_proxy_from_files

    metrics = fit_proxy_from_files(
        features_path=args.features,
        patch_path=args.patch,
        out_dir=args.out,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(json.dumps({"out": str(args.out), "mae": metrics["mae"], "mse": metrics["mse"]}, ensure_ascii=False, indent=2))


def cmd_fit_detector(args: argparse.Namespace) -> None:
    from .distill import fit_detector_from_features

    metrics = fit_detector_from_features(
        features_path=args.features,
        patch_path=args.patch,
        out_dir=args.out,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(json.dumps({"out": str(args.out), "model": metrics["model"], "metrics": metrics["metrics"]}, ensure_ascii=False, indent=2))


def cmd_collect_results(args: argparse.Namespace) -> None:
    from .reporting import collect_results, write_markdown_summary

    summary = collect_results(runs_dir=args.runs, out_path=args.out)
    if args.markdown:
        write_markdown_summary(summary, args.markdown)
    print(json.dumps({"out": str(args.out), "markdown": str(args.markdown) if args.markdown else None, "n": len(summary["artifacts"])}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchDistill experiment commands")
    sub = parser.add_subparsers(dest="command", required=True)

    make_data = sub.add_parser("make-data", help="Create synthetic direct PI JSONL data")
    make_data.add_argument("--n", type=int, default=160)
    make_data.add_argument("--seed", type=int, default=13)
    make_data.add_argument("--out", type=Path, required=True)
    make_data.set_defaults(func=cmd_make_data)

    surrogate = sub.add_parser("run-surrogate", help="Run local surrogate PatchDistill experiment")
    surrogate.add_argument("--data", type=Path, required=True)
    surrogate.add_argument("--out", type=Path, required=True)
    surrogate.add_argument("--split", choices=["random", "template"], default="template")
    surrogate.add_argument("--test-size", type=float, default=0.25)
    surrogate.add_argument("--seed", type=int, default=13)
    surrogate.add_argument("--pseudo-signature-dim", type=int, default=12)
    surrogate.set_defaults(func=cmd_run_surrogate)

    hf_extract = sub.add_parser("hf-extract", help="Extract one-pass hidden/attention/logit features")
    hf_extract.add_argument("--model", required=True)
    hf_extract.add_argument("--data", type=Path, required=True)
    hf_extract.add_argument("--out", type=Path, required=True)
    hf_extract.add_argument("--max-examples", type=int, default=None)
    hf_extract.add_argument("--layers", default="auto")
    hf_extract.add_argument("--device", default=None)
    hf_extract.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    hf_extract.add_argument("--max-length", type=int, default=512)
    hf_extract.add_argument("--trust-remote-code", action="store_true")
    hf_extract.add_argument("--attn-implementation", default=None)
    hf_extract.set_defaults(func=cmd_hf_extract)

    hf_patch = sub.add_parser("hf-patch", help="Compute selected residual patch signatures")
    hf_patch.add_argument("--model", required=True)
    hf_patch.add_argument("--data", type=Path, required=True)
    hf_patch.add_argument("--out", type=Path, required=True)
    hf_patch.add_argument("--layers", type=_parse_layers, required=True)
    hf_patch.add_argument("--max-examples", type=int, default=None)
    hf_patch.add_argument("--max-positions", type=int, default=4)
    hf_patch.add_argument("--device", default=None)
    hf_patch.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    hf_patch.add_argument("--max-length", type=int, default=512)
    hf_patch.add_argument("--trust-remote-code", action="store_true")
    hf_patch.add_argument("--attn-implementation", default=None)
    hf_patch.set_defaults(func=cmd_hf_patch)

    fit_proxy = sub.add_parser("fit-proxy", help="Fit q_phi(r(x)) ~= PE(x) from HF feature and patch files")
    fit_proxy.add_argument("--features", type=Path, required=True)
    fit_proxy.add_argument("--patch", type=Path, required=True)
    fit_proxy.add_argument("--out", type=Path, required=True)
    fit_proxy.add_argument("--test-size", type=float, default=0.25)
    fit_proxy.add_argument("--seed", type=int, default=13)
    fit_proxy.set_defaults(func=cmd_fit_proxy)

    fit_detector = sub.add_parser("fit-detector", help="Fit detector on HF features, optionally with distilled PE")
    fit_detector.add_argument("--features", type=Path, required=True)
    fit_detector.add_argument("--patch", type=Path, default=None)
    fit_detector.add_argument("--out", type=Path, required=True)
    fit_detector.add_argument("--test-size", type=float, default=0.25)
    fit_detector.add_argument("--seed", type=int, default=13)
    fit_detector.set_defaults(func=cmd_fit_detector)

    collect = sub.add_parser("collect-results", help="Collect JSON result artifacts under runs/")
    collect.add_argument("--runs", type=Path, default=Path("runs"))
    collect.add_argument("--out", type=Path, default=Path("runs/summary.json"))
    collect.add_argument("--markdown", type=Path, default=Path("runs/summary.md"))
    collect.set_defaults(func=cmd_collect_results)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
