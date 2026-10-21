"""CLI: dataset conversion, local evaluation and opt-in GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import dump_jsonl, iter_jsonl
from .eval import evaluate


def prepare(args: argparse.Namespace) -> None:
    from .prepare import prepare_dataset
    prepare_dataset(args)


def run_infer(args: argparse.Namespace) -> None:
    from .infer import QwenVLReasoner

    model = QwenVLReasoner(args.model, adapter_path=args.adapter,
        min_pixels=args.min_pixels, max_pixels=args.max_pixels, device_map=args.device_map,
        revision=args.revision, local_files_only=args.local_files_only)
    outputs = []
    for n, item in enumerate(iter_jsonl(args.manifest)):
        if args.limit and n >= args.limit:
            break
        if not item["question"].strip():
            outputs.append({"id": item["id"], "prediction": "", "status": "invalid_source_question"})
            print(f"{n + 1} {item['id']} => empty source question; retained with empty prediction")
            continue
        prediction = model.generate(item, max_new_tokens=args.max_new_tokens)
        outputs.append({"id": item["id"], "prediction": prediction})
        print(f"{n + 1} {item['id']} => {prediction[:180]}")
    n = dump_jsonl(args.output, outputs)
    print(f"Saved {n} predictions to {args.output}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="drivescene")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="DriveLM JSON -> ms-swift JSONL with scene-level holdout")
    prep.add_argument("--input", required=True, help="DriveLM v1.0/v1.1 training JSON")
    prep.add_argument("--image-root", required=True, help="nuScenes root containing samples/ or samples/ itself")
    prep.add_argument("--output", default="data/processed")
    prep.add_argument("--cameras", choices=["front", "all"], default="all")
    prep.add_argument("--purpose", choices=["sft", "infer"], default="sft", help="infer retains unanswered questions without splitting")
    prep.add_argument("--task", choices=["all", "perception", "prediction", "planning", "behavior"], default="all")
    prep.add_argument("--val-fraction", type=float, default=0.1)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--limit-per-split", type=int, default=0)
    prep.add_argument("--allow-missing-images", action="store_true", help="FORMAT TEST ONLY; SFT will fail if images missing")
    prep.set_defaults(handler=prepare)

    infer = sub.add_parser("infer", help="Run Qwen3-VL inference from a prepared manifest")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    infer.add_argument("--adapter", help="LoRA adapter directory, loaded onto --model")
    infer.add_argument("--revision", help="Pin a Hugging Face model commit")
    infer.add_argument("--local-files-only", action="store_true")
    infer.add_argument("--device-map", default="auto")
    infer.add_argument("--min-pixels", type=int, default=65536)
    infer.add_argument("--max-pixels", type=int, default=262144)
    infer.add_argument("--output", default="outputs/predictions.jsonl")
    infer.add_argument("--limit", type=int, default=5)
    infer.add_argument("--max-new-tokens", type=int, default=128)
    infer.set_defaults(handler=run_infer)

    ev = sub.add_parser("eval", help="Local normalized exact match and token F1 (NOT official leaderboard)")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--missing-policy", choices=["error", "wrong"], default="error")
    ev.set_defaults(handler=lambda args: print(json.dumps(evaluate(args.manifest, args.predictions, args.missing_policy), indent=2)))

    official = sub.add_parser("export-official", help="Export complete predictions with upstream frame-global IDs")
    official.add_argument("--manifest", required=True)
    official.add_argument("--predictions", required=True)
    official.add_argument("--output", required=True)
    official.set_defaults(handler=run_official_export)
    tag0 = sub.add_parser("eval-tag0", help="Exact tag-0 subset only, without upstream graph gating")
    tag0.add_argument("--manifest", required=True)
    tag0.add_argument("--predictions", required=True)
    tag0.set_defaults(handler=run_tag0)

    args = p.parse_args(argv)
    args.handler(args)


def run_official_export(args):
    from .protocol import export_official
    print(f"Exported {export_official(args.manifest, args.predictions, args.output)} official results")


def run_tag0(args):
    from .protocol import evaluate_tag0
    print(json.dumps(evaluate_tag0(args.manifest, args.predictions), indent=2))


if __name__ == "__main__":
    main()
