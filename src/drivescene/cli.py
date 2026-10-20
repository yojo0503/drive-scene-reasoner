"""CLI: dataset conversion, local evaluation and opt-in GPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (build_example, dump_jsonl, iter_jsonl, load_drivelm,
                   scene_split, to_swift_sft)
from .eval import evaluate


def prepare(args: argparse.Namespace) -> None:
    rows = load_drivelm(args.input)
    split = scene_split(rows, args.val_fraction, args.seed)
    samples: dict[str, list[dict]] = {"train": [], "val": []}
    for row in rows:
        if args.task != "all" and row.task != args.task:
            continue
        if row.answer is None:
            continue  # official challenge validation has no ground-truth answer
        item = build_example(row, args.image_root, args.cameras, not args.allow_missing_images)
        samples[split[row.scene_id]].append(item)
    out = Path(args.output)
    for name, group in samples.items():
        if args.limit_per_split > 0:
            group = group[:args.limit_per_split]
        dump_jsonl(out / f"{name}_manifest.jsonl", group)
        dump_jsonl(out / f"{name}_swift.jsonl", [to_swift_sft(s) for s in group])
        print(f"{name}: {len(group)} labeled examples")
    print(f"Files created under {out}; split is grouped by scene.")


def run_infer(args: argparse.Namespace) -> None:
    from .infer import QwenVLReasoner

    model = QwenVLReasoner(args.model)
    outputs = []
    for n, item in enumerate(iter_jsonl(args.manifest)):
        if args.limit and n >= args.limit:
            break
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
    prep.add_argument("--cameras", choices=["front", "all"], default="front")
    prep.add_argument("--task", choices=["all", "perception", "prediction", "planning", "behavior"], default="all")
    prep.add_argument("--val-fraction", type=float, default=0.1)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--limit-per-split", type=int, default=0)
    prep.add_argument("--allow-missing-images", action="store_true", help="FORMAT TEST ONLY; SFT will fail if images missing")
    prep.set_defaults(handler=prepare)

    infer = sub.add_parser("infer", help="Run Qwen3-VL inference from a prepared manifest")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    infer.add_argument("--output", default="outputs/predictions.jsonl")
    infer.add_argument("--limit", type=int, default=5)
    infer.add_argument("--max-new-tokens", type=int, default=128)
    infer.set_defaults(handler=run_infer)

    ev = sub.add_parser("eval", help="Local normalized exact match and token F1 (NOT official leaderboard)")
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--predictions", required=True)
    ev.set_defaults(handler=lambda args: print(json.dumps(evaluate(args.manifest, args.predictions), indent=2)))

    args = p.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
