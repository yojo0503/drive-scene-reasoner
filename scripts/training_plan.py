#!/usr/bin/env python3
"""Validate and snapshot an ms-swift LoRA command. Default: plan only, no model load."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drivescene.data import iter_jsonl
from drivescene.prepare import sha256_file


def build_plan(args):
    train, val = Path(args.dataset).resolve(), Path(args.val_dataset).resolve()
    if train.parent != val.parent:
        raise ValueError("Train and validation must come from the same preparation directory")
    report = json.loads((train.parent / "preparation.json").read_text())
    if report["purpose"] != "sft" or not report["images_validated"]:
        raise ValueError("Training requires SFT preparation with decoded images")
    if train.name != "train_swift.jsonl" or val.name != "val_swift.jsonl":
        raise ValueError("Use train_swift.jsonl and val_swift.jsonl from drivescene prepare")
    for name, digest in report["artifact_sha256"].items():
        if sha256_file(train.parent / name) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    manifests = {name: list(iter_jsonl(train.parent / f"{name}_manifest.jsonl")) for name in ("train", "val")}
    if any(not rows for rows in manifests.values()):
        raise ValueError("Training and validation must both contain labeled examples")
    if {r["scene_id"] for r in manifests["train"]} & {r["scene_id"] for r in manifests["val"]}:
        raise ValueError("Train/validation scene overlap")
    if not 0 < args.min_image_tokens <= args.max_image_tokens:
        raise ValueError("Invalid image token budget")
    if args.gpu_count < 1 or args.max_length < 1:
        raise ValueError("GPU count and maximum sequence length must be positive")
    for rows in manifests.values():
        for row in rows:
            if not row["images_validated"] or row["answer"] is None or any(not Path(p).is_file() for p in row["images"]):
                raise ValueError("Training requires labeled examples with existing validated images")
    command = ["swift", "sft", "--model", args.model, "--dataset", str(train), "--val_dataset", str(val),
        "--split_dataset_ratio", "0", "--tuner_type", "lora", "--torch_dtype", "bfloat16",
        "--lora_rank", "16", "--lora_alpha", "32", "--target_modules", "all-linear",
        "--freeze_vit", "true", "--freeze_aligner", "true", "--attn_impl", "sdpa",
        "--system", "", "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "8", "--learning_rate", "1e-4", "--num_train_epochs", "2",
        "--max_length", str(args.max_length), "--max_pixels", str(args.max_image_tokens * 32 * 32),
        "--truncation_strategy", "delete", "--gradient_checkpointing", "true",
        "--seed", str(args.seed), "--data_seed", str(args.seed), "--eval_strategy", "steps",
        "--eval_steps", "50", "--save_steps", "50", "--save_total_limit", "2", "--logging_steps", "5",
        "--report_to", "tensorboard", "--output_dir", str(Path(args.output_dir).resolve()), "--add_version", "false"]
    if args.gpu_count > 1:
        command += ["--deepspeed", "zero3"]
    if args.resume_from_checkpoint:
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("A full resume requires trainer_state.json, not just adapter weights")
        command += ["--resume_from_checkpoint", str(checkpoint)]
    env = {"NPROC_PER_NODE": str(args.gpu_count), "IMAGE_MIN_TOKEN_NUM": str(args.min_image_tokens),
        "IMAGE_MAX_TOKEN_NUM": str(args.max_image_tokens), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    return {"command": command, "environment": env, "seed": args.seed, "git_commit": result.stdout.strip(),
        "python": sys.version, "packages": packages, "preparation": report, "code_dirty": bool(status.stdout.strip()),
        "train_sha256": sha256_file(train), "val_sha256": sha256_file(val),
        "model": args.model, "model_inventory_sha256": sha256_file(Path(args.model) / "asset_inventory.json") if (Path(args.model) / "asset_inventory.json").is_file() else None,
        "status": "plan_only", "note": "Short-run RTX 3090 memory checks are documented in docs/VRAM_FIT.md; full-length runs still require monitoring. Swift deletes examples exceeding max_length; inspect preprocessing counts before accepting a run."}


def main(argv=None):
    default_config = Path(__file__).resolve().parents[1] / "configs/lora_qwen3vl8b.json"
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=default_config)
    selected, _ = pre.parse_known_args(argv)
    config = json.loads(selected.config.read_text())
    allowed = {"model", "dataset", "val_dataset", "gpu_count", "min_image_tokens", "max_image_tokens", "max_length", "seed", "output_dir"}
    if set(config) - allowed:
        raise ValueError(f"Unknown training config keys: {set(config) - allowed}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=os.getenv("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--dataset", default=os.getenv("DATASET", "data/processed/train_swift.jsonl"))
    parser.add_argument("--val-dataset", default=os.getenv("VAL_DATASET", "data/processed/val_swift.jsonl"))
    parser.add_argument("--gpu-count", type=int, default=int(os.getenv("GPU_COUNT", "2")))
    parser.add_argument("--min-image-tokens", type=int, default=64)
    parser.add_argument("--max-image-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/qwen3vl8b_lora")
    parser.add_argument("--resume-from-checkpoint")
    parser.set_defaults(**config)
    for environment, key in (("MODEL_ID", "model"), ("DATASET", "dataset"), ("VAL_DATASET", "val_dataset"), ("GPU_COUNT", "gpu_count")):
        if environment in os.environ:
            parser.set_defaults(**{key: int(os.environ[environment]) if key == "gpu_count" else os.environ[environment]})
    args = parser.parse_args(argv)
    plan = build_plan(args)
    print(shlex.join(plan["command"]))
    if not args.execute:
        print("PLAN ONLY: no model loaded or training launched")
        return
    if plan["code_dirty"]:
        raise ValueError("Commit code changes before a reproducible training run")
    if not os.environ.get("TMUX"):
        raise ValueError("Start long training inside tmux as required by the server convention")
    devices = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if len(devices) != args.gpu_count or len(set(devices)) != len(devices):
        raise ValueError("Explicitly select distinct CUDA_VISIBLE_DEVICES matching --gpu-count")
    if not shutil.which("swift"):
        raise ValueError("Install requirements/gpu-cu128.txt in a dedicated Python 3.11 environment")
    if importlib.metadata.version("ms-swift") != "4.6.0":
        raise ValueError("This recipe targets ms-swift==4.6.0")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError("Output directory is not empty; choose a new run directory or resume explicitly")
    output.mkdir(parents=True, exist_ok=True)
    plan["status"] = "launch_requested"
    plan["environment"]["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    snapshot = output / ("resume_plan.json" if args.resume_from_checkpoint else "run_plan.json")
    snapshot.write_text(json.dumps(plan, indent=2) + "\n")
    (output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    subprocess.run(plan["command"], env={**os.environ, **plan["environment"]}, check=True)


if __name__ == "__main__":
    main()
