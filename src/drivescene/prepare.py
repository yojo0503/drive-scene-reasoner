"""Prepare labeled holdouts or complete inference sets and record provenance."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .data import build_example, dump_jsonl, load_drivelm, scene_split, to_swift_sft


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(args):
    if args.limit_per_split < 0:
        raise ValueError("limit_per_split must be nonnegative")
    rows = load_drivelm(args.input)
    if not rows:
        raise ValueError("Input contains no QA records")
    purpose = getattr(args, "purpose", "sft")
    split = scene_split(rows, args.val_fraction, args.seed) if purpose == "sft" else {r.scene_id: "infer" for r in rows}
    groups = {name: [] for name in (["train", "val"] if purpose == "sft" else ["infer"])}
    selected_rows = []
    unlabeled = 0
    invalid_questions = 0
    for row in rows:
        if args.task != "all" and row.task != args.task:
            continue
        if not row.question.strip():
            invalid_questions += 1
            if purpose == "sft":
                continue
        if purpose == "sft" and (row.answer is None or not row.answer.strip()):
            unlabeled += 1
            continue
        name = split[row.scene_id]
        if args.limit_per_split > 0 and len(groups[name]) >= args.limit_per_split:
            continue
        groups[name].append(build_example(row, args.image_root, args.cameras, not args.allow_missing_images))
        selected_rows.append(row)
    if not any(groups.values()):
        raise ValueError("No examples retained. Use --purpose infer for question-only data")
    if purpose == "sft" and (not groups["train"] or (args.val_fraction > 0 and not groups["val"])):
        raise ValueError("Requested SFT holdout is empty; supply more labeled scenes or explicitly use --val-fraction 0")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Prevent old SFT files from surviving an inference-only conversion in the same directory.
    old_report = out / "preparation.json"
    conflicting = list(out.glob("*_swift.jsonl")) if purpose == "infer" else list(out.glob("infer_manifest.jsonl"))
    if conflicting:
        raise ValueError("Output directory contains artifacts for a different purpose; choose a fresh directory")
    if old_report.exists() and json.loads(old_report.read_text())["purpose"] != purpose:
        raise ValueError("Output directory was prepared for a different purpose; choose a fresh directory")
    for name, samples in groups.items():
        dump_jsonl(out / f"{name}_manifest.jsonl", samples)
        if purpose == "sft":
            dump_jsonl(out / f"{name}_swift.jsonl", [to_swift_sft(s) for s in samples])
        print(f"{name}: {len(samples)} examples")
    # Keep each frame's original QAs, including dependency fields, once in a graph sidecar.
    frames = {}
    selected_ids = {r.sample_id for r in selected_rows}
    selected_frames = {(r.scene_id, r.frame_id) for r in selected_rows}
    for row in rows:
        if (row.scene_id, row.frame_id) not in selected_frames:
            continue
        key = (row.scene_id, row.frame_id)
        frame = frames.setdefault(key, {"scene_id": row.scene_id, "frame_id": row.frame_id,
            "split": split[row.scene_id], "scene_metadata": row.scene_metadata,
            "frame_metadata": row.frame_metadata, "image_paths": row.image_paths, "nodes": []})
        frame["nodes"].append({"id": row.sample_id, "official_id": row.official_id, "task": row.task,
            "included_in_manifest": row.sample_id in selected_ids, "source_qa": row.qa_metadata})
    dump_jsonl(out / "frame_graphs.jsonl", list(frames.values()))
    report = {"schema_version": 2, "purpose": purpose, "source": str(Path(args.input).resolve()),
        "source_sha256": sha256_file(args.input), "cameras": args.cameras, "seed": args.seed,
        "val_fraction": args.val_fraction if purpose == "sft" else None,
        "image_root": str(Path(args.image_root).resolve()), "images_validated": not args.allow_missing_images,
        "split_by_scene": split, "source_records": len(rows), "skipped_unlabeled": unlabeled,
        "invalid_source_questions": invalid_questions,
        "invalid_question_policy": "exclude from SFT; retain with question_valid=false for inference and official ID coverage",
        "limit_per_split": args.limit_per_split, "task_filter": args.task,
        "counts": {name: len(samples) for name, samples in groups.items()},
        "tasks": dict(Counter(r.task for r in selected_rows)),
        "artifact_sha256": {p.name: sha256_file(p) for p in [out / "frame_graphs.jsonl"] +
            [out / f"{name}_{suffix}.jsonl" for name in groups for suffix in
                (["manifest", "swift"] if purpose == "sft" else ["manifest"])]},
        "metadata_note": "Annotation metadata is retained for audit/graph supervision, never added to inference prompts."}
    old_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
