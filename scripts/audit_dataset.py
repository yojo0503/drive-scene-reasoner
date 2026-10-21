#!/usr/bin/env python3
"""Audit complete DriveLM scene/frame/QA/image alignment using only CPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drivescene.data import load_drivelm, build_example
from drivescene.prepare import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = load_drivelm(args.input)
    if not rows:
        raise ValueError("Dataset contains no QA records")
    if len({r.sample_id for r in rows}) != len(rows) or len({r.official_id for r in rows}) != len(rows):
        raise ValueError("Dataset contains duplicate IDs")
    frames = {}
    for row in rows:
        frames.setdefault((row.scene_id, row.frame_id), row)
    images, dimensions = set(), Counter()
    for n, row in enumerate(frames.values(), 1):
        example = build_example(row, args.image_root, "all", True)
        images.update(example["images"])
        dimensions.update("x".join(map(str, size)) for size in example["image_sizes"].values())
        if n % 500 == 0:
            print(f"Decoded all six cameras for {n}/{len(frames)} frames", flush=True)
    report = {"source": str(args.input.resolve()), "source_sha256": sha256_file(args.input),
        "scenes": len({r.scene_id for r in rows}), "frames": len(frames), "qa_records": len(rows),
        "labeled_records": sum(r.answer is not None for r in rows), "tasks": dict(Counter(r.task for r in rows)),
        "invalid_source_questions": sum(not r.question.strip() for r in rows),
        "unique_images": len(images), "frame_camera_dimensions": dict(dimensions),
        "all_requested_images_decoded": True, "model_loaded": False,
        "dependency_fields_nonempty": {key: sum(bool(r.qa_metadata.get(key)) for r in rows) for key in ("con_up", "con_down", "cluster", "layer", "tag")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
