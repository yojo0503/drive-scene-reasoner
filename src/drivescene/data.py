"""Read DriveLM-nuScenes scene/keyframe/QA data without copying upstream code.

Expected input follows v1.0/v1.1 nested scene_id -> key_frames -> QA format.
No question/answer normalization that changes the label semantics is done here.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
TASKS = ("perception", "prediction", "planning", "behavior")


@dataclass(frozen=True)
class QARecord:
    sample_id: str
    scene_id: str
    frame_id: str
    task: str
    question: str
    answer: str | None
    image_paths: dict[str, str]


def load_drivelm(path: str | Path) -> list[QARecord]:
    """Load DriveLM JSON, accepting question-only evaluation files (answer=None)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        source = json.load(f)
    if not isinstance(source, dict):
        raise ValueError("DriveLM data must be a dictionary keyed by scene id")

    rows: list[QARecord] = []
    for scene_id, scene in source.items():
        for frame_id, frame in scene.get("key_frames", {}).items():
            images = frame.get("image_paths", {})
            if not isinstance(images, dict):
                images = {}
            qa_groups = frame.get("QA", {})
            for task in TASKS:
                for i, qa in enumerate(qa_groups.get(task, [])):
                    question = qa.get("Q", "")
                    if not isinstance(question, str) or not question.strip():
                        continue
                    answer = qa.get("A")
                    answer = str(answer) if answer is not None else None
                    rows.append(
                        QARecord(
                            sample_id=f"{scene_id}:{frame_id}:{task}:{i}",
                            scene_id=scene_id,
                            frame_id=frame_id,
                            task=task,
                            question=question.strip(),
                            answer=answer,
                            image_paths={str(k): str(v) for k, v in images.items()},
                        )
                    )
    return rows


def resolve_image_path(original: str, image_root: str | Path) -> Path:
    """Resolve DriveLM's ../../nuscenes/samples/... or samples/... paths.

    image_root should point to a directory containing samples/ or the samples/ directory.
    If the file is missing, return the expected location (caller chooses to validate).
    """
    image_root = Path(image_root).expanduser().resolve()
    original_path = Path(original)
    if original_path.is_absolute() and original_path.exists():
        return original_path
    raw = original.replace("\\", "/")
    if "/samples/" in f"/{raw}":
        suffix = raw.split("/samples/", 1)[1]
        subpath = Path("samples") / suffix
    elif raw.startswith("samples/"):
        subpath = Path(raw)
    else:
        # Be conservative: no trust in arbitrary nested ../ in dataset annotations.
        subpath = Path(original_path.name)
    candidates = [image_root / subpath]
    if subpath.parts and subpath.parts[0] == "samples":
        candidates.append(image_root / Path(*subpath.parts[1:]))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def scene_split(rows: list[QARecord], val_fraction: float = 0.1, seed: int = 42) -> dict[str, str]:
    """Split by scene, never by question, to avoid frame/question leakage."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be >= 0 and < 1")
    scene_ids = sorted({r.scene_id for r in rows})
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    n_val = min(len(scene_ids) - 1, max(1, round(len(scene_ids) * val_fraction))) if len(scene_ids) > 1 and val_fraction > 0 else 0
    validation = set(scene_ids[:n_val])
    return {s: "val" if s in validation else "train" for s in scene_ids}


def build_example(row: QARecord, image_root: str | Path, cameras: str = "all", validate_images: bool = True) -> dict:
    if cameras not in ("all", "front"):
        raise ValueError("cameras must be 'all' or 'front'")
    wanted = ("CAM_FRONT",) if cameras == "front" else CAMERAS
    selected: list[tuple[str, Path]] = []
    for cam in wanted:
        if cam in row.image_paths:
            resolved = resolve_image_path(row.image_paths[cam], image_root)
            if validate_images and not resolved.is_file():
                raise FileNotFoundError(f"Missing image {cam} for {row.sample_id}: {resolved}")
            selected.append((cam, resolved))
    if not selected:
        raise ValueError(f"No requested camera views for {row.sample_id}")
    image_header = "\n".join(f"{cam}: <image>" for cam, _ in selected)
    prompt = (
        "You are analyzing a driving scene for offline research. "
        "Answer only from visible evidence. If evidence is insufficient, state uncertainty. "
        "Do not claim to control the vehicle.\n"
        f"Task: {row.task}\n{image_header}\nQuestion: {row.question}"
    )
    return {
        "id": row.sample_id,
        "scene_id": row.scene_id,
        "frame_id": row.frame_id,
        "task": row.task,
        "question": row.question,
        "answer": row.answer,
        "camera_names": [cam for cam, _ in selected],
        "images": [str(p) for _, p in selected],
        "prompt": prompt,
    }


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(path: str | Path, records: Iterator[dict] | list[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def to_swift_sft(sample: dict) -> dict:
    if sample["answer"] is None:
        raise ValueError("Question-only examples cannot be used for supervised training")
    return {
        "messages": [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": sample["answer"]},
        ],
        "images": sample["images"],
    }
