import json
from pathlib import Path

from drivescene.data import build_example, load_drivelm, resolve_image_path, scene_split, to_swift_sft


def fixture(tmp_path):
    src = {}
    for idx in range(3):
        scene = f"scene{idx}"
        frame = f"frame{idx}"
        src[scene] = {"key_frames": {frame: {
            "image_paths": {"CAM_FRONT": f"../nuscenes/samples/CAM_FRONT/{idx}.jpg"},
            "QA": {"perception": [{"Q": "What is there?", "A": "A car"}],
                   "prediction": [{"Q": "Will it move?", "A": "Unknown"}]},
        }}}
    path = tmp_path / "train.json"
    path.write_text(json.dumps(src), encoding="utf-8")
    images = tmp_path / "nuscenes" / "samples" / "CAM_FRONT"
    images.mkdir(parents=True)
    for idx in range(3):
        (images / f"{idx}.jpg").write_bytes(b"placeholder")
    return path, tmp_path / "nuscenes"


def test_load_and_scene_split(tmp_path):
    source, root = fixture(tmp_path)
    rows = load_drivelm(source)
    assert len(rows) == 6
    split = scene_split(rows, val_fraction=0.34, seed=42)
    assert list(split.values()).count("val") == 1
    assert all(split[r.scene_id] == split[r2.scene_id] for r in rows for r2 in rows if r.scene_id == r2.scene_id)
    assert split == scene_split(rows, val_fraction=0.34, seed=42)
    example = build_example(rows[0], root, cameras="front")
    assert len(example["images"]) == 1
    assert Path(example["images"][0]).is_file()
    assert to_swift_sft(example)["messages"][1]["content"] == "A car"


def test_resolve_two_data_roots(tmp_path):
    _, root = fixture(tmp_path)
    relative = "../nuscenes/samples/CAM_FRONT/1.jpg"
    assert resolve_image_path(relative, root).exists()
    assert resolve_image_path(relative, root / "samples").exists()


def test_question_only_not_trainable(tmp_path):
    source, root = fixture(tmp_path)
    original = json.loads(source.read_text())
    original["scene0"]["key_frames"]["frame0"]["QA"]["perception"][0].pop("A")
    source.write_text(json.dumps(original))
    rows = load_drivelm(source)
    assert any(r.answer is None for r in rows)
