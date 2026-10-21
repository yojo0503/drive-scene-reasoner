import json
from dataclasses import replace
from argparse import Namespace

import pytest
from drivescene.data import CAMERAS, build_example, dump_jsonl, iter_jsonl, load_drivelm, to_swift_sft
from drivescene.eval import evaluate
from drivescene.prepare import prepare_dataset
from drivescene.prompts import build_prompt, multimodal_content
from drivescene.protocol import evaluate_tag0, export_official


def arguments(source, root, output, **overrides):
    values = dict(input=source, image_root=root, output=output, purpose="sft", cameras="all",
        task="all", val_fraction=0.34, seed=42, limit_per_split=0, allow_missing_images=False)
    values.update(overrides)
    return Namespace(**values)


def test_all_views_are_required_and_decodable(dataset):
    source, root, _ = dataset
    row = load_drivelm(source)[0]
    partial = replace(row, image_paths={"CAM_FRONT": row.image_paths["CAM_FRONT"]})
    with pytest.raises(ValueError, match="Missing requested cameras"):
        build_example(partial, root)
    sample = build_example(row, root)
    assert sample["camera_names"] == list(CAMERAS)
    assert sample["image_sizes"]["CAM_FRONT"] == [64, 32]
    (root / "samples/CAM_BACK/0.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Unreadable image"):
        build_example(row, root)


def test_metadata_ids_and_no_annotation_leakage(dataset):
    source, root, _ = dataset
    rows = load_drivelm(source)
    assert rows[0].official_id == "scene0_frame0_0"
    assert rows[1].official_id == "scene0_frame0_1"
    assert rows[1].sample_id == "scene0:frame0:prediction:0"
    sample = build_example(rows[0], root)
    assert sample["qa_metadata"]["future_field"] == 9
    assert sample["qa_metadata"]["con_down"] == [1]
    assert sample["frame_metadata"]["key_object_infos"]
    swift = to_swift_sft(sample)
    assert set(swift) == {"messages", "images"}
    assert "ANNOTATION_SECRET" not in swift["messages"][0]["content"]
    assert "2d_bbox" not in swift["messages"][0]["content"]


def test_sft_provenance_and_graph_sidecar(dataset, tmp_path):
    source, root, _ = dataset
    out = tmp_path / "processed"
    report = prepare_dataset(arguments(source, root, out))
    assert report["counts"] == {"train": 4, "val": 2}
    train = list(iter_jsonl(out / "train_manifest.jsonl"))
    val = list(iter_jsonl(out / "val_manifest.jsonl"))
    assert {r["scene_id"] for r in train}.isdisjoint({r["scene_id"] for r in val})
    frames = list(iter_jsonl(out / "frame_graphs.jsonl"))
    assert len(frames) == 3
    assert frames[0]["nodes"][0]["source_qa"]["con_down"] == [1]
    assert (out / "preparation.json").is_file()


def test_inference_retains_question_only_without_sft(dataset, tmp_path):
    source, root, raw = dataset
    for scene in raw.values():
        for frame in scene["key_frames"].values():
            for group in frame["QA"].values():
                for qa in group:
                    qa.pop("A")
    source.write_text(json.dumps(raw))
    out = tmp_path / "infer"
    report = prepare_dataset(arguments(source, root, out, purpose="infer"))
    assert report["counts"] == {"infer": 6}
    assert all(row["answer"] is None for row in iter_jsonl(out / "infer_manifest.jsonl"))
    assert not list(out.glob("*_swift.jsonl"))
    with pytest.raises(ValueError, match="No examples retained"):
        prepare_dataset(arguments(source, root, tmp_path / "sft"))


def test_empty_answer_is_unlabeled_but_raw_metadata_is_preserved(dataset):
    source, _, raw = dataset
    raw["scene0"]["key_frames"]["frame0"]["QA"]["perception"][0]["A"] = "  "
    source.write_text(json.dumps(raw))
    row = load_drivelm(source)[0]
    assert row.answer is None
    assert row.qa_metadata["A"] == "  "


def test_blank_official_question_keeps_its_id_and_is_excluded_from_sft(dataset, tmp_path):
    source, root, raw = dataset
    raw["scene0"]["key_frames"]["frame0"]["QA"]["perception"][0]["Q"] = ""
    source.write_text(json.dumps(raw))
    out = tmp_path / "infer"
    report = prepare_dataset(arguments(source, root, out, purpose="infer"))
    rows = list(iter_jsonl(out / "infer_manifest.jsonl"))
    assert len(rows) == 6
    assert rows[0]["question_valid"] is False
    assert rows[1]["official_id"] == "scene0_frame0_1"
    assert report["invalid_source_questions"] == 1
    sft = prepare_dataset(arguments(source, root, tmp_path / "sft"))
    assert sum(sft["counts"].values()) == 5


def test_missing_predictions_are_rejected_or_counted_wrong(tmp_path):
    ref, pred = tmp_path / "r.jsonl", tmp_path / "p.jsonl"
    dump_jsonl(ref, [{"id": "1", "answer": "yes", "task": "perception"}, {"id": "2", "answer": "no", "task": "perception"}])
    dump_jsonl(pred, [{"id": "1", "prediction": "yes"}])
    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate(ref, pred)
    result = evaluate(ref, pred, "wrong")
    assert result["coverage"] == 0.5
    assert result["overall"]["exact_match"] == 0.5
    assert result["evaluated"] == 2
    dump_jsonl(pred, [{"id": "1", "prediction": "yes"}] * 2)
    with pytest.raises(ValueError, match="Duplicate id"):
        evaluate(ref, pred)
    dump_jsonl(pred, [{"id": "unknown", "prediction": "yes"}])
    with pytest.raises(ValueError, match="Unknown prediction"):
        evaluate(ref, pred)


def test_official_export_and_exact_tag0(dataset, tmp_path):
    source, root, _ = dataset
    samples = [build_example(r, root) for r in load_drivelm(source)]
    manifest, pred, out = tmp_path / "m.jsonl", tmp_path / "p.jsonl", tmp_path / "official.json"
    dump_jsonl(manifest, samples)
    dump_jsonl(pred, [{"id": r["id"], "prediction": r["answer"]} for r in samples])
    assert export_official(manifest, pred, out) == 6
    assert json.loads(out.read_text())[1] == {"id": "scene0_frame0_1", "answer": "No."}
    assert evaluate_tag0(manifest, pred)["accuracy"] == 1
    dump_jsonl(pred, [{"id": r["id"], "prediction": r["answer"].lower()} for r in samples])
    assert evaluate_tag0(manifest, pred)["accuracy"] == 0


def test_sft_and_inference_share_exact_user_text():
    sample = {"task": "perception", "question": "Is <image> visible?", "camera_names": list(CAMERAS)}
    content = multimodal_content(sample, [object()] * 6)
    reconstructed = "".join(x["text"] if x["type"] == "text" else "<image>" for x in content)
    assert reconstructed == build_prompt(sample["task"], sample["question"], sample["camera_names"])


def test_filtered_graphs_preserve_unselected_dependencies_and_ids(dataset, tmp_path):
    source, root, _ = dataset
    out = tmp_path / "filtered"
    prepare_dataset(arguments(source, root, out, task="prediction", limit_per_split=1))
    for name in ("train", "val"):
        row = next(iter_jsonl(out / f"{name}_manifest.jsonl"))
        assert row["official_id"].endswith("_1")
    for frame in iter_jsonl(out / "frame_graphs.jsonl"):
        assert len(frame["nodes"]) == 2
        assert frame["nodes"][0]["included_in_manifest"] is False
        assert frame["nodes"][1]["included_in_manifest"] is True


def test_inference_refuses_stale_training_artifacts_without_provenance(dataset, tmp_path):
    source, root, _ = dataset
    out = tmp_path / "old-preparation"
    out.mkdir()
    (out / "train_swift.jsonl").write_text("old version training data\n")
    with pytest.raises(ValueError, match="different purpose"):
        prepare_dataset(arguments(source, root, out, purpose="infer"))
    assert not (out / "infer_manifest.jsonl").exists()
