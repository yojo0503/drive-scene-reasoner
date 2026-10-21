import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from drivescene.prepare import prepare_dataset
from test_alignment import arguments

spec = importlib.util.spec_from_file_location("training_plan", Path(__file__).parents[1] / "scripts/training_plan.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def options(out):
    return SimpleNamespace(dataset=str(out / "train_swift.jsonl"), val_dataset=str(out / "val_swift.jsonl"),
        min_image_tokens=64, max_image_tokens=256, gpu_count=2, max_length=4096,
        model="not-loaded", seed=42, output_dir=str(out / "run"), resume_from_checkpoint=None)


def test_plan_only_has_explicit_validation_budget_and_resume_checks(dataset, tmp_path):
    source, root, _ = dataset
    out = tmp_path / "prep"
    prepare_dataset(arguments(source, root, out))
    args = options(out)
    plan = training.build_plan(args)
    assert plan["status"] == "plan_only"
    assert "--val_dataset" in plan["command"]
    assert "--tuner_type" in plan["command"]
    assert plan["environment"]["IMAGE_MAX_TOKEN_NUM"] == "256"
    args.resume_from_checkpoint = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="full resume"):
        training.build_plan(args)


def test_training_rejects_tampered_preparation(dataset, tmp_path):
    source, root, _ = dataset
    out = tmp_path / "prep"
    prepare_dataset(arguments(source, root, out))
    with (out / "train_swift.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        training.build_plan(options(out))
