from drivescene.eval import normalize_answer, token_f1, evaluate
from drivescene.data import dump_jsonl


def test_normalized_metrics(tmp_path):
    assert normalize_answer("  Going LEFT! ") == "going left"
    assert token_f1("A red car", "a car") > 0
    manifest = tmp_path / "ref.jsonl"
    pred = tmp_path / "pred.jsonl"
    dump_jsonl(manifest, [{"id": "s1", "task": "perception", "answer": "Yes."}, {"id": "s2", "task": "prediction", "answer": None}])
    dump_jsonl(pred, [{"id": "s1", "prediction": "yes"}, {"id": "s2", "prediction": "anything"}])
    report = evaluate(manifest, pred)
    assert report["evaluated"] == 1
    assert report["overall"]["exact_match"] == 1.0
