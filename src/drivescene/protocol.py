"""Official DriveLM result IDs and a deliberately limited tag-0 metric adapter.

Full scoring also needs upstream graph gating, language metrics and a paid judge.
This module does not call that judge or present its subset as the official score.
"""

import json
from pathlib import Path

from .data import iter_jsonl
from .eval import checked_predictions, unique_rows


def official_records(manifest_path, predictions_path):
    refs = unique_rows(iter_jsonl(manifest_path), "manifest")
    preds = checked_predictions(refs, predictions_path)
    missing = set(refs) - set(preds)
    if missing:
        raise ValueError(f"Missing predictions for official export: {len(missing)}")
    output = []
    seen = set()
    for key, row in refs.items():
        official_id = row.get("official_id")
        if not official_id or official_id in seen:
            raise ValueError(f"Missing or duplicate official_id: {official_id}")
        seen.add(official_id)
        output.append({"id": official_id, "answer": preds[key]})
    return output


def export_official(manifest_path, predictions_path, output_path):
    rows = official_records(manifest_path, predictions_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(rows)


def evaluate_tag0(manifest_path, predictions_path):
    refs = unique_rows(iter_jsonl(manifest_path), "manifest")
    preds = checked_predictions(refs, predictions_path)
    subset = {key: row for key, row in refs.items() if 0 in (row.get("qa_metadata", {}).get("tag") or []) and row.get("answer") is not None}
    if not subset:
        raise ValueError("No labeled tag-0 examples; do not invent official tags for untagged training data")
    if set(subset) - set(preds):
        raise ValueError("Missing tag-0 predictions")
    correct = sum(preds[key] == row["answer"] for key, row in subset.items())
    return {"metric": "drivelm_tag0_exact_accuracy_without_graph_gating", "count": len(subset),
        "correct": correct, "accuracy": correct / len(subset),
        "note": "Uses upstream tag-0 exact string comparison only. Not the combined official score; graph eligibility filtering is not applied."}
