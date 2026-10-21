"""Transparent local QA metrics; these are NOT official DriveLM challenge scores."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .data import iter_jsonl


def normalize_answer(s: str) -> str:
    text = str(s).strip().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)).strip()


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def unique_rows(rows, label="records") -> dict:
    result = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"Missing/non-string id in {label}")
        if key in result:
            raise ValueError(f"Duplicate id in {label}: {key}")
        result[key] = row
    return result


def checked_predictions(refs, predictions_path):
    rows = unique_rows(iter_jsonl(predictions_path), "predictions")
    unknown = set(rows) - set(refs)
    if unknown:
        raise ValueError(f"Unknown prediction IDs: {sorted(unknown)[:5]}")
    for key, row in rows.items():
        if not isinstance(row.get("prediction"), str):
            raise ValueError(f"Prediction must be a string: {key}")
    return {key: row["prediction"] for key, row in rows.items()}


def evaluate(manifest_path: str | Path, predictions_path: str | Path, missing_policy: str = "error") -> dict:
    if missing_policy not in ("error", "wrong"):
        raise ValueError("missing_policy must be error or wrong")
    refs = unique_rows(iter_jsonl(manifest_path), "manifest")
    preds = checked_predictions(refs, predictions_path)
    labeled = {key: row for key, row in refs.items() if row.get("answer") is not None}
    if not labeled:
        raise ValueError("No labeled references to evaluate")
    missing = set(labeled) - set(preds)
    if missing and missing_policy == "error":
        raise ValueError(f"Missing predictions for {len(missing)} labeled examples; use --missing-policy wrong to count them as errors")
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample_id, row in labeled.items():
        answer = str(row["answer"])
        pred = preds.get(sample_id, "")
        pair = (0.0, 0.0) if sample_id in missing else (float(normalize_answer(pred) == normalize_answer(answer)), token_f1(pred, answer))
        groups[row["task"]].append(pair)

    def score(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"count": 0, "exact_match": None, "token_f1": None}
        return {"count": len(pairs), "exact_match": round(sum(x for x, _ in pairs) / len(pairs), 4),
                "token_f1": round(sum(y for _, y in pairs) / len(pairs), 4)}

    all_pairs = [p for group in groups.values() for p in group]
    return {
        "metrics_note": "Local normalized exact match / token F1, not official DriveLM leaderboard metrics",
        "references": len(refs), "predictions": len(preds),
        "labeled_references": len(labeled), "missing_predictions": len(missing),
        "coverage": round((len(labeled) - len(missing)) / len(labeled), 6),
        "missing_policy": missing_policy,
        "evaluated": len(all_pairs), "overall": score(all_pairs),
        "by_task": {task: score(pairs) for task, pairs in sorted(groups.items())},
    }
