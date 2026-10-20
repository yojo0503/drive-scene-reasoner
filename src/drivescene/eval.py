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


def evaluate(manifest_path: str | Path, predictions_path: str | Path) -> dict:
    refs = {row["id"]: row for row in iter_jsonl(manifest_path)}
    preds = {row["id"]: row.get("prediction", "") for row in iter_jsonl(predictions_path)}
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample_id, row in refs.items():
        if sample_id not in preds or row.get("answer") is None:
            continue
        answer = str(row["answer"])
        pred = str(preds[sample_id])
        groups[row["task"]].append((float(normalize_answer(pred) == normalize_answer(answer)), token_f1(pred, answer)))

    def score(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"count": 0, "exact_match": None, "token_f1": None}
        return {"count": len(pairs), "exact_match": round(sum(x for x, _ in pairs) / len(pairs), 4),
                "token_f1": round(sum(y for _, y in pairs) / len(pairs), 4)}

    all_pairs = [p for group in groups.values() for p in group]
    return {
        "metrics_note": "Local normalized exact match / token F1, not official DriveLM leaderboard metrics",
        "references": len(refs), "predictions": len(preds),
        "evaluated": len(all_pairs), "overall": score(all_pairs),
        "by_task": {task: score(pairs) for task, pairs in sorted(groups.items())},
    }
