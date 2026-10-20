"""Local research API. Reads only prepared dataset samples, not arbitrary file paths."""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import iter_jsonl

app = FastAPI(title="Drive Scene Reasoner", version="0.1.0")


class ReasonRequest(BaseModel):
    sample_id: str
    max_new_tokens: int = Field(default=128, ge=8, le=512)


@lru_cache(maxsize=1)
def _rows():
    manifest = Path(os.environ.get("DRS_MANIFEST", "data/processed/val_manifest.jsonl"))
    if not manifest.exists():
        return {}
    return {row["id"]: row for row in iter_jsonl(manifest)}


@lru_cache(maxsize=1)
def _model():
    from .infer import QwenVLReasoner
    return QwenVLReasoner(os.environ.get("DRS_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct"))


@app.get("/health")
def health():
    return {"status": "ok", "samples": len(_rows()), "model_loaded": _model.cache_info().currsize > 0}


@app.get("/v1/samples")
def list_samples(limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(422, "limit must be 1..100")
    return [{"id": x["id"], "scene_id": x["scene_id"], "task": x["task"], "question": x["question"]}
            for x in list(_rows().values())[:limit]]


@app.post("/v1/reason")
def reason(req: ReasonRequest):
    sample = _rows().get(req.sample_id)
    if sample is None:
        raise HTTPException(404, "sample_id not in prepared local manifest")
    if os.environ.get("DRS_ENABLE_MODEL") != "1":
        raise HTTPException(503, "Set DRS_ENABLE_MODEL=1 to allow expensive GPU inference")
    prediction = _model().generate(sample, max_new_tokens=req.max_new_tokens)
    return {"id": req.sample_id, "prediction": prediction, "task": sample["task"]}
