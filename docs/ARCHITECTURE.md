# Architecture design (v0.1 and planned v1.x)

## Boundary

This is a **scene understanding / reasoning research service**, not a planning controller. No direct acceleration, steering, or braking outputs. Planning questions refer to textual analysis of driving situations; they are not safety-certified.

## Data path

DriveLM v1.0/v1.1 nested JSON -> QARecord -> image paths -> examples -> scene-level splits -> ms-swift `messages` + `images` format -> LoRA SFT.

- Keep source `scene_id` and `frame_id` stable as evaluation keys.
- Source `image_paths` are from synchronized camera views in the dataset.
- `answer=None` is possible for challenge validation and must be excluded from SFT.
- Choose `front` or `all` views per experiment; do not interpret front-only as six-view reproduction.
- Do not serialize upstream QA or image bytes into this GitHub repository.

## Model path

`QwenVLReasoner`: multi-image input with labeled cameras, task/question -> text response.

Planned `ReasoningEngine` contract:

```
infer(scene_id, frame_id, question, camera_views, options) -> {
  answer: string,
  task: perception|prediction|planning|behavior,
  evidence: [{camera_id, object_id?, bbox?}],
  uncertainty: optional reason,
  model_version: string,
}
```

**Evidence is future functionality.** There are object tags and boxes in some labeled DriveLM records, but answering grounded boxes requires separate supervised conversion and metrics. Do not present unsupported generated bounding boxes as verified evidence.

## Roadmap of research hypotheses

| Hypothesis | Comparison | Validation |
| --- | --- | --- |
| Domain SFT helps road-scene QA | base VLM vs LoRA | held-out scenes, per-task metrics |
| Six views improve scene QA | front-only vs all-six | task breakdown and VRAM |
| Adaptive camera selection saves tokens | fixed six vs view selector | latency, QA scores, selected-view accuracy |
| Structured question dependencies improve consistency | independent QA vs stage-conditioned QA | logical contradiction rate |
| Evidence grounding reduces hallucination | raw answer vs grounded answer | object-ID/bbox consistency on annotated subset |

## Evaluation protocol

- Avoid splitting by QA or adjacent frames; group by source scene at minimum.
- Separate train/val/test by scenes and document dataset versions.
- For multiple choice, normalized exact match is meaningful; open-ended QA needs task-specific matching and human auditing.
- `drivescene eval` is a *local transparent proxy*, not competition scoring.
- Track p50/p95 inference latency, images per sample, visual tokens (when available), peak VRAM, and hallucination/error categories.

## Production ambitions (not yet built)

- Offline batch job queue, dataset browser, model registry, GPU inference service, benchmark dashboard.
- Rate limiting, authentication, task audit trails; redaction/retention policy for future real driving data.
- Multi-GPU model-serving benchmark for high-resolution six-view input.
