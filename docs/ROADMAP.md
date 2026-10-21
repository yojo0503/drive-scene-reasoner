# Roadmap

## Completed v0.2 engineering

- Strict requested-camera checks, image decoding, dimensions and repeated-frame validation cache.
- Labeled scene holdouts and separate question-only inference preparation.
- Original QA/scene/frame metadata, graph sidecars and official result IDs.
- Missing/duplicate/unknown prediction checks, coverage and missing-as-wrong mode.
- Official output export and an explicitly limited tag-0 metric adapter.
- Shared prompts, LoRA loader wiring, image budget/device/revision options.
- JSON training recipe, validation set, snapshots and checkpoint-resume wiring.
- Revision-pinned, resumable and hash-checked downloads; safe image archive extraction.
- CPU regression tests and an official public demo data preparation check.

## Next, only after separate user authorization

- Resolve/install the pinned GPU environment and verify model loading.
- Run a held-out-scene base-model benchmark and measure memory/latency.
- Run a small LoRA training experiment and reload its checkpoint for the same benchmark.
- Compare front-only and six-camera performance on suitable question subsets.
- Implement stage-conditioned Graph VQA and grounded object metrics.
- Integrate complete upstream scoring with a controlled judge configuration.

No training or inference is part of the v0.2 setup task.
