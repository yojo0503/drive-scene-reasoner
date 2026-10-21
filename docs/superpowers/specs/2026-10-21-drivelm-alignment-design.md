# DriveLM compatibility and server assets

The user approved repairing all five findings and downloading the existing Qwen3-VL-8B-Instruct model and DriveLM data. No training, model loading, generation, or paid evaluator calls may run in this task.

Keep the independently authored Qwen backend and existing CLI; align the data contract with upstream DriveLM rather than adopting the legacy LLaMA environment. A complete replacement would discard useful modern components; a minimal patch would leave official IDs and reproducibility unresolved.

1. Data: require every requested camera, decode images, preserve raw QA fields, scene/frame metadata, object boxes, task-local indices and official frame-global IDs. Keep annotation metadata out of model prompts to prevent answer leakage.
2. Preparation: separate labeled scene-holdout SFT preparation from unsplit inference manifests that retain unanswered questions. Record source hash, split membership, camera settings, image checks and counts.
3. Evaluation: reject duplicate/unknown IDs, reject missing labeled predictions by default, support explicit missing-as-wrong scoring, and export the official list-of-id/answer format. Report the exact tag-0 subset separately; retain the full upstream evaluator as an explicit optional external workflow, never claim local proxies reproduce its combined score.
4. Graph data: preserve C, con_up, con_down, cluster, layer, tag, key_object_infos and all other original QA/scene/frame fields. Add a frame graph export without inventing dependencies or feeding ground truth to inference.
5. Models/training: add base+LoRA adapter loading, aligned SFT/inference prompts, image token limits, explicit devices, deterministic generation, validation data, environment/config snapshots and checkpoint resume. Pin an installation recipe to verified downloadable distributions; do not claim GPU compatibility before running it.
6. Assets: use pinned repository revisions, resumable downloads, expected size and content hashes, inventory, and safe extraction. Honor gated dataset access; credentials stay in the user's cache and are sent only to the official Hugging Face host. Reference upstream in an ignored third_party checkout, with its commit recorded.

Validation uses CPU-only unit/CLI tests, synthetic fixtures, official public annotations and image decoding, shell syntax checks, and checksums. Long asset jobs run in tmux. Existing server experiments are not touched.
