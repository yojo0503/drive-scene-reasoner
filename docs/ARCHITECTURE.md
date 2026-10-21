# Architecture v0.2

DriveLM nested scene/key_frames/QA → QARecord with raw metadata and dual IDs → strict camera/image validation → scene holdout SFT or unsplit inference manifests → shared multimodal prompt → Qwen3-VL with optional PEFT adapter.

## Contracts

- Internal ID: scene:frame:task:task-local-index.
- Official ID: scene_frame_frame-global-index in perception/prediction/planning/behavior order.
- Filtering never renumbers source IDs. Do not merge different dataset versions under a shared manifest.
- All requested cameras are mandatory. Camera ordering follows the CAMERAS constant.
- Original QA dictionaries, scene metadata and frame metadata (including key_object_infos) are retained without rewriting coordinates or dependency semantics.
- frame_graphs.jsonl contains all original nodes in each retained frame and identifies nodes included in the manifest.
- No ground-truth metadata enters either the Swift user prompt or inference multimodal content.
- Local metrics require labeled references; official result export also works with unanswered questions.
- LoRA is loaded onto the selected base, not treated as complete model weights.
- Training reads an actual JSON configuration, checks preparation hashes, uses an explicit scene-separated validation set and records its command, environment, package versions and hashes.
- Asset downloads and checksum checks use CPU and do not import torch.

## Unimplemented research behavior

There is no stage-conditioned graph execution, learned camera selector or bbox prediction head. The retained raw fields provide source evidence for implementing these later. Full DriveLM combined evaluation remains a separate upstream workflow.
