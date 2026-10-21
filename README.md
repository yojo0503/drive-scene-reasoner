# Drive Scene Reasoner

[中文说明](README.zh-CN.md)

DriveLM-nuScenes data preparation, multi-camera QA interfaces and Qwen3-VL LoRA workflows.
Version 0.2. CPU behavior is tested. Real model loading, generation and training have not been run.

## What follows DriveLM

The parser retains original QAs (including C, con_up, con_down, cluster, layer, tag and unknown fields), scene/frame metadata and key_object_infos. Each record has a stable internal ID and an official ID: scene_frame_index, where the index follows perception + prediction + planning + behavior within the source frame. Filtering does not renumber IDs.

Graph sidecars preserve complete source QAs for retained frames; they do not infer edges or include annotation answers in model inputs. The model backend remains Qwen3-VL rather than the upstream LLaMA-Adapter baseline.

## CPU setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python scripts/make_toy_data.py --output data/toy
drivescene prepare --input data/toy/drivelm_train.json \
  --image-root data/toy/nuscenes --output data/processed --cameras all
```

All requested cameras must exist. By default every image is decoded and its dimensions recorded. Repeated QAs reuse a file-identity-based validation cache. The explicitly named --allow-missing-images option is for format checks only; training plans reject such artifacts.

## Labeled training preparation

```bash
drivescene prepare --input data/drivelm/v1_1_train_nus.json \
  --image-root data/nuscenes --output data/processed \
  --purpose sft --cameras all --val-fraction 0.1 --seed 42
```

Produces train/val_manifest.jsonl, train/val_swift.jsonl, frame_graphs.jsonl and preparation.json. Splits are by scene. Provenance records the input hash, output hashes, retained tasks, selected cameras and scene membership. --limit-per-split limits retained examples before image validation; it is an ordered debugging subset, not a balanced benchmark.

## Question-only inference preparation

```bash
drivescene prepare --input data/drivelm/v1_1_val_nus_q_only.json \
  --image-root data/nuscenes --output data/challenge \
  --purpose infer --cameras all
```

Produces infer_manifest.jsonl without splitting or dropping unanswered questions, and without SFT files. Choose a different output directory from training preparation.

The official v1.1 question-only source contains one blank question. Its original ID is retained with question_valid=false. SFT excludes blank questions; the inference CLI records an empty prediction with status=invalid_source_question instead of fabricating a question or calling generate for that record.

## Evaluation and upstream interoperability

```bash
drivescene eval --manifest data/processed/val_manifest.jsonl --predictions outputs/predictions.jsonl
# Explicitly count missing answers as wrong:
drivescene eval --manifest data/processed/val_manifest.jsonl --predictions outputs/predictions.jsonl --missing-policy wrong
drivescene export-official --manifest data/challenge/infer_manifest.jsonl \
  --predictions outputs/challenge_predictions.jsonl --output outputs/official_output.json
```

Duplicate and unknown IDs are errors. Missing labeled predictions are errors by default. Reports include coverage; missing-as-wrong keeps every labeled reference in the denominator. Official export requires every manifest prediction and emits the upstream list of id/answer dictionaries.

eval-tag0 implements the upstream tag-0 exact string comparison on explicitly tagged, labeled records. It does not apply upstream graph eligibility filtering, call a judge, or compute the official combined score. For the full workflow see [compatibility notes](docs/DRIVELM_COMPATIBILITY.md).

## Model and LoRA interfaces

```bash
pip install -e '.[vision,lora]'
# Optional future invocation; it loads and executes a model:
drivescene infer --manifest data/processed/val_manifest.jsonl \
  --model /path/to/Qwen3-VL-8B-Instruct --adapter /path/to/adapter \
  --local-files-only --min-pixels 65536 --max-pixels 262144
```

--adapter is optional. SFT and inference share the same user prompt and camera order. Image budget defaults to 64–256 visual tokens per image (32×32 pixels per token); this is a starting configuration, not a validated quality/memory trade-off.

## Training recipe (not executed in setup)

```bash
# CPU-only plan validation; this does not load a model:
bash scripts/train_lora.sh --dry-run
```

configs/lora_qwen3vl8b.json is consumed by the planner; CLI overrides win, and legacy MODEL_ID/DATASET/VAL_DATASET/GPU_COUNT environment variables override file defaults. The planner checks artifact hashes, image validation and scene separation. It requires an explicit validation set and supports --resume-from-checkpoint.

An actual launch additionally requires --execute, a tmux session, explicitly selected CUDA_VISIBLE_DEVICES and the pinned ms-swift version. Run metadata, package versions, pip freeze, data hashes and model inventory hash are saved before launch. See [GPU recipe](requirements/gpu-cu128.txt). Distribution availability was checked; dependency resolution and GPU memory fit still need validation.

## Downloads

scripts/download_assets.py downloads a pinned repository revision, resumes partial files and checks upstream SHA256 (large files) or Git blob SHA1 (small files), then writes asset_inventory.json. It does not import torch. Tokens are read from HF_TOKEN or the user cache and are sent only to the official Hugging Face host. For public Qwen models, --modelscope-weights uses the official Qwen mirror with the original pinned SHA256.

scripts/extract_dataset.py rejects unsafe ZIP paths/symlinks and verifies archive-entry CRC during extraction and resume. Data and weights stay outside Git. Server paths and verification state are recorded in docs/SERVER_SETUP.md.

## Layout

- src/drivescene/data.py, prepare.py: source data, strict views, preparation and graph sidecars.
- prompts.py, infer.py, api.py: shared prompt, optional Qwen+LoRA and local API.
- eval.py, protocol.py: strict local scores and official result IDs.
- scripts/training_plan.py: validation, configuration snapshots and explicit launch.
- scripts/download_assets.py, extract_dataset.py: verified assets.
- tests/: CPU regressions with synthetic and stubbed model contracts.

## Attribution

No upstream code, annotations, images or weights are bundled in this repository. Upstream references are pinned in configs/upstream.json. DriveLM language data is CC BY-NC-SA 4.0; nuScenes has separate terms. See [upstream notices](docs/UPSTREAM.md). This repository's original code has no project license selected yet.
