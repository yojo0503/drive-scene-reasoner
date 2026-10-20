# Drive Scene Reasoner

[中文使用说明](README.zh-CN.md)

**DriveLM-compatible driving scene understanding and reasoning, starting with multi-camera visual question answering.**

> Stage: **v0.1 scaffold**. Data conversion and CPU smoke tests run locally. Qwen3-VL inference, LoRA training and GPU memory usage have **not** yet been validated on your RTX 3090 setup.
>
> For **offline research only**: no vehicle control, no safety-critical deployment.

[Original DriveLM](https://github.com/OpenDriveLab/DriveLM) · [DriveLM dataset](https://huggingface.co/datasets/OpenDriveLab/DriveLM) · [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) · [ms-swift](https://github.com/modelscope/ms-swift)

## Goals

- **Perception** — understand surrounding vehicles, pedestrians, signals and roadway context.
- **Prediction** — analyze plausible future states of road users.
- **Planning QA** — explain safe *high-level* behavior in language; **never** generate real vehicle commands.
- **Behavior** — reason about the ego vehicle's described motion.
- **Next** — compare front-only vs six-view VLM input, group-structured reasoning, and visual evidence grounding.

## Project layout

```text
src/drivescene/
  data.py          # DriveLM v1.0/v1.1 reader, image-path resolving, scene-level split
  cli.py           # prepare / infer / eval
  infer.py         # optional Qwen3-VL-8B inference backend
  eval.py          # transparent local metrics (not official competition metrics)
  api.py           # FastAPI, local sample-IDs, optional GPU inference
scripts/
  train_lora.sh    # ms-swift multi-GPU LoRA starting configuration, unverified
  make_toy_data.py # generate self-owned synthetic images for smoke testing
tests/              # CPU unit tests
configs/            # planned task/model configs
docs/               # architecture, reproducibility, upstream notices
```

## 1. Quick start without a GPU

Use Python 3.10+ (3.11 recommended). **The included toy data is synthetic; it is not DriveLM data or a meaningful driving benchmark.**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install pillow
python scripts/make_toy_data.py --output data/toy

drivescene prepare \
  --input data/toy/drivelm_train.json \
  --image-root data/toy/nuscenes \
  --output data/processed \
  --cameras front

pytest -q
```

This creates `train_manifest.jsonl` and `train_swift.jsonl` plus scene-level held-out `val` equivalents. Manifests retain IDs, task category, image paths and ground truth for **research only**. Swift JSONL only stores input/output training pairs. Splits are grouped by scene ID to avoid question-level leakage.

## 2. Prepare real DriveLM data

Follow the official dataset's access terms and download instructions: https://github.com/OpenDriveLab/DriveLM/tree/main/challenge

Place files locally (do not commit them):

```text
data/
  drivelm/v1_1_train_nus.json
  nuscenes/samples/CAM_FRONT/*.jpg
  nuscenes/samples/CAM_FRONT_LEFT/*.jpg
  ... other cameras ...
```

```bash
drivescene prepare \
  --input data/drivelm/v1_1_train_nus.json \
  --image-root data/nuscenes \
  --output data/processed \
  --cameras all \
  --val-fraction 0.1 \
  --seed 42
```

Start with `--cameras front --limit-per-split 100` when debugging memory. **Front-only inputs lose essential evidence for rear/side questions; this is an ablation, not an equivalent reproduction.**

`v1_1_val_nus_q_only.json` provides questions without ground-truth answers; don't mistake it for a locally scoreable validation split. Hold out scenes from the labeled training set for local metrics, and use the official test protocol separately if available.

## 3. Run Qwen3-VL inference (requires NVIDIA GPU and model access)

Install compatible PyTorch, CUDA, Transformers and Accelerate versions. An example:

```bash
pip install -e '.[vision]'

drivescene infer \
  --manifest data/processed/val_manifest.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output outputs/val_predictions.jsonl \
  --limit 5

drivescene eval \
  --manifest data/processed/val_manifest.jsonl \
  --predictions outputs/val_predictions.jsonl
```

`infer` uses Hugging Face's Qwen3-VL processor and multimodal chat template. **This backend has not been run against the full 8B checkpoint in this scaffold**; check your CUDA/Transformers version and image token budget. If running on 24GB, consider 4-bit quantization or multi-GPU sharding after establishing a small-model baseline.

The local evaluator reports normalized Exact Match and token F1 by task. **These are not the official DriveLM leaderboard's metrics**, which include additional evaluators.

## 4. LoRA fine-tuning (multi-GPU RTX 3090 starting point)

```bash
pip install -U ms-swift transformers deepspeed
GPU_COUNT=2 MODEL_ID=Qwen/Qwen3-VL-8B-Instruct bash scripts/train_lora.sh
```

`train_lora.sh` is a **starting template**, not a proven memory-fitting recipe. The model is large, and six camera images can exceed the visual token budget quickly. Begin with front-only, short sequence length, `batch_size=1`, and adapt the current version's memory/quantization flags. Compare 2 vs 4 GPUs and verify with `nvidia-smi` before increasing the number of views. A single 24GB 3090 is especially constrained.

## 5. Local API

```bash
pip install -e '.[api,vision]'
DRS_MANIFEST=data/processed/val_manifest.jsonl uvicorn drivescene.api:app --host 127.0.0.1 --port 8000
# GET /health and GET /v1/samples work without a model.
# To opt in to heavy GPU inference:
DRS_ENABLE_MODEL=1 DRS_MANIFEST=data/processed/val_manifest.jsonl uvicorn drivescene.api:app --host 127.0.0.1 --port 8000
```

`POST /v1/reason` accepts a **manifest sample ID**; arbitrary path inference is intentionally not exposed. The API is for local development and does not yet provide authentication, concurrency scheduling or model-serving optimizations. Don't expose it publicly without hardening.

## First experiment checklist

- [ ] Download a licensed DriveLM subset; validate 20 six-view samples.
- [ ] Run Qwen3-VL base inference on held-out *scenes*.
- [ ] Report per-task Exact Match / token F1; add official metric adapter separately.
- [ ] Train 8B LoRA on a small labeled subset; compare against frozen base.
- [ ] Compare front-only vs six-view, latency, peak VRAM and error types.
- [ ] Add visual evidence (camera ID + object IDs + coordinates) using grounded labels, not generated guesses.

For research plans and limitations see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Credits / licensing

This project scaffold is independently authored and **does not bundle DriveLM code, weights, images, or QA annotations**. Credit and licensing details are in [docs/UPSTREAM.md](docs/UPSTREAM.md). Upstream DriveLM source is Apache-2.0 unless noted; its language dataset is **CC BY-NC-SA 4.0**; nuScenes images are subject to their own terms. Check all terms before using and redistributing any data. Choose an appropriate license for your new repository before publicly releasing reusable code.
