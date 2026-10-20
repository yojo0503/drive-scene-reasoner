#!/usr/bin/env bash
set -euo pipefail
# Requires ms-swift; run after data preparation. Set GPU_COUNT=1/2/4 as available.
# This is an UNVERIFIED GPU starting template; tune image tokens, ZeRO and batch for your setup.
GPU_COUNT="${GPU_COUNT:-2}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET="${DATASET:-data/processed/train_swift.jsonl}"
NPROC_PER_NODE="$GPU_COUNT" swift sft \
  --model "$MODEL_ID" \
  --dataset "$DATASET" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --freeze_vit true \
  --freeze_aligner true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 2 \
  --max_length 2048 \
  --gradient_checkpointing true \
  --deepspeed zero3 \
  --output_dir outputs/qwen3vl8b_lora
