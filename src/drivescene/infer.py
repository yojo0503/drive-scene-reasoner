"""Optional Qwen3-VL inference backend; not loaded during CPU-only preparation/tests."""

from __future__ import annotations

from pathlib import Path


class QwenVLReasoner:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", *,
                 adapter_path: str | None = None, min_pixels: int = 65536,
                 max_pixels: int = 262144, device_map: str = "auto",
                 revision: str | None = None, local_files_only: bool = False):
        if not 0 < min_pixels <= max_pixels:
            raise ValueError("Require 0 < min_pixels <= max_pixels")
        if adapter_path and not (Path(adapter_path) / "adapter_config.json").is_file():
            raise ValueError(f"LoRA adapter_config.json not found: {adapter_path}")
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the vision dependencies and a compatible Transformers release") from exc
        self._torch = torch
        load_options = {"revision": revision, "local_files_only": local_files_only}
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype="auto", device_map=device_map, attn_implementation="sdpa", **load_options
        ).eval()
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Install the lora optional dependency to load adapters") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False,
                                                   local_files_only=local_files_only).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, **load_options)
        self.processor.image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

    def generate(self, sample: dict, max_new_tokens: int = 128) -> str:
        if not sample["question"].strip():
            raise ValueError("Cannot generate an answer to an empty source question")
        from PIL import Image

        handles = []
        try:
            if len(sample["camera_names"]) != len(sample["images"]) or not sample["images"]:
                raise ValueError("Camera names and image paths must have matching nonzero lengths")
            for cam, image_path in zip(sample["camera_names"], sample["images"]):
                with Image.open(Path(image_path)) as source:
                    im = source.convert("RGB")
                handles.append(im)
            from .prompts import multimodal_content
            content = multimodal_content(sample, handles)
            messages = [{"role": "user", "content": content}]
            batch = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            batch.pop("token_type_ids", None)
            batch = batch.to(self.model.device)
            with self._torch.inference_mode():
                output = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
            generated = output[0, batch["input_ids"].shape[-1]:]
            return self.processor.decode(generated, skip_special_tokens=True).strip()
        finally:
            for im in handles:
                im.close()
