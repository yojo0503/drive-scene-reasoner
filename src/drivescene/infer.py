"""Optional Qwen3-VL inference backend; not loaded during CPU-only preparation/tests."""

from __future__ import annotations

from pathlib import Path


class QwenVLReasoner:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the vision dependencies and a compatible Transformers release") from exc
        self._torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, sample: dict, max_new_tokens: int = 128) -> str:
        from PIL import Image

        content = []
        handles = []
        try:
            for cam, image_path in zip(sample["camera_names"], sample["images"]):
                content.append({"type": "text", "text": f"{cam} camera view:"})
                im = Image.open(Path(image_path)).convert("RGB")
                handles.append(im)
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": (
                "Answer this driving-scene question using visible evidence only. "
                "If the view is insufficient, say so. This is offline analysis, not vehicle control. "
                f"Task={sample['task']}. Question={sample['question']}"
            )})
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
