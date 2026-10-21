"""Exercise loader wiring with fake modules, never download or load real weights."""

import sys
from types import SimpleNamespace
import pytest
from drivescene.infer import QwenVLReasoner


def test_base_and_adapter_loader_wiring(monkeypatch, tmp_path):
    calls = []

    class Model:
        def eval(self):
            return self

    def base_load(model_id, **kwargs):
        calls.append(("base", model_id, kwargs))
        return Model()

    def adapter_load(base, path, **kwargs):
        calls.append(("adapter", path, kwargs))
        return base

    processor = SimpleNamespace(image_processor=SimpleNamespace(size={}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=base_load),
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=adapter_load)))
    (tmp_path / "adapter_config.json").write_text("{}")
    QwenVLReasoner("local-base", adapter_path=str(tmp_path), min_pixels=1024,
        max_pixels=4096, device_map="cuda:1", local_files_only=True)
    assert calls[0][2]["device_map"] == "cuda:1"
    assert calls[1][2] == {"is_trainable": False, "local_files_only": True}
    assert processor.image_processor.size == {"shortest_edge": 1024, "longest_edge": 4096}


def test_invalid_budget_is_rejected_before_model_import():
    with pytest.raises(ValueError, match="min_pixels"):
        QwenVLReasoner(min_pixels=20, max_pixels=10)
