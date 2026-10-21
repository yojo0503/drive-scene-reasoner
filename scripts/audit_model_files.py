#!/usr/bin/env python3
"""Inspect Qwen config/index/safetensors headers without loading model tensors."""
import argparse
import json
from pathlib import Path
import struct


def inspect_model(root):
    inventory = json.loads((root / "asset_inventory.json").read_text())
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Unexpected model architecture")
    files = {row["path"]: row for row in inventory["files"]}
    actual = {}
    total_parameters = 0
    for name in sorted(set(index["weight_map"].values())):
        path = root / name
        if name not in files or path.stat().st_size != files[name]["size"]:
            raise ValueError(f"Missing or incomplete shard: {name}")
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_length < 64 * 1024 * 1024:
                raise ValueError(f"Invalid safetensors header length: {name}")
            tensors = json.loads(stream.read(header_length))
        for key, entry in tensors.items():
            if key == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= path.stat().st_size - 8 - header_length:
                raise ValueError(f"Invalid tensor offsets: {name}/{key}")
            if key in actual:
                raise ValueError(f"Duplicate tensor name: {key}")
            actual[key] = name
            parameters = 1
            for dimension in entry["shape"]:
                parameters *= dimension
            total_parameters += parameters
    if actual != index["weight_map"]:
        raise ValueError("Safetensors headers do not match the weight index")
    required = ["tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json", "generation_config.json"]
    for name in required:
        json.loads((root / name).read_text())
    return {"repo": inventory["repo"], "revision": inventory["revision"], "model_type": config["model_type"],
        "shards": len(set(actual.values())), "tensors": len(actual), "stored_parameters": total_parameters,
        "total_download_bytes": sum(row["size"] for row in files.values()),
        "config_processor_tokenizer_index_checked": True, "safetensors_header_offsets_checked": True,
        "weights_loaded": False, "generation_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
