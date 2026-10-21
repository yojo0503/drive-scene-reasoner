#!/usr/bin/env python3
"""Download a pinned HF repository without importing torch or loading models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import shutil
from urllib.parse import quote, urlparse


def file_digest(path, algorithm="sha256", git_blob=False):
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranged_download(command, url, partial, total_size, segments):
    """Resume verified-length byte ranges; assemble before the caller checks the full hash."""
    common = []
    i = 0
    while i < len(command):
        if command[i] in ("--continue-at", "--output"):
            i += 2
        else:
            common.append(command[i])
            i += 1
    width = (total_size + segments - 1) // segments
    prefix_size = partial.stat().st_size if partial.is_file() else 0
    chunks = []
    for n, start in enumerate(range(0, total_size, width)):
        length = min(width, total_size - start)
        chunk = partial.with_name(partial.name + f".range{n:03d}")
        if not chunk.exists() and prefix_size > start:
            remaining = min(length, prefix_size - start)
            with partial.open("rb") as source, chunk.open("wb") as dest:
                source.seek(start)
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Incomplete resume prefix")
                    dest.write(block)
                    remaining -= len(block)
        chunks.append((chunk, start, length))

    def fetch(item):
        chunk, start, length = item
        have = chunk.stat().st_size if chunk.is_file() else 0
        if have > length:
            raise ValueError(f"Oversized range file: {chunk.name}")
        if have < length:
            following = chunk.with_name(chunk.name + ".next")
            status = subprocess.check_output(common + ["--range", f"{start + have}-{start + length - 1}",
                "--max-filesize", str(length - have), "--output", str(following), "--write-out", "%{http_code}", url], text=True).strip()
            if status != "206" or following.stat().st_size != length - have:
                raise ValueError(f"Server did not return the exact requested range: {chunk.name}")
            with following.open("rb") as source, chunk.open("ab") as dest:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
            following.unlink()
        print(f"Range ready: {chunk.name} ({length:,} bytes)", flush=True)

    with ThreadPoolExecutor(max_workers=segments) as pool:
        list(pool.map(fetch, chunks))
    assembled = partial.with_name(partial.name + ".assembling")
    with assembled.open("wb") as dest:
        for chunk, _, _ in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, dest, 8 * 1024 * 1024)
    assembled.replace(partial)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--kind", choices=["model", "dataset"], default="model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include", nargs="+", default=["*"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--segments", type=int, default=1, help="Parallel byte ranges per large file")
    parser.add_argument("--direct-cdn", action="store_true", help="Bypass the configured proxy for official public CDN redirects")
    parser.add_argument("--modelscope-weights", action="store_true", help="Download public Qwen weights from its official ModelScope mirror, still verify pinned HF SHA256")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.segments < 1:
        raise ValueError("workers and segments must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output / "asset_inventory.json"
    if args.verify_only:
        inventory = json.loads(inventory_path.read_text())
        for row in inventory["files"]:
            path = args.output / row["path"]
            if path.stat().st_size != row["size"] or file_digest(path) != row["sha256"]:
                raise ValueError(f"Integrity check failed: {path}")
        print(f"Verified {len(inventory['files'])} files; no model was loaded.", flush=True)
        return

    endpoint = args.endpoint.rstrip("/")
    token = os.environ.get("HF_TOKEN")
    if not token and urlparse(endpoint).hostname == "huggingface.co":
        token_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
        if token_path.is_file():
            token = token_path.read_text().strip()
    if token and urlparse(endpoint).hostname != "huggingface.co":
        raise ValueError("Credentials may only be sent to huggingface.co; unset HF_TOKEN for a public mirror")
    auth_config = args.output / ".download-auth.tmp"
    if token:
        fd = os.open(auth_config, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(f'header = "Authorization: Bearer {token}"\n')
    family = "models" if args.kind == "model" else "datasets"
    meta_url = f"{endpoint}/api/{family}/{args.repo}/revision/{quote(args.revision, safe='')}?blobs=true"
    metadata_command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60"]
    if token:
        metadata_command += ["--config", str(auth_config)]
    try:
        meta = json.loads(subprocess.check_output(metadata_command + [meta_url]))
    except BaseException:
        auth_config.unlink(missing_ok=True)
        raise
    revision = meta["sha"]
    if len(args.revision) == 40 and args.revision != revision:
        auth_config.unlink(missing_ok=True)
        raise ValueError("Repository API returned a different revision than requested")
    selected = [s for s in meta["siblings"] if any(fnmatch.fnmatch(s["rfilename"], p) for p in args.include)]
    unmatched = [p for p in args.include if not any(fnmatch.fnmatch(s["rfilename"], p) for s in selected)]
    if not selected or unmatched:
        auth_config.unlink(missing_ok=True)
        raise ValueError("No files matched")
    prefix = "datasets/" if args.kind == "dataset" else ""
    print(f"Pinned {args.repo}@{revision}: {len(selected)} files, {sum(s.get('size', 0) for s in selected):,} bytes", flush=True)
    # curl config is private and avoids putting the bearer token in process arguments.
    def download(row):
        name = row["rfilename"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe repository path")
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_size = row.get("size")
        lfs = row.get("lfs") or {}
        expected_sha = lfs.get("sha256")
        blob_id = row.get("blobId")

        def valid(candidate):
            if not candidate.is_file() or expected_size is None or candidate.stat().st_size != expected_size:
                return False
            if expected_sha:
                return file_digest(candidate) == expected_sha
            return bool(blob_id) and file_digest(candidate, "sha1", True) == blob_id

        if not valid(path):
            partial = path.with_name(path.name + ".partial")
            url = f"{endpoint}/{prefix}{args.repo}/resolve/{revision}/{quote(name, safe='/')}"
            modelscope = args.modelscope_weights and args.kind == "model" and args.repo.startswith("Qwen/") and name.endswith(".safetensors")
            if modelscope:
                if not expected_sha:
                    raise ValueError("Cross-provider downloads require the pinned upstream SHA256")
                url = f"https://modelscope.cn/models/{args.repo}/resolve/master/{quote(name, safe='/')}"
            if partial.is_file() and partial.stat().st_size == expected_size:
                if valid(partial):
                    partial.replace(path)
                else:
                    partial.unlink()
            if not valid(path):
                command = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "6", "--retry-delay", "5", "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-", "--output", str(partial)]
                if modelscope:
                    command += ["--noproxy", ".modelscope.cn,.aliyuncs.com"]
                elif token:
                    command += ["--config", str(auth_config)]
                if args.direct_cdn:
                    command += ["--noproxy", ".hf.co,.cloudfront.net"]
                # curl does not forward Authorization to a different redirect host.
                if args.segments > 1 and expected_sha and expected_size > 64 * 1024 * 1024:
                    ranged_download(command, url, partial, expected_size, args.segments)
                else:
                    subprocess.run(command + [url], check=True)
                if not valid(partial):
                    raise ValueError(f"Size/hash mismatch: {name}; incomplete file retained for diagnosis")
                partial.replace(path)
        for chunk in path.parent.glob(path.name + ".partial.range*"):
            chunk.unlink()
        digest = file_digest(path)
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
        return {"path": name, "size": path.stat().st_size, "sha256": digest, "upstream_sha256": expected_sha, "git_blob_sha1": blob_id,
                "weight_mirror": "https://modelscope.cn/models/" + args.repo if args.modelscope_weights and name.endswith(".safetensors") else None}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            files = list(pool.map(download, selected))
        inventory = {"repo": args.repo, "kind": args.kind, "revision": revision, "canonical_source": f"https://huggingface.co/{prefix}{args.repo}", "download_endpoint": endpoint, "verified_at_unix": time.time(), "files": files, "model_executed": False}
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
        print(f"COMPLETE: {inventory_path}", flush=True)
    finally:
        auth_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
