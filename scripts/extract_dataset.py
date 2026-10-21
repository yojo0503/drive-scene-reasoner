#!/usr/bin/env python3
"""Safely extract verified DriveLM image archives; no model dependencies."""
import argparse
import json
import stat
from pathlib import Path, PurePosixPath
import zipfile
import zlib


def safe_target(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe ZIP entry: {name}")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"ZIP entry escapes output: {name}")
    return target


def crc32_file(path):
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xffffffff


def extract(archive, output):
    output.mkdir(parents=True, exist_ok=True)
    files = 0
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            target = safe_target(output, entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Symlink in ZIP: {entry.filename}")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (target.is_file() and target.stat().st_size == entry.file_size and crc32_file(target) == entry.CRC):
                temporary = target.with_name(target.name + ".extracting")
                with package.open(entry) as source, temporary.open("wb") as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
                if temporary.stat().st_size != entry.file_size:
                    raise ValueError(f"Incomplete extraction: {entry.filename}")
                temporary.replace(target)
            files += 1
            if files % 5000 == 0:
                print(f"{archive.name}: {files} files", flush=True)
    return {"archive": str(archive.resolve()), "output": str(output.resolve()), "files": files, "zip_crc_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [extract(path, args.output) for path in args.archives]
    (args.output / "extraction_inventory.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
