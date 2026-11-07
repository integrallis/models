#!/usr/bin/env python3
"""Download only pinned, hash-verified artifacts, on the remote worker."""
import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path

if platform.system() != "Linux":
    raise SystemExit("Real-model artifacts belong on the remote Linux worker")
manifest = json.loads(Path(__file__).with_name("artifacts.json").read_text())
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
for name, info in sorted(manifest["files"].items(), key=lambda item: "mmproj" not in item[0]):
    if "--projector-only" in sys.argv and "mmproj" not in name:
        continue
    path = destination / name
    if not path.exists():
        url = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{name}'
        partial = path.with_suffix(".partial")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != info["size"] or digest != info["sha256"]:
        raise SystemExit(f"Artifact verification failed: {path}")
    print(name, digest, flush=True)
