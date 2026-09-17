#!/usr/bin/env python3
"""Prepares the pinned public datasets for the logit-fusion study.

Downloads exact Hugging Face dataset revisions, verifies every source file against a SHA-256
pinned in this script, and writes the unified JSONL files plus data-manifest.json described in
the study contract. Data preparation only: nothing here is in the measured inference path.

Usage: python3 prepare_fusion_data.py --out-dir DIR [--cache-dir DIR]
Requires: pyarrow (see requirements-data.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

DEV_SEED = 20260917
DEV_SIZE = 500

SOURCES = {
    "gsm8k-test": {
        "repo": "openai/gsm8k",
        "config": "main",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "sourceFile": "main/test-00000-of-00001.parquet",
        "sourceSha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        "output": "gsm8k-test.jsonl",
    },
    "gsm8k-dev": {
        "repo": "openai/gsm8k",
        "config": "main",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "sourceFile": "main/train-00000-of-00001.parquet",
        "sourceSha256": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
        "output": "gsm8k-dev.jsonl",
        "sampleSeed": DEV_SEED,
        "sampleSize": DEV_SIZE,
    },
    "arc-challenge-test": {
        "repo": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "sourceFile": "ARC-Challenge/test-00000-of-00001.parquet",
        "sourceSha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        "output": "arc-challenge-test.jsonl",
    },
    "math500-test": {
        "repo": "HuggingFaceH4/MATH-500",
        "config": "default",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "sourceFile": "test.jsonl",
        "sourceSha256": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        "output": "math500-test.jsonl",
    },
}


# ---------------------------------------------------------------------------------------------
# Pure transforms (unit-tested without network)
# ---------------------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ids_sha256(ids: list[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def gsm8k_final_answer(solution: str) -> str:
    if "####" not in solution:
        raise ValueError("GSM8K solution has no #### marker")
    return solution.rsplit("####", 1)[1].replace(",", "").strip()


def source_ref(spec: dict, row: int) -> dict:
    return {"repo": spec["repo"], "revision": spec["revision"], "file": spec["sourceFile"], "row": row}


def gsm8k_records(rows: list[dict], spec: dict, split: str, indices=None) -> list[dict]:
    selected = range(len(rows)) if indices is None else indices
    return [
        {
            "id": f"gsm8k-{split}-{row:04d}",
            "dataset": "gsm8k",
            "split": split,
            "question": rows[row]["question"],
            "answer": gsm8k_final_answer(rows[row]["answer"]),
            "answerRaw": rows[row]["answer"],
            "choices": None,
            "source": source_ref(spec, row),
        }
        for row in selected
    ]


def dev_indices(total: int, seed: int = DEV_SEED, size: int = DEV_SIZE) -> list[int]:
    return sorted(random.Random(seed).sample(range(total), size))


def arc_records(rows: list[dict], spec: dict) -> list[dict]:
    records = []
    for row, item in enumerate(rows):
        labels = list(item["choices"]["label"])
        texts = list(item["choices"]["text"])
        if len(labels) != len(texts) or item["answerKey"] not in labels:
            raise ValueError(f"malformed ARC row {row}")
        records.append(
            {
                "id": f"arc-challenge-test-{item['id']}",
                "dataset": "arc",
                "split": "test",
                "question": item["question"],
                "answer": item["answerKey"],
                "answerRaw": item["answerKey"],
                "choices": [{"label": label, "text": text} for label, text in zip(labels, texts)],
                "source": source_ref(spec, row),
            }
        )
    return records


def math500_records(rows: list[dict], spec: dict) -> list[dict]:
    return [
        {
            "id": f"math500-test-{row:03d}",
            "dataset": "math500",
            "split": "test",
            "question": item["problem"],
            "answer": item["answer"],
            "answerRaw": item["solution"],
            "choices": None,
            "subject": item.get("subject"),
            "level": item.get("level"),
            "uniqueId": item.get("unique_id"),
            "source": source_ref(spec, row),
        }
        for row, item in enumerate(rows)
    ]


def to_jsonl(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


def manifest_entry(spec: dict, source_sha: str, output_bytes: bytes, records: list[dict]) -> dict:
    entry = {k: v for k, v in spec.items()}
    entry["sourceSha256"] = source_sha
    entry["outputSha256"] = sha256_bytes(output_bytes)
    entry["items"] = len(records)
    entry["idsSha256"] = ids_sha256([r["id"] for r in records])
    return entry


# ---------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------


def fetch(spec: dict, cache_dir: Path) -> bytes:
    cached = cache_dir / spec["repo"].replace("/", "__") / spec["revision"] / spec["sourceFile"]
    if cached.is_file():
        data = cached.read_bytes()
    else:
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['sourceFile']}"
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    actual = sha256_bytes(data)
    if actual != spec["sourceSha256"]:
        raise SystemExit(f"HASH MISMATCH {spec['sourceFile']}: expected {spec['sourceSha256']} got {actual}")
    return data


def parquet_rows(data: bytes) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pylist()


def jsonl_rows(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    cache_dir: Path = args.cache_dir or out_dir / ".cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for key, spec in SOURCES.items():
        data = fetch(spec, cache_dir)
        if key == "gsm8k-test":
            records = gsm8k_records(parquet_rows(data), spec, "test")
        elif key == "gsm8k-dev":
            rows = parquet_rows(data)
            records = gsm8k_records(rows, spec, "train", dev_indices(len(rows)))
        elif key == "arc-challenge-test":
            records = arc_records(parquet_rows(data), spec)
        else:
            records = math500_records(jsonl_rows(data), spec)
        output = to_jsonl(records)
        (out_dir / spec["output"]).write_bytes(output)
        datasets[key] = manifest_entry(spec, sha256_bytes(data), output, records)
        print(f"wrote {spec['output']} items={len(records)} sha256={datasets[key]['outputSha256']}")

    manifest = {"schemaVersion": 1, "generator": "prepare_fusion_data.py", "datasets": datasets}
    (out_dir / "data-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("DONE data")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
