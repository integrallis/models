#!/usr/bin/env python3
"""Fetch pinned inputs and select cases before inference. Remote Linux only."""
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(uri, destination, digest, size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".download")
        subprocess.run(["curl", "--fail", "--location", "--retry", "4", "--output", str(temporary), uri], check=True)
        if temporary.stat().st_size != size or sha(temporary) != digest:
            raise ValueError(f"Corrupt download: {uri}")
        temporary.rename(destination)
    if destination.stat().st_size != size or sha(destination) != digest:
        raise ValueError(f"Corrupt cached input: {destination}")


def select(shards, seed, count):
    ordered = []
    for rows in shards:
        unique = {}
        for row in rows:
            unique.setdefault(str(row["question_id"]), row)
        ordered.append(sorted(unique.values(), key=lambda r: hashlib.sha256((seed + str(r["question_id"])).encode()).hexdigest()))
    selected, seen = [], set()
    for index in range(max(map(len, ordered))):
        for rows in ordered:
            if index >= len(rows):
                continue
            row = rows[index]
            identity = str(row["question_id"])
            if identity not in seen:
                selected.append(row)
                seen.add(identity)
            if len(selected) == count:
                return selected
    raise ValueError("Not enough unique cases")


def main():
    if platform.system() != "Linux":
        raise SystemExit("Use the remote Linux worker")
    import pyarrow.parquet as pq
    from PIL import Image
    root = Path(sys.argv[1]).resolve()
    manifest_path = Path(__file__).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for name, file in manifest["files"].items():
        uri = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{name}'
        fetch(uri, root / "artifacts" / name, file["sha256"], file["size"])
    cases = []
    for dataset in manifest["datasets"]:
        name = dataset["repository"].split("/")[-1].lower()
        shards = []
        for file in dataset["files"]:
            uri = f'https://huggingface.co/datasets/{dataset["repository"]}/resolve/{dataset["revision"]}/{file["path"]}'
            path = root / "datasets" / name / Path(file["path"]).name
            fetch(uri, path, file["sha256"], file["size"])
            shards.append(pq.read_table(path).to_pylist())
        selected = select(shards, manifest["caseSelection"]["seed"], manifest["caseSelection"]["casesPerDataset"])
        for index, row in enumerate(selected):
            identity = f'{name}-{row["question_id"]}'
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            path = root / "images" / (identity + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            answers = row["answers"]
            if isinstance(answers, dict):
                answers = answers["answer"]
            answers = [a["answer"] if isinstance(a, dict) else a for a in answers]
            if len(answers) != 10 or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"Expected ten human answers: {identity}")
            case = dict(id=identity, dataset=name, questionId=str(row["question_id"]),
                        question=row["question"], answers=answers, image=str(path.relative_to(root)),
                        imageSha256=sha(path), warmups=1 if index < 2 else 0,
                        repetitions=3 if index < 2 else 1, blank=False)
            cases.append(case)
            if index < manifest["accuracy"]["blankImageCasesPerDataset"]:
                blank = root / "images" / (identity + "-blank.png")
                Image.new("RGB", image.size, (122, 116, 104)).save(blank)
                cases.append(dict(case, id=identity + "-blank", image=str(blank.relative_to(root)),
                                  imageSha256=sha(blank), blank=True, warmups=0, repetitions=1))
        del shards
    target = root / "cases.json"
    if target.exists():
        raise ValueError("Refusing to replace frozen cases")
    target.write_text(json.dumps(dict(manifestSha256=sha(manifest_path), cases=cases), indent=2) + "\n")
    print(f"Prepared {len(cases)} image cases in {target}")


if __name__ == "__main__":
    main()
