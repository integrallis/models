#!/usr/bin/env python3
"""Sequential, same-worker two-arm collection. Does not infer admission from process success."""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from compare import selected_variants


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log):
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    if platform.system() != "Linux":
        raise SystemExit("Use the remote Linux worker")
    if len(sys.argv) not in (5, 6):
        raise ValueError("Usage: run.py corpus distribution reference commit [variant]")
    root, distribution, reference = [Path(p).resolve() for p in sys.argv[1:4]]
    provider = "com/integrallis/models/backend/nativekernel/NativeDenseProjectionProvider.class"
    native_runtime = False
    for jar in distribution.glob("lib/*.jar"):
        if jar.name.startswith("._"):
            continue
        with zipfile.ZipFile(jar) as archive:
            native_runtime |= provider in archive.namelist()
    if not native_runtime:
        raise ValueError("Missing native runtime: build with -PmodelsBenchNative=true")
    revision = sys.argv[4]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Full immutable source commit required")
    script = Path(__file__).resolve().parent
    manifest = json.loads((script / "manifest.json").read_text())
    variants = selected_variants(manifest, [sys.argv[5]] if len(sys.argv) == 6 else None)
    corpus = root / "cases.json"
    cases = json.loads(corpus.read_text())["cases"]
    output = root / "results"
    output.mkdir(exist_ok=False)
    native = Path(os.environ["VISION_NATIVE_LIBRARY"]).resolve()
    receipt = dict(schema=2, variants=variants, referenceFlashAttention="auto", host=socket.gethostname(), revision=revision,
                   manifestSha256=digest(script / "manifest.json"), corpusSha256=digest(corpus),
                   nativeSha256=digest(native), threads=manifest["threads"], context=1024,
                   maximumOutputTokens=32, kvType="f32",
                   jars={p.name: digest(p) for p in distribution.glob("lib/*.jar")},
                   artifacts={p.name: digest(p) for p in (root / "artifacts").glob("*.gguf")},
                   oracleBinarySha256=digest(reference / "task-oracle"),
                   oracleCommit=subprocess.check_output(["git", "-C", str(reference / "source"), "rev-parse", "HEAD"], text=True).strip())
    for name, artifact in manifest["files"].items():
        if receipt["artifacts"].get(name) != artifact["sha256"]:
            raise ValueError("Artifact mismatch: " + name)
    if subprocess.check_output(["git", "-C", str(reference / "source"), "status", "--porcelain"], text=True).strip():
        raise ValueError("Reference source is dirty")
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    run(["lscpu"], output / "cpu.txt")
    run([sys.executable, "-m", "pip", "freeze"], output / "environment.txt")
    mmproj = root / "artifacts/llava-v1.5-7b-mmproj-model-f16.gguf"
    run([sys.executable, script / "encode_reference.py", mmproj, corpus, output / "encoder"], output / "encoder.log")
    for variant in variants:
        model = root / "artifacts" / f"llava-v1.5-7b-{variant}.gguf"
        destination = output / variant
        destination.mkdir()
        java = [os.environ["VISION_JAVA"], "--add-modules=jdk.incubator.vector", "--enable-native-access=ALL-UNNAMED",
                "-Xmx12g", "-Dvectors.gguf.threads=8", "-Dmodels.purejava.maxContextLength=1024",
                f"-Dmodels.native.kernels.library={native}", f"-Dmodels.qualification.revision={revision}",
                "-cp", str(distribution / "lib/*"), "com.integrallis.models.bench.VisionTaskQualificationCli",
                model, mmproj, corpus, destination / "candidate"]
        run(["/usr/bin/time", "-v", "-o", destination / "candidate-resources.txt", *java], destination / "candidate.log")
        lines = []
        for case in cases:
            identity = case["id"]
            paths = [output / "encoder" / identity / "embeddings.f32",
                     destination / "candidate" / identity / "prefix.i32",
                     destination / "candidate" / identity / "suffix.i32"]
            if any(" " in str(p) for p in paths):
                raise ValueError("Reference input paths must not contain spaces")
            lines.append("\t".join([identity, *map(str, paths), str(case["warmups"]), str(case["repetitions"])]))
        tsv = destination / "reference-cases.tsv"
        tsv.write_text("\n".join(lines) + "\n")
        # Keep JSON stdout separate from llama.cpp diagnostics.
        with (destination / "reference.jsonl").open("w") as stdout, (destination / "reference.log").open("w") as stderr:
            subprocess.run(["/usr/bin/time", "-v", "-o", str(destination / "reference-resources.txt"),
                            str(reference / "task-oracle"), str(model), str(tsv), "8"],
                           stdout=stdout, stderr=stderr, check=True)
    (output / "COLLECTION_COMPLETE").write_text("Collected both arms; verdict still required.\n")


if __name__ == "__main__":
    main()
