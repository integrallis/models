#!/usr/bin/env python3
"""Persistent independent encoder arm; never imports production code."""
import json
import platform
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision-qualification"))
from oracle import load_encoder, prepare_encoder, sha, torch, transformers


def main():
    if platform.system() != "Linux":
        raise SystemExit("Use the remote Linux worker")
    mmproj, corpus, output = map(Path, sys.argv[1:])
    output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    state = load_encoder(mmproj)
    configuration = dict(host=socket.gethostname(), threads=8, mmprojSha256=sha(mmproj),
                         corpusSha256=sha(corpus), loadMs=state[8], torch=torch.__version__,
                         transformers=transformers.__version__)
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2))
    for case in json.loads(corpus.read_text())["cases"]:
        destination = output / case["id"]
        destination.mkdir()
        image = corpus.parent / case["image"]
        if sha(image) != case["imageSha256"]:
            raise ValueError("Image changed")
        runs = []
        for run in range(-case["warmups"], case["repetitions"]):
            started = time.perf_counter()
            _, _, encode = prepare_encoder(state, image)
            embeddings = encode()
            duration = (time.perf_counter() - started) * 1000
            if not torch.isfinite(embeddings).all():
                raise ValueError("Nonfinite embeddings")
            runs.append(dict(run=run, encoderMs=duration))
        embeddings.numpy().astype("<f4").tofile(destination / "embeddings.f32")
        (destination / "runs.json").write_text(json.dumps(runs, indent=2))
        print(case["id"], flush=True)


if __name__ == "__main__":
    main()
