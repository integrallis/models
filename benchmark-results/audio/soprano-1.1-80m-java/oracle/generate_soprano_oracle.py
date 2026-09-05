#!/usr/bin/env python3
"""Generate independent Soprano LM and vocoder fixtures from the official PyTorch code."""

import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = "The JVM can speak for itself."
FRAMES = 4
HIDDEN_SIZE = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_array(path: Path, values: np.ndarray, dtype: str) -> None:
    values.astype(dtype, copy=False).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    soprano_package = types.ModuleType("soprano")
    soprano_package.__path__ = [str(args.reference / "soprano")]
    sys.modules["soprano"] = soprano_package
    from soprano.vocos.decoder import SopranoDecoder

    torch.set_num_threads(1)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    framed_prompt = f"[STOP][TEXT]{PROMPT}[START]"
    token_ids = tokenizer(framed_prompt, return_tensors="pt")["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, device_map="cpu"
    ).eval()
    with torch.no_grad():
        output = model(token_ids, output_hidden_states=True, use_cache=False)
    logits = output.logits[0, -1].detach().cpu().numpy().astype("<f4")
    hidden = output.hidden_states[-1][0, -1].detach().cpu().numpy().astype("<f4")

    features = np.empty((FRAMES, HIDDEN_SIZE), dtype=np.float32)
    for frame in range(FRAMES):
        for channel in range(HIDDEN_SIZE):
            features[frame, channel] = math.sin((frame + 1) * (channel + 1) * 0.001) * 0.05
    decoder = SopranoDecoder().eval()
    decoder.load_state_dict(torch.load(args.model / "decoder.pth", map_location="cpu", weights_only=True))
    with torch.no_grad():
        pcm = decoder(torch.from_numpy(features).transpose(0, 1).unsqueeze(0))[0]
    pcm = pcm.detach().cpu().numpy().astype("<f4")

    files = {
        "prompt-tokens.i32le": token_ids[0].detach().cpu().numpy().astype("<i4"),
        "prompt-logits.f32le": logits,
        "prompt-hidden.f32le": hidden,
        "vocoder-features.f32le": features,
        "vocoder-pcm.f32le": pcm,
    }
    for name, values in files.items():
        write_array(args.output / name, values, values.dtype.str)

    metadata = {
        "oracle": "official Soprano PyTorch/Transformers implementation",
        "referenceCommit": "12fac06eb8fa53bad8b3941d3cb11e9c869477c4",
        "modelRepository": "ekwek/Soprano-1.1-80M",
        "modelSha256": sha256(args.model / "model.safetensors"),
        "decoderSha256": sha256(args.model / "decoder.pth"),
        "prompt": PROMPT,
        "framedPrompt": framed_prompt,
        "promptTokens": token_ids[0].tolist(),
        "logitsLength": int(logits.size),
        "hiddenLength": int(hidden.size),
        "vocoderFrames": FRAMES,
        "vocoderHiddenSize": HIDDEN_SIZE,
        "pcmLength": int(pcm.size),
        "files": {},
    }
    for name in files:
        path = args.output / name
        metadata["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "oracle.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
