#!/usr/bin/env python3
"""Independent Transformers CLIP + PyTorch MLP oracle over the identical GGUF bytes.

No production code is imported. This tests converted-weight fidelity, not publisher task accuracy.
Run on a remote Linux worker, never on the development Mac.
"""
import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

import gguf
import numpy as np
import torch
import transformers
from PIL import Image, ImageOps
from transformers import CLIPVisionConfig, CLIPVisionModel


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mmproj", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Real-model qualification must run on the remote Linux worker")
    if args.threads < 1 or args.warmups < 1 or args.repetitions < 3:
        parser.error("Require positive threads, warmups >= 1 and repetitions >= 3")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    reader = gguf.GGUFReader(str(args.mmproj))
    tensors = {t.name: t for t in reader.tensors}

    def field(name):
        return reader.get_field(name).contents()

    def weight(name):
        tensor = tensors[name]
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True).reshape(tuple(reversed(tensor.shape.tolist()))))

    size = int(field("clip.vision.image_size"))
    hidden = int(field("clip.vision.embedding_length"))
    layers = int(field("clip.vision.block_count"))
    assert layers == 23 and "v.post_ln.weight" not in tensors
    config = CLIPVisionConfig(
        hidden_size=hidden,
        intermediate_size=int(field("clip.vision.feed_forward_length")),
        num_hidden_layers=layers,
        num_attention_heads=int(field("clip.vision.attention.head_count")),
        image_size=size,
        patch_size=int(field("clip.vision.patch_size")),
        hidden_act="gelu" if field("clip.use_gelu") else "quick_gelu",
        layer_norm_eps=float(field("clip.vision.attention.layer_norm_epsilon")),
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = CLIPVisionModel(config).eval()
    state = {
        "vision_model.embeddings.class_embedding": weight("v.class_embd"),
        "vision_model.embeddings.patch_embedding.weight": weight("v.patch_embd.weight"),
        "vision_model.embeddings.position_embedding.weight": weight("v.position_embd.weight"),
        "vision_model.pre_layrnorm.weight": weight("v.pre_ln.weight"),
        "vision_model.pre_layrnorm.bias": weight("v.pre_ln.bias"),
        # CLIP's final LN applies only to pooled CLS, not to last_hidden_state used below.
        "vision_model.post_layernorm.weight": torch.ones(hidden),
        "vision_model.post_layernorm.bias": torch.zeros(hidden),
    }
    for layer in range(layers):
        source = f"v.blk.{layer}."
        target = f"vision_model.encoder.layers.{layer}."
        swapped = tensors[source + "ffn_down.weight"].shape[0] == hidden
        mapping = {
            "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v", "self_attn.out_proj": "attn_out",
            "layer_norm1": "ln1", "layer_norm2": "ln2",
            "mlp.fc1": "ffn_down" if swapped else "ffn_up",
            "mlp.fc2": "ffn_up" if swapped else "ffn_down",
        }
        for destination, origin in mapping.items():
            for suffix in ("weight", "bias"):
                state[target + destination + "." + suffix] = weight(source + origin + "." + suffix)
    model.load_state_dict(state, strict=True)
    del state
    first, first_bias = weight("mm.0.weight"), weight("mm.0.bias")
    second, second_bias = weight("mm.2.weight"), weight("mm.2.bias")
    load_ms = (time.perf_counter() - started) * 1000
    mean = np.array(field("clip.vision.image_mean"), dtype=np.float32)
    std = np.array(field("clip.vision.image_std"), dtype=np.float32)
    started = time.perf_counter()
    source = Image.open(args.image).convert("RGB")
    side = max(source.size)
    square = Image.new("RGB", (side, side), tuple(int(x * 255) for x in mean))
    square.paste(source, ((side - source.width) // 2, (side - source.height) // 2))
    resized = square.resize((size, size), Image.Resampling.BICUBIC)
    pixels = ((np.asarray(resized, dtype=np.float32) / np.float32(255) - mean) / std).transpose(2, 0, 1).copy()
    preprocess_ms = (time.perf_counter() - started) * 1000
    tensor = torch.from_numpy(pixels).unsqueeze(0)

    @torch.inference_mode()
    def encode():
        patches = model(pixel_values=tensor).last_hidden_state[:, 1:, :]
        return torch.nn.functional.linear(torch.nn.functional.gelu(
            torch.nn.functional.linear(patches, first, first_bias), approximate="none"), second, second_bias)

    for _ in range(args.warmups):
        encode()
    timings = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        embeddings = encode()
        timings.append((time.perf_counter() - started) * 1000)
    args.output.mkdir(parents=True, exist_ok=True)
    pixels.astype("<f4").tofile(args.output / "pixels.f32")
    embeddings.numpy().astype("<f4").tofile(args.output / "embeddings.f32")
    report = dict(schema="vision-integration-v1", arm="transformers-pytorch", host=socket.gethostname(),
                  torch=torch.__version__, transformers=transformers.__version__,
                  mmprojSha256=sha(args.mmproj), imageSha256=sha(args.image),
                  threads=args.threads, warmups=args.warmups, repetitions=args.repetitions,
                  loadMs=load_ms, preprocessMs=preprocess_ms, encoderProjectorMs=timings,
                  imageSize=size, tokens=embeddings.shape[1], dimension=embeddings.shape[2])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
