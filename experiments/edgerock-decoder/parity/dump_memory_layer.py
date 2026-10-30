#!/usr/bin/env python3
"""Dump a parity fixture for EdgeRock's memory layer from the reference implementation.

Uses the reference *class* -- IndependentGQAMemory out of modeling_edgerock.py -- at small
dimensions with seeded random weights, rather than the 4B checkpoint. Three reasons.

The arithmetic is what is under test, not the weights. A fixture built from the real checkpoint
conflates our implementation against theirs with Q4_K quantisation against float32, and a
mismatch would not say which.

It has to be small enough to commit and read. Dumping one real layer's parameters as JSON came
to 848 MB.

And it has to be non-trivial. The real layer with 3 tokens never reaches commit_chunk = 256, so
the bank stays empty and the reference output is all zeros -- a fixture that passes whatever the
read does. Here the chunk is small and the token count crosses it several times, so the expected
output depends on admission, eviction and the read together.

    .venv/bin/python dump_memory_layer.py --out fixture.json
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import types


def load_reference(path: pathlib.Path):
    """Import modeling_edgerock.py directly; it is not an installed package.

    Its directory goes on sys.path first: the module does `from configuration_edgerock import
    EdgeRockConfig`, a sibling import that a bare spec_from_file_location cannot resolve.
    """
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("modeling_edgerock", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modeling_edgerock"] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--writer", type=int, default=16)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=7)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    reference = load_reference(args.reference)

    config = types.SimpleNamespace(
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        num_key_value_heads=args.groups,
        head_dim=args.dim,
        rms_norm_eps=1.0e-6,
        initializer_range=0.05,
        fixed_memory_swa_window=args.window,
        backend="torch",
        independent_memory={
            "capacity": args.capacity,
            "max_commit_chunk": args.chunk,
            "admission_rank": args.rank,
            "writer_intermediate_size": args.writer,
        },
    )
    memory = reference.IndependentGQAMemory(config).to(torch.float32).eval()

    # The zero-initialised parameters are the reference's own choice and make an untrained layer
    # the identity. Give them real values, or the fixture would only prove that zero times
    # anything is zero.
    for name in ("write_down_proj", "admission_up"):
        torch.nn.init.normal_(getattr(memory, name).weight, std=0.05)
    torch.nn.init.normal_(memory.read_scale, std=0.5)
    torch.nn.init.normal_(memory.read_gate_weight, std=0.1)
    torch.nn.init.normal_(memory.read_gate_bias, std=0.1)

    hidden = torch.randn(1, args.tokens, args.hidden, dtype=torch.float32)

    # Rope factors in HuggingFace convention, which is what the reference expects.
    #
    # _memory_rope does `width = cos.shape[-1]; rotated = values[..., :width]` and then
    # rotate_half over that slice -- so cos's width IS the rotated span, and rotate_half pairs
    # lane i with lane i + width/2. HF builds cos as cat(freqs, freqs), so cos[i] == cos[i + W/2]
    # and there are only W/2 distinct frequencies.
    #
    # The first version of this script emitted W/2 entries as cos. The reference then rotated only
    # W/2 lanes, pairing (i, i + W/4), with W/2 distinct frequencies -- a convention the real model
    # never uses. Our rope takes the distinct frequencies and rotates 2x that span, which is
    # correct for the real layout, so the two disagreed by a systematic ~0.005 per element.
    half = args.dim // 2
    position = torch.arange(args.tokens, dtype=torch.float32)[:, None]
    inverse = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    angle = position * inverse[None, :]
    cos_half, sin_half = angle.cos(), angle.sin()
    # Doubled for the reference; the halves are what our implementation consumes.
    cos = torch.cat((cos_half, cos_half), dim=-1)[None]
    sin = torch.cat((sin_half, sin_half), dim=-1)[None]

    output = memory(hidden, (cos, sin))
    if isinstance(output, tuple):
        output = output[0]

    def flat(tensor):
        return [float(x) for x in tensor.detach().reshape(-1).tolist()]

    # Stage-by-stage intermediates, so a mismatch names the stage instead of the layer. The
    # submethods are called exactly as forward() calls them.
    u = memory.input_norm(hidden)
    query, admission, gate = memory._query_scores(u, (cos, sin))
    angles = tuple(x.reshape(-1, x.shape[-1]) for x in (cos, sin))
    written_key, written_value = memory._write_kv(u.reshape(-1, args.hidden), angles)
    stages = {
        "u": flat(u),
        "query": flat(query),
        "admission": flat(admission),
        "gate": flat(gate),
        "writtenKey": flat(written_key),
        "writtenValue": flat(written_value),
    }


    # Norm weights are emitted in GGUF convention, not PyTorch's.
    #
    # Qwen3_5RMSNorm -- which EdgeRock uses for input_norm, q_norm and k_norm -- is
    # `normalized * (1.0 + weight)` with a zero-initialised weight, the Gemma convention. Our
    # runtime's TensorOps.rmsNorm is `normalized * weight`, the Llama one, because the GGUF
    # conversion folds the +1 into the stored tensor. That is not a guess: the qwen35 decoder
    # shares that norm and is qualified, and three unsloth_qwen3_5_* models passed their
    # library-default correctness smokes at correctAnswerRate 1.0 on 2026-10-29, which norms off
    # by a whole unit could not do.
    #
    # So a fixture carrying raw PyTorch norm weights would be comparing two different conventions
    # and the Java side would read zeros. The first run of this script did exactly that: every
    # norm weight was 0.0, so `normalized * weight` annihilated the input and our layer returned
    # zeros while the reference returned real values.
    norm_parameters = {"input_norm.weight", "q_norm.weight", "k_norm.weight"}
    weights = {}
    for name, parameter in memory.named_parameters():
        values = flat(parameter)
        if name in norm_parameters:
            values = [1.0 + v for v in values]
        weights[name] = values

    fixture = {
        "source": "IndependentGQAMemory from rockmatrix-ai/EdgeRock modeling_edgerock.py",
        "normConvention": "GGUF: norm weights emitted as 1 + PyTorch weight",
        "note": "reference class at small dimensions; seeded weights; chunk crossed several times",
        "seed": args.seed,
        "config": {
            "hidden": args.hidden,
            "heads": args.heads,
            "groups": args.groups,
            "dim": args.dim,
            "capacity": args.capacity,
            "chunk": args.chunk,
            "rank": args.rank,
            "writer": args.writer,
            "window": args.window,
            "tokens": args.tokens,
            "ropeWidth": half,
            "rmsNormEps": 1.0e-6,
        },
        "hidden": flat(hidden),
        "cos": flat(cos_half),
        "sin": flat(sin_half),
        "cosDoubled": flat(cos),
        "output": flat(output),
        "stages": stages,
        "weights": weights,
        "weightShapes": {name: list(p.shape) for name, p in memory.named_parameters()},
    }
    args.out.write_text(json.dumps(fixture, indent=1))
    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size / 1024:.1f} KB)")
    print(f"  tokens={args.tokens} chunk={args.chunk} -> commits={args.tokens // args.chunk}")
    print(f"  output[0:4] = {fixture['output'][:4]}")
    nonzero = sum(1 for v in fixture["output"] if abs(v) > 1e-9)
    print(f"  non-zero outputs: {nonzero} of {len(fixture['output'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
