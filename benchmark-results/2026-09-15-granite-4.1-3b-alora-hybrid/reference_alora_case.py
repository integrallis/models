#!/usr/bin/env python3
"""Run one qualification-window case through the IBM reference implementation of an activated
LoRA (Transformers + PEFT) and print the greedy continuation.

This is a measurement oracle only, never in the product execution path. It separates "the
adapter itself produces this output" from "the Java runtime deviates from the reference" for a
single case; the prompt is rendered exactly as ``render_oracle_prompt.py`` renders it for the
specialist arm (chat template with ``documents``, generation prompt appended), so the invocation
sequence ``<|start_of_role|>assistant<|end_of_role|>`` is the last three prompt tokens and PEFT
activates the adapter there.

Usage:
  reference_alora_case.py --base ibm-granite/granite-4.1-3b --base-revision <sha> \
      --adapter <dir with adapter_config.json> --window qualification-window.json \
      --suite squad-v2-dev --case-id <id> [--max-new-tokens 8] [--output report.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


def find_case(window_path: Path, suite_name: str, case_id: str) -> dict:
    window = json.loads(window_path.read_text())
    suite = next(s for s in window["suites"] if s["name"] == suite_name)
    return next(c for c in suite["cases"] if c["id"] == case_id)


def render(tokenizer, case: dict) -> str:
    messages = [{"role": m["role"], "content": m["text"]} for m in case["messages"]]
    documents = [{"doc_id": d["doc_id"], "text": d["text"]} for d in case["documents"]]
    return tokenizer.apply_chat_template(
        messages, documents=documents, add_generation_prompt=True, tokenize=False
    )


GGUF_TO_HF = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
BLOCK_TO_HF = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def unpermute(weights, n_head: int):
    """Inverse of llama.cpp's convert_hf_to_gguf permute for q_proj / k_proj rows."""
    rows, cols = weights.shape
    return (
        weights.reshape(n_head, rows // n_head // 2, 2, cols).swapaxes(1, 2).reshape(rows, cols)
    )


def patch_with_gguf(model, gguf_path: Path) -> dict:
    """Replace the model's weights with the dequantized tensors of a llama.cpp GGUF file.

    Returns per-tensor relative error against the weights being replaced, which is the sanity
    check that the name mapping and the q/k un-permutation are right (Q4_K_M error is a few
    percent; a wrong permutation is ~100%).
    """
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(gguf_path))
    n_head = model.config.num_attention_heads
    n_head_kv = model.config.num_key_value_heads
    state = model.state_dict()
    errors = {}
    replaced = 0
    for tensor in reader.tensors:
        name = tensor.name
        if name in GGUF_TO_HF:
            target = GGUF_TO_HF[name]
        elif name.startswith("blk."):
            _, layer, rest = name.split(".", 2)
            key, suffix = rest.rsplit(".", 1)
            if key not in BLOCK_TO_HF:
                raise KeyError(f"unmapped GGUF tensor {name}")
            target = f"model.layers.{layer}.{BLOCK_TO_HF[key]}.{suffix}"
        else:
            raise KeyError(f"unmapped GGUF tensor {name}")
        if target not in state:
            if target == "lm_head.weight" and model.config.tie_word_embeddings:
                continue
            raise KeyError(f"{name} -> {target} is not a model parameter")
        array = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
        shape = tuple(int(d) for d in reversed(tensor.shape))
        array = array.reshape(shape)
        if target.endswith("q_proj.weight"):
            array = unpermute(array, n_head)
        elif target.endswith("k_proj.weight"):
            array = unpermute(array, n_head_kv)
        current = state[target]
        new = torch.from_numpy(np.ascontiguousarray(array)).to(current.dtype)
        if new.shape != current.shape:
            raise ValueError(f"{name}: GGUF shape {tuple(new.shape)} != {tuple(current.shape)}")
        denominator = float(current.float().norm()) or 1.0
        errors[target] = float((new.float() - current.float()).norm()) / denominator
        with torch.no_grad():
            current.copy_(new)
        replaced += 1
    if model.config.tie_word_embeddings and "lm_head.weight" in state:
        with torch.no_grad():
            state["lm_head.weight"].copy_(state["model.embed_tokens.weight"])
    return {"replaced": replaced, "maxRelativeError": max(errors.values()), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gguf-weights",
        type=Path,
        help="replace the base weights with this llama.cpp GGUF file's dequantized tensors",
    )
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
    case = find_case(args.window, args.suite, args.case_id)
    text = render(tokenizer, case)
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"]
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
    invocation = adapter_config.get("alora_invocation_tokens")
    tail = input_ids[0, -len(invocation):].tolist() if invocation else None
    if invocation and tail != invocation:
        print(f"prompt does not end with the invocation tokens: {tail} != {invocation}", file=sys.stderr)
        sys.exit(2)

    started = time.time()
    base = AutoModelForCausalLM.from_pretrained(args.base, revision=args.base_revision, torch_dtype=dtype)
    base.eval()
    gguf_patch = None
    if args.gguf_weights:
        gguf_patch = patch_with_gguf(base, args.gguf_weights)
        worst = sorted(gguf_patch["errors"].items(), key=lambda item: -item[1])[:5]
        print(
            f"patched {gguf_patch['replaced']} tensors from {args.gguf_weights.name}; "
            f"max relative error {gguf_patch['maxRelativeError']:.4f}; worst {worst}",
            file=sys.stderr,
        )
    with torch.no_grad():
        base_logits = base(input_ids=input_ids).logits[0, -1]
    base_top = torch.topk(base_logits, 5)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    with torch.no_grad():
        outcome = model.generate(
            input_ids=input_ids,
            attention_mask=encoded["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated = outcome.sequences
        first_logits = model(input_ids=input_ids).logits[0, -1]
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    steps = []
    for step, scores in enumerate(outcome.scores):
        step_top = torch.topk(scores[0], 5)
        steps.append(
            {
                "step": step,
                "chosen": new_tokens[step],
                "top5": [
                    {"token": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
                    for v, i in zip(step_top.values, step_top.indices)
                ],
            }
        )
    top = torch.topk(first_logits, 5)
    report = {
        "caseId": case["id"],
        "label": case["label"],
        "base": args.base,
        "baseRevision": args.base_revision,
        "adapterConfigSha256": hashlib.sha256((args.adapter / "adapter_config.json").read_bytes()).hexdigest(),
        "adapterWeightsSha256": hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "promptSha256": hashlib.sha256(text.encode()).hexdigest(),
        "promptTokens": input_ids.shape[1],
        "invocationTokens": invocation,
        "dtype": args.dtype,
        "baseWeights": (
            {
                "gguf": args.gguf_weights.name,
                "ggufSha256": hashlib.sha256(args.gguf_weights.read_bytes()).hexdigest(),
                "replacedTensors": gguf_patch["replaced"],
                "maxRelativeErrorVsUnquantized": gguf_patch["maxRelativeError"],
            }
            if gguf_patch
            else {"unquantized": True}
        ),
        "generatedTokens": new_tokens,
        "generatedText": tokenizer.decode(new_tokens, skip_special_tokens=False),
        "specialistSteps": steps,
        "specialistFirstTokenTop5": [
            {"token": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
            for v, i in zip(top.values, top.indices)
        ],
        "baseFirstTokenTop5": [
            {"token": int(i), "text": tokenizer.decode([int(i)]), "logit": float(v)}
            for v, i in zip(base_top.values, base_top.indices)
        ],
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "peft": __import__("peft").__version__,
            "transformers": __import__("transformers").__version__,
            "machine": platform.machine(),
        },
        "seconds": round(time.time() - started, 1),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
