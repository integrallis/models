#!/usr/bin/env python3
"""Third-party cross-check of the fusion rule with Hugging Face transformers.

A documented cross-check only: it is NEVER in the measured path, and no study result comes from
it. It lets someone outside Integrallis confirm what the Java `logit-fusion` runner computes per
step, on one GSM8K item, with an implementation that shares no code with ours.

It loads the bf16 safetensors checkpoints Qwen/Qwen3-0.6B and Qwen/Qwen3-1.7B at pinned revisions,
while the study runs Q8_0 GGUF files. Quantization changes logits, so token-for-token identity with
the Java runs is NOT expected; what should agree is the rule (log_softmax per member, then the
weighted poe sum or the mixture logsumexp, then greedy argmax) and qualitative behaviour.

Usage: pip install -r requirements-hf.txt
       python3 hf_reference_fusion.py --data gsm8k-test.jsonl --item-id gsm8k-test-0000 \
           [--rule poe|mixture] [--weights 0.5,0.5] [--max-tokens 256]
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MEMBERS = [("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
           ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")]
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def fuse(rule, log_probs, weights):
    if rule == "poe":
        return sum(w * lp for w, lp in zip(weights, log_probs))
    terms = torch.stack([torch.log(torch.tensor(w, dtype=torch.float64)) + lp
                         for w, lp in zip(weights, log_probs) if w > 0])
    return torch.logsumexp(terms, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--rule", choices=("poe", "mixture"), default="poe")
    parser.add_argument("--weights", default="0.5,0.5")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != len(MEMBERS) or abs(sum(weights) - 1.0) > 1e-9:
        raise SystemExit("--weights must list one weight per member and sum to 1")
    item = next(json.loads(l) for l in open(args.data, encoding="utf-8")
                if json.loads(l)["id"] == args.item_id)

    tokenizers = [AutoTokenizer.from_pretrained(r, revision=v) for r, v in MEMBERS]
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise SystemExit("tokenizer vocabularies differ; refusing to fuse (gate G0)")
    tokenizer = tokenizers[0]
    models = [AutoModelForCausalLM.from_pretrained(r, revision=v, dtype=torch.bfloat16).eval()
              for r, v in MEMBERS]
    messages = [{"role": "user", "content": item["question"] + "\n" + INSTRUCTION}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    stop_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}

    caches, step_input, generated, agreements = [None] * len(models), input_ids, [], 0
    with torch.no_grad():
        for step in range(args.max_tokens):
            log_probs = []
            for index, model in enumerate(models):
                out = model(input_ids=step_input, past_key_values=caches[index], use_cache=True)
                caches[index] = out.past_key_values
                log_probs.append(torch.log_softmax(out.logits[0, -1].double(), dim=-1))
            token = int(torch.argmax(fuse(args.rule, log_probs, weights)))
            argmaxes = [int(torch.argmax(lp)) for lp in log_probs]
            agreements += int(len(set(argmaxes)) == 1)
            print(f"step={step:4d} token={token:6d} {tokenizer.decode([token])!r:>16} "
                  f"memberArgmax={argmaxes} matches={[a == token for a in argmaxes]}")
            generated.append(token)
            if token in stop_ids:
                break
            step_input = torch.tensor([[token]])

    print("fused text:", tokenizer.decode(generated, skip_special_tokens=True))
    print(f"member argmax agreement: {agreements}/{len(generated)} steps; gold answer: {item['answer']}")


if __name__ == "__main__":
    main()
